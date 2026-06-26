use nerva_core::types::{DeviceOrdinal, ExecutionOwner, MemoryTier, NervaError, Result};
use nerva_ledger::types::{
    CandidateCost, ExecutionDecision, LedgerEvent, LedgerEventKind, MetricSource, TokenLedger,
};

use crate::common::hash::hash_f32s;
use crate::common::math::dot;
use crate::common::shape::TransformerBlockShape;
use crate::common::validate::require_len;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KvAttentionBlock<'a> {
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub token_count: usize,
    pub tier: MemoryTier,
}

impl<'a> KvAttentionBlock<'a> {
    pub const fn new(
        keys: &'a [f32],
        values: &'a [f32],
        token_count: usize,
        tier: MemoryTier,
    ) -> Self {
        Self {
            keys,
            values,
            token_count,
            tier,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BlockwiseAttentionScratch {
    shape: TransformerBlockShape,
    local_output: Vec<f32>,
    global_m: Vec<f32>,
    global_l: Vec<f32>,
}

impl BlockwiseAttentionScratch {
    pub fn new(shape: TransformerBlockShape) -> Result<Self> {
        shape.validate()?;
        Ok(Self {
            shape,
            local_output: vec![0.0; shape.hidden],
            global_m: vec![f32::NEG_INFINITY; shape.heads],
            global_l: vec![0.0; shape.heads],
        })
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

    fn require_shape(&self, shape: TransformerBlockShape) -> Result<()> {
        if self.shape == shape {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "blockwise attention scratch shape does not match".to_string(),
            })
        }
    }
}

pub fn exact_blockwise_attention_into(
    shape: TransformerBlockShape,
    query: &[f32],
    blocks: &[KvAttentionBlock<'_>],
    scratch: &mut BlockwiseAttentionScratch,
    output: &mut [f32],
    ledger: &mut TokenLedger,
) -> Result<()> {
    shape.validate()?;
    scratch.require_shape(shape)?;
    require_len("attention query", query.len(), shape.hidden)?;
    require_len("attention output", output.len(), shape.hidden)?;

    scratch.local_output.fill(0.0);
    scratch.global_m.fill(f32::NEG_INFINITY);
    scratch.global_l.fill(0.0);
    output.fill(0.0);

    let head_dim = shape.head_dim();
    let scale = (head_dim as f32).sqrt().recip();
    let mut total_tokens = 0usize;

    for block in blocks {
        let values = block.token_count.checked_mul(shape.hidden).ok_or_else(|| {
            NervaError::InvalidArgument {
                reason: "KV attention block token count overflow".to_string(),
            }
        })?;
        require_len("KV block keys", block.keys.len(), values)?;
        require_len("KV block values", block.values.len(), values)?;
        if block.token_count == 0 {
            continue;
        }
        total_tokens = total_tokens.checked_add(block.token_count).ok_or_else(|| {
            NervaError::InvalidArgument {
                reason: "KV attention total token count overflow".to_string(),
            }
        })?;
        record_attention_block_event(shape, block, ledger);

        for head in 0..shape.heads {
            let head_start = head * head_dim;
            let head_end = head_start + head_dim;
            scratch.local_output[head_start..head_end].fill(0.0);
            let mut local_m = f32::NEG_INFINITY;
            let mut local_l = 0.0f32;

            for token_index in 0..block.token_count {
                let token_start = token_index * shape.hidden + head_start;
                let token_end = token_start + head_dim;
                let score = dot(
                    &query[head_start..head_end],
                    &block.keys[token_start..token_end],
                ) * scale;
                let next_m = local_m.max(score);
                let old_scale = if local_l == 0.0 {
                    0.0
                } else {
                    (local_m - next_m).exp()
                };
                let new_scale = (score - next_m).exp();
                for (local, value) in scratch.local_output[head_start..head_end]
                    .iter_mut()
                    .zip(block.values[token_start..token_end].iter().copied())
                {
                    *local = *local * old_scale + value * new_scale;
                }
                local_l = local_l * old_scale + new_scale;
                local_m = next_m;
            }

            let global_m = scratch.global_m[head];
            let global_l = scratch.global_l[head];
            let next_m = global_m.max(local_m);
            let global_scale = if global_l == 0.0 {
                0.0
            } else {
                (global_m - next_m).exp()
            };
            let local_scale = (local_m - next_m).exp();
            for (global, local) in output[head_start..head_end]
                .iter_mut()
                .zip(scratch.local_output[head_start..head_end].iter().copied())
            {
                *global = *global * global_scale + local * local_scale;
            }
            scratch.global_l[head] = global_l * global_scale + local_l * local_scale;
            scratch.global_m[head] = next_m;
        }
    }

    if total_tokens == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "blockwise attention requires at least one KV token".to_string(),
        });
    }

    for head in 0..shape.heads {
        let normalizer = scratch.global_l[head];
        if normalizer == 0.0 || !normalizer.is_finite() {
            return Err(NervaError::InvalidArgument {
                reason: "blockwise attention produced invalid normalizer".to_string(),
            });
        }
        let head_start = head * head_dim;
        let head_end = head_start + head_dim;
        for value in &mut output[head_start..head_end] {
            *value /= normalizer;
        }
    }

    Ok(())
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BlockwiseAttentionSmokeStatus {
    Ok,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockwiseAttentionSmokeSummary {
    pub status: BlockwiseAttentionSmokeStatus,
    pub hidden: usize,
    pub heads: usize,
    pub blocks: usize,
    pub tokens: usize,
    pub output: [f32; 2],
    pub output_hash: u64,
    pub cpu_block_events: u64,
    pub device_block_events: u64,
    pub hot_path_allocations: u64,
}

impl BlockwiseAttentionSmokeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            BlockwiseAttentionSmokeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"heads\":{},\"blocks\":{},\"tokens\":{},\"output\":[{},{}],\"output_hash\":{},\"cpu_block_events\":{},\"device_block_events\":{},\"hot_path_allocations\":{}}}",
            status,
            self.hidden,
            self.heads,
            self.blocks,
            self.tokens,
            self.output[0],
            self.output[1],
            self.output_hash,
            self.cpu_block_events,
            self.device_block_events,
            self.hot_path_allocations,
        )
    }
}
pub fn blockwise_attention_smoke() -> Result<BlockwiseAttentionSmokeSummary> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let query = [1.0, 0.25];
    let dram_keys = [0.2, 0.0, 0.0, 0.4];
    let dram_values = [1.0, 0.0, 0.5, 0.5];
    let vram_keys = [0.5, 0.1, -0.2, 0.3];
    let vram_values = [0.0, 1.0, 2.0, -1.0];
    let blocks = [
        KvAttentionBlock::new(&dram_keys, &dram_values, 2, MemoryTier::Dram),
        KvAttentionBlock::new(&vram_keys, &vram_values, 2, MemoryTier::Vram),
    ];
    let mut scratch = BlockwiseAttentionScratch::new(shape)?;
    let mut output = [0.0; 2];
    let mut ledger = TokenLedger::new(0);
    exact_blockwise_attention_into(
        shape,
        &query,
        &blocks,
        &mut scratch,
        &mut output,
        &mut ledger,
    )?;

    Ok(BlockwiseAttentionSmokeSummary {
        status: BlockwiseAttentionSmokeStatus::Ok,
        hidden: shape.hidden,
        heads: shape.heads,
        blocks: blocks.len(),
        tokens: blocks.iter().map(|block| block.token_count).sum(),
        output,
        output_hash: hash_f32s(&output),
        cpu_block_events: ledger.event_count(LedgerEventKind::CpuActivity),
        device_block_events: ledger.event_count(LedgerEventKind::DeviceActivity),
        hot_path_allocations: ledger.hot_path_allocations,
    })
}

fn record_attention_block_event(
    shape: TransformerBlockShape,
    block: &KvAttentionBlock<'_>,
    ledger: &mut TokenLedger,
) {
    let (kind, executor_selected, reason) = match block.tier {
        MemoryTier::Vram | MemoryTier::SharedHbmOrLpddr => (
            LedgerEventKind::DeviceActivity,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            "hot KV block is already device resident",
        ),
        MemoryTier::PinnedDram | MemoryTier::Dram | MemoryTier::Cxl | MemoryTier::Disk => (
            LedgerEventKind::CpuActivity,
            ExecutionOwner::Cpu,
            "warm KV block is cheaper to compute near than stage",
        ),
    };
    let label = match kind {
        LedgerEventKind::DeviceActivity => "attention_hot_kv_block",
        LedgerEventKind::CpuActivity => "attention_warm_kv_block",
        _ => "attention_kv_block",
    };
    let latency_ns = block.token_count as u64;
    ledger.record_execution_decision(ExecutionDecision {
        operation: "blockwise_attention",
        executor_selected,
        candidate_costs: vec![
            CandidateCost::estimated("compute-near-current-tier", latency_ns),
            CandidateCost::estimated("stage-to-gpu", latency_ns + 2),
        ],
        reason,
        predicted_visible_ns: latency_ns,
        actual_visible_ns: Some(latency_ns),
        metric_source: MetricSource::EstimatedModel,
    });
    ledger.record(LedgerEvent {
        kind,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(block.tier),
        to_tier: Some(block.tier),
        bytes: block.token_count * shape.hidden * core::mem::size_of::<f32>() * 2,
        latency_ns,
        label,
    });
}
