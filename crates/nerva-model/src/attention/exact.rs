use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::DeviceOrdinal;
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::TokenLedger;

use crate::attention::block::KvAttentionBlock;
use crate::attention::scratch::BlockwiseAttentionScratch;
use crate::common::math::dot;
use crate::common::shape::TransformerBlockShape;
use crate::common::validate::require_len;

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
