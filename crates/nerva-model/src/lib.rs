#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use nerva_core::{BlockKind, DType, MemoryTier, NervaError, Result, TokenId};
use nerva_ledger::{LedgerEvent, LedgerEventKind, TokenLedger};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransformerBlockShape {
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
}

impl TransformerBlockShape {
    pub const fn new(hidden: usize, heads: usize, intermediate: usize) -> Self {
        Self {
            hidden,
            heads,
            intermediate,
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.hidden == 0 || self.heads == 0 || self.intermediate == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "transformer block dimensions must be non-zero".to_string(),
            });
        }
        if !self.hidden.is_multiple_of(self.heads) {
            return Err(NervaError::InvalidArgument {
                reason: "hidden size must be divisible by head count".to_string(),
            });
        }
        Ok(())
    }

    pub const fn head_dim(self) -> usize {
        self.hidden / self.heads
    }
}

#[derive(Clone, Debug)]
pub struct ReferenceTransformerBlock {
    shape: TransformerBlockShape,
    rms_attn_weight: Vec<f32>,
    rms_mlp_weight: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
    rms_eps: f32,
}

impl ReferenceTransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shape: TransformerBlockShape,
        rms_attn_weight: Vec<f32>,
        rms_mlp_weight: Vec<f32>,
        w_q: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
        w_o: Vec<f32>,
        w_gate: Vec<f32>,
        w_up: Vec<f32>,
        w_down: Vec<f32>,
        rms_eps: f32,
    ) -> Result<Self> {
        shape.validate()?;
        require_len("rms_attn_weight", rms_attn_weight.len(), shape.hidden)?;
        require_len("rms_mlp_weight", rms_mlp_weight.len(), shape.hidden)?;
        require_len("w_q", w_q.len(), shape.hidden * shape.hidden)?;
        require_len("w_k", w_k.len(), shape.hidden * shape.hidden)?;
        require_len("w_v", w_v.len(), shape.hidden * shape.hidden)?;
        require_len("w_o", w_o.len(), shape.hidden * shape.hidden)?;
        require_len("w_gate", w_gate.len(), shape.intermediate * shape.hidden)?;
        require_len("w_up", w_up.len(), shape.intermediate * shape.hidden)?;
        require_len("w_down", w_down.len(), shape.hidden * shape.intermediate)?;
        if rms_eps <= 0.0 || !rms_eps.is_finite() {
            return Err(NervaError::InvalidArgument {
                reason: "rms epsilon must be positive and finite".to_string(),
            });
        }
        Ok(Self {
            shape,
            rms_attn_weight,
            rms_mlp_weight,
            w_q,
            w_k,
            w_v,
            w_o,
            w_gate,
            w_up,
            w_down,
            rms_eps,
        })
    }

    pub fn zero_for_shape(shape: TransformerBlockShape) -> Result<Self> {
        shape.validate()?;
        Self::new(
            shape,
            vec![1.0; shape.hidden],
            vec![1.0; shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.intermediate * shape.hidden],
            vec![0.0; shape.intermediate * shape.hidden],
            vec![0.0; shape.hidden * shape.intermediate],
            1e-5,
        )
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

    pub fn forward_into(
        &self,
        input: &[f32],
        scratch: &mut TransformerBlockScratch,
        output: &mut [f32],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let _ = ledger;
        let shape = self.shape;
        require_len("input", input.len(), shape.hidden)?;
        require_len("output", output.len(), shape.hidden)?;
        scratch.require_shape(shape)?;

        rms_norm_into(
            input,
            &self.rms_attn_weight,
            self.rms_eps,
            &mut scratch.attn_norm,
        );
        mat_vec_row_major(&self.w_q, &scratch.attn_norm, &mut scratch.q);
        mat_vec_row_major(&self.w_k, &scratch.attn_norm, &mut scratch.k);
        mat_vec_row_major(&self.w_v, &scratch.attn_norm, &mut scratch.v);

        single_token_attention(shape, &scratch.q, &scratch.k, &scratch.v, &mut scratch.attn);
        mat_vec_row_major(&self.w_o, &scratch.attn, output);
        for (out, residual) in output.iter_mut().zip(input.iter().copied()) {
            *out += residual;
        }

        rms_norm_into(
            output,
            &self.rms_mlp_weight,
            self.rms_eps,
            &mut scratch.mlp_norm,
        );
        mat_vec_row_major(&self.w_gate, &scratch.mlp_norm, &mut scratch.gate);
        mat_vec_row_major(&self.w_up, &scratch.mlp_norm, &mut scratch.up);
        for ((ff, gate), up) in scratch
            .ff
            .iter_mut()
            .zip(scratch.gate.iter().copied())
            .zip(scratch.up.iter().copied())
        {
            *ff = silu(gate) * up;
        }
        mat_vec_row_major(&self.w_down, &scratch.ff, &mut scratch.down);
        for (out, mlp) in output.iter_mut().zip(scratch.down.iter().copied()) {
            *out += mlp;
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TransformerBlockScratch {
    shape: TransformerBlockShape,
    attn_norm: Vec<f32>,
    mlp_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    ff: Vec<f32>,
    down: Vec<f32>,
}

impl TransformerBlockScratch {
    pub fn new(shape: TransformerBlockShape) -> Result<Self> {
        shape.validate()?;
        Ok(Self {
            shape,
            attn_norm: vec![0.0; shape.hidden],
            mlp_norm: vec![0.0; shape.hidden],
            q: vec![0.0; shape.hidden],
            k: vec![0.0; shape.hidden],
            v: vec![0.0; shape.hidden],
            attn: vec![0.0; shape.hidden],
            gate: vec![0.0; shape.intermediate],
            up: vec![0.0; shape.intermediate],
            ff: vec![0.0; shape.intermediate],
            down: vec![0.0; shape.hidden],
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
                reason: "transformer block scratch shape does not match block shape".to_string(),
            })
        }
    }
}

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
pub struct ModelBlockContract {
    pub block_kind: BlockKind,
    pub weight_dtype: DType,
    pub activation_dtype: DType,
    pub weight_tier: MemoryTier,
    pub activation_tier: MemoryTier,
}

impl ModelBlockContract {
    pub const fn reference_f32() -> Self {
        Self {
            block_kind: BlockKind::Weight,
            weight_dtype: DType::F32,
            activation_dtype: DType::F32,
            weight_tier: MemoryTier::Dram,
            activation_tier: MemoryTier::Dram,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReferenceBlockSmokeStatus {
    Ok,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReferenceBlockSmokeSummary {
    pub status: ReferenceBlockSmokeStatus,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub output: [f32; 2],
    pub output_hash: u64,
    pub hot_path_allocations: u64,
}

impl ReferenceBlockSmokeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            ReferenceBlockSmokeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"heads\":{},\"intermediate\":{},\"output\":[{},{}],\"output_hash\":{},\"hot_path_allocations\":{}}}",
            status,
            self.hidden,
            self.heads,
            self.intermediate,
            self.output[0],
            self.output[1],
            self.output_hash,
            self.hot_path_allocations,
        )
    }
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

pub fn reference_block_smoke() -> Result<ReferenceBlockSmokeSummary> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let block = ReferenceTransformerBlock::new(
        shape,
        vec![1.0, 1.0],
        vec![1.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.5, 0.0, 0.0, 0.5],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        1e-5,
    )?;
    let input = [1.0, 2.0];
    let mut scratch = TransformerBlockScratch::new(shape)?;
    let mut output = [0.0; 2];
    let mut ledger = TokenLedger::new(0);
    block.forward_into(&input, &mut scratch, &mut output, &mut ledger)?;
    Ok(ReferenceBlockSmokeSummary {
        status: ReferenceBlockSmokeStatus::Ok,
        hidden: shape.hidden,
        heads: shape.heads,
        intermediate: shape.intermediate,
        output,
        output_hash: hash_f32s(&output),
        hot_path_allocations: ledger.hot_path_allocations,
    })
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

#[derive(Clone, Debug)]
pub struct TinyGreedyModel {
    vocab_size: usize,
    shape: TransformerBlockShape,
    block: ReferenceTransformerBlock,
    embeddings: Vec<f32>,
    lm_head: Vec<f32>,
}

impl TinyGreedyModel {
    pub fn new(
        vocab_size: usize,
        block: ReferenceTransformerBlock,
        embeddings: Vec<f32>,
        lm_head: Vec<f32>,
    ) -> Result<Self> {
        if vocab_size == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "tiny model vocabulary must be non-zero".to_string(),
            });
        }
        let shape = block.shape();
        require_len("embeddings", embeddings.len(), vocab_size * shape.hidden)?;
        require_len("lm_head", lm_head.len(), vocab_size * shape.hidden)?;
        Ok(Self {
            vocab_size,
            shape,
            block,
            embeddings,
            lm_head,
        })
    }

    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

    pub fn decode_greedy(
        &self,
        seed_token: TokenId,
        steps: usize,
        scratch: &mut TinyGreedyDecodeScratch,
    ) -> Result<TinyGreedyDecodeOutput> {
        if steps == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "tiny greedy decode steps must be non-zero".to_string(),
            });
        }
        scratch.require_shape(self.shape, self.vocab_size)?;
        require_token_in_vocab(seed_token, self.vocab_size)?;

        let mut current_token = seed_token;
        let mut tokens = Vec::with_capacity(steps);
        let mut ledgers = Vec::with_capacity(steps);
        for step in 0..steps {
            copy_embedding_row(
                &self.embeddings,
                self.shape.hidden,
                current_token,
                &mut scratch.hidden,
            )?;
            let mut ledger = TokenLedger::new(step as u64);
            self.block.forward_into(
                &scratch.hidden,
                &mut scratch.block_scratch,
                &mut scratch.block_output,
                &mut ledger,
            )?;
            mat_vec_row_major(&self.lm_head, &scratch.block_output, &mut scratch.logits);
            let next_token = greedy_argmax(&scratch.logits)?;
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::DeviceActivity,
                block_id: None,
                from_tier: Some(MemoryTier::Dram),
                to_tier: Some(MemoryTier::Dram),
                bytes: (self.shape.hidden + self.vocab_size) * core::mem::size_of::<f32>(),
                latency_ns: 1,
                label: "tiny_greedy_decode_reference",
            });
            ledger.require_zero_hot_path_allocations()?;
            tokens.push(next_token);
            ledgers.push(ledger);
            current_token = next_token;
        }

        Ok(TinyGreedyDecodeOutput { tokens, ledgers })
    }
}

#[derive(Clone, Debug)]
pub struct TinyGreedyDecodeScratch {
    shape: TransformerBlockShape,
    vocab_size: usize,
    block_scratch: TransformerBlockScratch,
    hidden: Vec<f32>,
    block_output: Vec<f32>,
    logits: Vec<f32>,
}

impl TinyGreedyDecodeScratch {
    pub fn new(shape: TransformerBlockShape, vocab_size: usize) -> Result<Self> {
        shape.validate()?;
        if vocab_size == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "tiny greedy scratch vocabulary must be non-zero".to_string(),
            });
        }
        Ok(Self {
            shape,
            vocab_size,
            block_scratch: TransformerBlockScratch::new(shape)?,
            hidden: vec![0.0; shape.hidden],
            block_output: vec![0.0; shape.hidden],
            logits: vec![0.0; vocab_size],
        })
    }

    fn require_shape(&self, shape: TransformerBlockShape, vocab_size: usize) -> Result<()> {
        if self.shape == shape && self.vocab_size == vocab_size {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "tiny greedy scratch shape does not match model shape".to_string(),
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TinyGreedyDecodeOutput {
    pub tokens: Vec<TokenId>,
    pub ledgers: Vec<TokenLedger>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TinyGreedyDecodeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TinyGreedyDecodeSummary {
    pub status: TinyGreedyDecodeStatus,
    pub seed_token: TokenId,
    pub steps: usize,
    pub vocab_size: usize,
    pub tokens: Vec<TokenId>,
    pub expected_tokens: Vec<TokenId>,
    pub parity: bool,
    pub ledger_count: u64,
    pub device_events: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub output_hash: u64,
}

impl TinyGreedyDecodeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            TinyGreedyDecodeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"seed_token\":{},\"steps\":{},\"vocab_size\":{},\"tokens\":{},\"expected_tokens\":{},\"parity\":{},\"ledger_count\":{},\"device_events\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"output_hash\":{}}}",
            status,
            self.seed_token.0,
            self.steps,
            self.vocab_size,
            token_ids_to_json(&self.tokens),
            token_ids_to_json(&self.expected_tokens),
            self.parity,
            self.ledger_count,
            self.device_events,
            self.total_latency_ns,
            self.hot_path_allocations,
            self.output_hash,
        )
    }
}

pub fn tiny_greedy_decode_smoke(steps: usize) -> Result<TinyGreedyDecodeSummary> {
    let model = tiny_cycle_model()?;
    let seed_token = TokenId(0);
    let mut scratch = TinyGreedyDecodeScratch::new(model.shape(), model.vocab_size())?;
    let output = model.decode_greedy(seed_token, steps, &mut scratch)?;
    let expected_tokens = expected_cycle(seed_token, steps, model.vocab_size());
    let parity = output.tokens == expected_tokens;
    if !parity {
        return Err(NervaError::InvalidArgument {
            reason: "tiny greedy decode token parity failed".to_string(),
        });
    }
    let hot_path_allocations = output
        .ledgers
        .iter()
        .map(|ledger| ledger.hot_path_allocations)
        .sum();
    let device_events = output
        .ledgers
        .iter()
        .map(|ledger| ledger.event_count(LedgerEventKind::DeviceActivity))
        .sum();
    let total_latency_ns = output
        .ledgers
        .iter()
        .map(TokenLedger::total_latency_ns)
        .sum();

    Ok(TinyGreedyDecodeSummary {
        status: TinyGreedyDecodeStatus::Ok,
        seed_token,
        steps,
        vocab_size: model.vocab_size(),
        output_hash: hash_tokens(&output.tokens),
        tokens: output.tokens,
        expected_tokens,
        parity,
        ledger_count: output.ledgers.len() as u64,
        device_events,
        total_latency_ns,
        hot_path_allocations,
    })
}

fn tiny_cycle_model() -> Result<TinyGreedyModel> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let block = ReferenceTransformerBlock::zero_for_shape(shape)?;
    TinyGreedyModel::new(
        4,
        block,
        vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
        vec![0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
    )
}

fn require_len(label: &'static str, got: usize, expected: usize) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!("{label} length {got} does not match expected {expected}"),
        })
    }
}

fn require_token_in_vocab(token: TokenId, vocab_size: usize) -> Result<()> {
    if token.0 as usize >= vocab_size {
        Err(NervaError::InvalidArgument {
            reason: format!(
                "token id {} is outside tiny model vocabulary {}",
                token.0, vocab_size
            ),
        })
    } else {
        Ok(())
    }
}

fn copy_embedding_row(
    embeddings: &[f32],
    hidden: usize,
    token: TokenId,
    output: &mut [f32],
) -> Result<()> {
    require_token_in_vocab(token, embeddings.len() / hidden)?;
    let start = token.0 as usize * hidden;
    let end = start + hidden;
    output.copy_from_slice(&embeddings[start..end]);
    Ok(())
}

fn rms_norm_into(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    for ((out, value), weight) in output
        .iter_mut()
        .zip(input.iter().copied())
        .zip(weight.iter().copied())
    {
        *out = value * scale * weight;
    }
}

fn mat_vec_row_major(matrix: &[f32], input: &[f32], output: &mut [f32]) {
    let cols = input.len();
    for (row, out) in matrix.chunks_exact(cols).zip(output.iter_mut()) {
        *out = row
            .iter()
            .zip(input.iter())
            .map(|(weight, value)| weight * value)
            .sum();
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn single_token_attention(
    shape: TransformerBlockShape,
    _q: &[f32],
    _k: &[f32],
    v: &[f32],
    output: &mut [f32],
) {
    let head_dim = shape.head_dim();
    for head in 0..shape.heads {
        let start = head * head_dim;
        let end = start + head_dim;
        output[start..end].copy_from_slice(&v[start..end]);
    }
}

fn record_attention_block_event(
    shape: TransformerBlockShape,
    block: &KvAttentionBlock<'_>,
    ledger: &mut TokenLedger,
) {
    let kind = match block.tier {
        MemoryTier::Vram | MemoryTier::SharedHbmOrLpddr => LedgerEventKind::DeviceActivity,
        MemoryTier::PinnedDram | MemoryTier::Dram | MemoryTier::Cxl | MemoryTier::Disk => {
            LedgerEventKind::CpuActivity
        }
    };
    let label = match kind {
        LedgerEventKind::DeviceActivity => "attention_hot_kv_block",
        LedgerEventKind::CpuActivity => "attention_warm_kv_block",
        _ => "attention_kv_block",
    };
    ledger.record(LedgerEvent {
        kind,
        block_id: None,
        from_tier: Some(block.tier),
        to_tier: Some(block.tier),
        bytes: block.token_count * shape.hidden * core::mem::size_of::<f32>() * 2,
        latency_ns: block.token_count as u64,
        label,
    });
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn greedy_argmax(logits: &[f32]) -> Result<TokenId> {
    if logits.is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: "greedy argmax requires non-empty logits".to_string(),
        });
    }
    let mut best_index = 0usize;
    let mut best_value = logits[0];
    if !best_value.is_finite() {
        return Err(NervaError::InvalidArgument {
            reason: "greedy argmax saw non-finite logit".to_string(),
        });
    }
    for (index, value) in logits.iter().copied().enumerate().skip(1) {
        if !value.is_finite() {
            return Err(NervaError::InvalidArgument {
                reason: "greedy argmax saw non-finite logit".to_string(),
            });
        }
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    Ok(TokenId(best_index as u32))
}

fn expected_cycle(seed_token: TokenId, steps: usize, vocab_size: usize) -> Vec<TokenId> {
    (0..steps)
        .map(|step| TokenId((seed_token.0 + step as u32 + 1) % vocab_size as u32))
        .collect()
}

fn hash_f32s(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn hash_tokens(values: &[TokenId]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.0.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn token_ids_to_json(tokens: &[TokenId]) -> String {
    let mut out = String::from("[");
    for (index, token) in tokens.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&token.0.to_string());
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_block_preserves_residual() {
        let shape = TransformerBlockShape::new(4, 2, 8);
        let block = ReferenceTransformerBlock::zero_for_shape(shape).unwrap();
        let mut scratch = TransformerBlockScratch::new(shape).unwrap();
        let mut output = [0.0; 4];
        let input = [1.0, -2.0, 3.0, -4.0];
        let mut ledger = TokenLedger::new(0);

        block
            .forward_into(&input, &mut scratch, &mut output, &mut ledger)
            .unwrap();

        assert_eq!(output, input);
        assert_eq!(ledger.hot_path_allocations, 0);
        assert!(ledger.require_zero_hot_path_allocations().is_ok());
    }

    #[test]
    fn nontrivial_block_matches_hand_reference() {
        let shape = TransformerBlockShape::new(2, 1, 2);
        let block = ReferenceTransformerBlock::new(
            shape,
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.0, 0.0, 0.5],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            1e-5,
        )
        .unwrap();
        let mut scratch = TransformerBlockScratch::new(shape).unwrap();
        let mut output = [0.0; 2];
        let input = [1.0, 2.0];
        let mut ledger = TokenLedger::new(7);

        block
            .forward_into(&input, &mut scratch, &mut output, &mut ledger)
            .unwrap();

        let attn_norm_scale = ((1.0_f32 + 4.0) / 2.0 + 1e-5).sqrt().recip();
        let attn = [input[0] * attn_norm_scale, input[1] * attn_norm_scale];
        let residual = [input[0] + attn[0], input[1] + attn[1]];
        let mlp_norm_scale = ((residual[0] * residual[0] + residual[1] * residual[1]) / 2.0 + 1e-5)
            .sqrt()
            .recip();
        let mlp_norm = [residual[0] * mlp_norm_scale, residual[1] * mlp_norm_scale];
        let expected = [
            residual[0] + silu(0.5 * mlp_norm[0]) * mlp_norm[0],
            residual[1] + silu(0.5 * mlp_norm[1]) * mlp_norm[1],
        ];

        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(ledger.hot_path_allocations, 0);
    }

    #[test]
    fn rejects_bad_shapes_and_scratch_mismatch() {
        assert!(TransformerBlockShape::new(3, 2, 4).validate().is_err());
        let block =
            ReferenceTransformerBlock::zero_for_shape(TransformerBlockShape::new(4, 2, 8)).unwrap();
        let mut scratch =
            TransformerBlockScratch::new(TransformerBlockShape::new(2, 1, 2)).unwrap();
        let mut ledger = TokenLedger::new(0);
        let mut output = [0.0; 4];
        assert!(
            block
                .forward_into(&[0.0; 4], &mut scratch, &mut output, &mut ledger)
                .is_err()
        );
    }

    #[test]
    fn reference_block_smoke_reports_hash_and_no_allocations() {
        let summary = reference_block_smoke().unwrap();
        assert_eq!(summary.status, ReferenceBlockSmokeStatus::Ok);
        assert_eq!(summary.hidden, 2);
        assert_eq!(summary.heads, 1);
        assert_eq!(summary.intermediate, 2);
        assert_eq!(summary.hot_path_allocations, 0);
        assert_eq!(summary.output_hash, 3_850_145_622_605_741_247);
        assert!(summary.to_json().contains("\"status\":\"ok\""));
    }

    #[test]
    fn blockwise_attention_matches_dense_reference_across_tiers() {
        let shape = TransformerBlockShape::new(4, 2, 4);
        let query = [0.5, -1.0, 0.25, 0.75];
        let keys = [0.1, 0.2, 0.3, 0.4, 0.0, -0.5, 0.6, 0.2, 0.7, 0.1, -0.2, 0.3];
        let values = [
            1.0, 0.0, 0.5, -0.5, -1.0, 2.0, 0.25, 0.75, 0.3, -0.8, 1.5, 0.2,
        ];
        let blocks = [
            KvAttentionBlock::new(&keys[..4], &values[..4], 1, MemoryTier::Dram),
            KvAttentionBlock::new(&keys[4..], &values[4..], 2, MemoryTier::Vram),
        ];
        let mut scratch = BlockwiseAttentionScratch::new(shape).unwrap();
        let mut output = [0.0; 4];
        let mut ledger = TokenLedger::new(11);

        exact_blockwise_attention_into(
            shape,
            &query,
            &blocks,
            &mut scratch,
            &mut output,
            &mut ledger,
        )
        .unwrap();

        let expected = dense_attention_reference(shape, &query, &keys, &values, 3);
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(ledger.event_count(LedgerEventKind::CpuActivity), 1);
        assert_eq!(ledger.event_count(LedgerEventKind::DeviceActivity), 1);
        assert_eq!(ledger.total_latency_ns(), 3);
        assert_eq!(ledger.hot_path_allocations, 0);
    }

    #[test]
    fn blockwise_attention_rejects_empty_and_malformed_blocks() {
        let shape = TransformerBlockShape::new(2, 1, 2);
        let query = [1.0, 0.0];
        let mut scratch = BlockwiseAttentionScratch::new(shape).unwrap();
        let mut output = [0.0; 2];
        let mut ledger = TokenLedger::new(0);

        assert!(
            exact_blockwise_attention_into(
                shape,
                &query,
                &[],
                &mut scratch,
                &mut output,
                &mut ledger
            )
            .is_err()
        );

        let bad_block = [KvAttentionBlock::new(
            &[1.0],
            &[1.0, 0.0],
            1,
            MemoryTier::Dram,
        )];
        assert!(
            exact_blockwise_attention_into(
                shape,
                &query,
                &bad_block,
                &mut scratch,
                &mut output,
                &mut ledger,
            )
            .is_err()
        );
    }

    #[test]
    fn blockwise_attention_smoke_reports_tier_events() {
        let summary = blockwise_attention_smoke().unwrap();
        assert_eq!(summary.status, BlockwiseAttentionSmokeStatus::Ok);
        assert_eq!(summary.hidden, 2);
        assert_eq!(summary.heads, 1);
        assert_eq!(summary.blocks, 2);
        assert_eq!(summary.tokens, 4);
        assert_eq!(summary.cpu_block_events, 1);
        assert_eq!(summary.device_block_events, 1);
        assert_eq!(summary.hot_path_allocations, 0);
        assert!(summary.to_json().contains("\"device_block_events\":1"));
    }

    #[test]
    fn tiny_greedy_model_matches_expected_token_cycle() {
        let model = tiny_cycle_model().unwrap();
        let mut scratch = TinyGreedyDecodeScratch::new(model.shape(), model.vocab_size()).unwrap();
        let output = model.decode_greedy(TokenId(0), 8, &mut scratch).unwrap();

        assert_eq!(
            output.tokens,
            vec![
                TokenId(1),
                TokenId(2),
                TokenId(3),
                TokenId(0),
                TokenId(1),
                TokenId(2),
                TokenId(3),
                TokenId(0),
            ]
        );
        assert_eq!(output.ledgers.len(), 8);
        assert_eq!(
            output
                .ledgers
                .iter()
                .map(|ledger| ledger.event_count(LedgerEventKind::DeviceActivity))
                .sum::<u64>(),
            8
        );
        assert_eq!(
            output
                .ledgers
                .iter()
                .map(|ledger| ledger.hot_path_allocations)
                .sum::<u64>(),
            0
        );
    }

    #[test]
    fn tiny_greedy_model_rejects_bad_decode_inputs() {
        let model = tiny_cycle_model().unwrap();
        let mut scratch = TinyGreedyDecodeScratch::new(model.shape(), model.vocab_size()).unwrap();

        assert!(model.decode_greedy(TokenId(0), 0, &mut scratch).is_err());
        assert!(model.decode_greedy(TokenId(99), 1, &mut scratch).is_err());

        let mut wrong_scratch =
            TinyGreedyDecodeScratch::new(TransformerBlockShape::new(4, 2, 4), model.vocab_size())
                .unwrap();
        assert!(
            model
                .decode_greedy(TokenId(0), 1, &mut wrong_scratch)
                .is_err()
        );
    }

    #[test]
    fn tiny_greedy_decode_smoke_reports_parity_and_ledger() {
        let summary = tiny_greedy_decode_smoke(8).unwrap();

        assert_eq!(summary.status, TinyGreedyDecodeStatus::Ok);
        assert_eq!(summary.seed_token, TokenId(0));
        assert_eq!(summary.steps, 8);
        assert_eq!(summary.vocab_size, 4);
        assert!(summary.parity);
        assert_eq!(summary.tokens, summary.expected_tokens);
        assert_eq!(summary.ledger_count, 8);
        assert_eq!(summary.device_events, 8);
        assert_eq!(summary.total_latency_ns, 8);
        assert_eq!(summary.hot_path_allocations, 0);
        assert!(summary.to_json().contains("\"parity\":true"));
    }

    fn dense_attention_reference(
        shape: TransformerBlockShape,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        token_count: usize,
    ) -> Vec<f32> {
        let head_dim = shape.head_dim();
        let scale = (head_dim as f32).sqrt().recip();
        let mut output = vec![0.0; shape.hidden];
        for head in 0..shape.heads {
            let head_start = head * head_dim;
            let head_end = head_start + head_dim;
            let mut max_score = f32::NEG_INFINITY;
            let mut scores = Vec::with_capacity(token_count);
            for token_index in 0..token_count {
                let token_start = token_index * shape.hidden + head_start;
                let token_end = token_start + head_dim;
                let score =
                    dot(&query[head_start..head_end], &keys[token_start..token_end]) * scale;
                max_score = max_score.max(score);
                scores.push(score);
            }
            let mut denom = 0.0f32;
            for (token_index, score) in scores.iter().copied().enumerate() {
                let weight = (score - max_score).exp();
                denom += weight;
                let token_start = token_index * shape.hidden + head_start;
                let token_end = token_start + head_dim;
                for (out, value) in output[head_start..head_end]
                    .iter_mut()
                    .zip(values[token_start..token_end].iter().copied())
                {
                    *out += weight * value;
                }
            }
            for out in &mut output[head_start..head_end] {
                *out /= denom;
            }
        }
        output
    }
}
