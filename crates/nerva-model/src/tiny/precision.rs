use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::TokenId;
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::TokenLedger;

use crate::common::hash::hash_tokens;
use crate::common::shape::TransformerBlockShape;
use crate::common::token::{expected_cycle, greedy_argmax, require_token_in_vocab};
use crate::common::validate::require_len;
use crate::precision::bits::{decode_f32_for_dtype, dtype_label, encode_f32_for_dtype};
use crate::precision::block::PrecisionTransformerBlock;
use crate::precision::scratch::PrecisionTransformerBlockScratch;

#[derive(Clone, Debug)]
pub struct TinyPrecisionGreedyModel {
    dtype: DType,
    vocab_size: usize,
    shape: TransformerBlockShape,
    block: PrecisionTransformerBlock,
    embeddings: Vec<u16>,
    lm_head: Vec<u16>,
}

impl TinyPrecisionGreedyModel {
    pub fn new_from_f32(
        dtype: DType,
        block: PrecisionTransformerBlock,
        embeddings: &[f32],
        lm_head: &[f32],
    ) -> Result<Self> {
        let shape = block.shape();
        let vocab_size = embeddings.len() / shape.hidden;
        if vocab_size == 0 || embeddings.len() % shape.hidden != 0 {
            return Err(NervaError::InvalidArgument {
                reason: "tiny precision embeddings must be complete vocabulary rows".to_string(),
            });
        }
        require_len("lm_head", lm_head.len(), vocab_size * shape.hidden)?;
        let embeddings = encode_slice(dtype, embeddings)?;
        let lm_head = encode_slice(dtype, lm_head)?;
        Ok(Self {
            dtype,
            vocab_size,
            shape,
            block,
            embeddings,
            lm_head,
        })
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
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
        scratch: &mut TinyPrecisionGreedyDecodeScratch,
    ) -> Result<TinyPrecisionGreedyDecodeOutput> {
        if steps == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "tiny precision greedy decode steps must be non-zero".to_string(),
            });
        }
        scratch.require_shape(self.shape, self.vocab_size)?;
        require_token_in_vocab(seed_token, self.vocab_size)?;

        let mut current_token = seed_token;
        let mut tokens = Vec::with_capacity(steps);
        let mut ledgers = Vec::with_capacity(steps);
        for step in 0..steps {
            copy_encoded_embedding_row(
                &self.embeddings,
                self.shape.hidden,
                current_token,
                &mut scratch.hidden_bits,
            )?;
            let mut ledger = TokenLedger::new(step as u64);
            self.block.forward_into(
                &scratch.hidden_bits,
                &mut scratch.block_scratch,
                &mut scratch.block_output_bits,
                &mut ledger,
            )?;
            decode_slice_into(
                self.dtype,
                &scratch.block_output_bits,
                &mut scratch.decoded_output,
            )?;
            encoded_lm_head_into(
                self.dtype,
                &self.lm_head,
                &scratch.decoded_output,
                &mut scratch.logits,
            )?;
            let next_token = greedy_argmax(&scratch.logits)?;
            ledger.record_execution_decision(ExecutionDecision {
                operation: "tiny_precision_greedy_decode",
                executor_selected: ExecutionOwner::Cpu,
                candidate_costs: vec![
                    CandidateCost::estimated("cpu-resident-encoded", 1),
                    CandidateCost::estimated("gpu-staged-encoded", 3),
                ],
                reason: "tiny precision model is encoded and already resident in DRAM",
                predicted_visible_ns: 1,
                actual_visible_ns: Some(1),
                metric_source: MetricSource::EstimatedModel,
            });
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::CpuActivity,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: None,
                from_tier: Some(MemoryTier::Dram),
                to_tier: Some(MemoryTier::Dram),
                bytes: (self.shape.hidden * 2 + self.lm_head.len()) * core::mem::size_of::<u16>(),
                latency_ns: 1,
                label: "tiny_precision_greedy_decode",
            });
            ledger.require_zero_hot_path_allocations()?;
            tokens.push(next_token);
            ledgers.push(ledger);
            current_token = next_token;
        }

        Ok(TinyPrecisionGreedyDecodeOutput { tokens, ledgers })
    }
}

#[derive(Clone, Debug)]
pub struct TinyPrecisionGreedyDecodeScratch {
    shape: TransformerBlockShape,
    vocab_size: usize,
    block_scratch: PrecisionTransformerBlockScratch,
    hidden_bits: Vec<u16>,
    block_output_bits: Vec<u16>,
    decoded_output: Vec<f32>,
    logits: Vec<f32>,
}

impl TinyPrecisionGreedyDecodeScratch {
    pub fn new(shape: TransformerBlockShape, vocab_size: usize) -> Result<Self> {
        shape.validate()?;
        if vocab_size == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "tiny precision greedy scratch vocabulary must be non-zero".to_string(),
            });
        }
        Ok(Self {
            shape,
            vocab_size,
            block_scratch: PrecisionTransformerBlockScratch::new(shape)?,
            hidden_bits: vec![0; shape.hidden],
            block_output_bits: vec![0; shape.hidden],
            decoded_output: vec![0.0; shape.hidden],
            logits: vec![0.0; vocab_size],
        })
    }

    fn require_shape(&self, shape: TransformerBlockShape, vocab_size: usize) -> Result<()> {
        if self.shape == shape && self.vocab_size == vocab_size {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "tiny precision scratch shape does not match model shape".to_string(),
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TinyPrecisionGreedyDecodeOutput {
    pub tokens: Vec<TokenId>,
    pub ledgers: Vec<TokenLedger>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TinyPrecisionGreedyDecodeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TinyPrecisionGreedyDecodeSummary {
    pub status: TinyPrecisionGreedyDecodeStatus,
    pub dtype: DType,
    pub seed_token: TokenId,
    pub steps: usize,
    pub vocab_size: usize,
    pub tokens: Vec<TokenId>,
    pub expected_tokens: Vec<TokenId>,
    pub parity: bool,
    pub ledger_count: u64,
    pub cpu_events: u64,
    pub execution_decisions: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub output_hash: u64,
}

impl TinyPrecisionGreedyDecodeSummary {
    pub fn passed(&self) -> bool {
        self.parity
            && self.ledger_count == self.steps as u64
            && self.cpu_events == self.steps as u64
            && self.execution_decisions == self.steps as u64
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            TinyPrecisionGreedyDecodeStatus::Ok => "ok",
        };
        let dtype = dtype_label(self.dtype).unwrap_or("unsupported");
        format!(
            "{{\"status\":\"{}\",\"dtype\":\"{}\",\"seed_token\":{},\"steps\":{},\"vocab_size\":{},\"tokens\":{},\"expected_tokens\":{},\"parity\":{},\"ledger_count\":{},\"cpu_events\":{},\"execution_decisions\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"output_hash\":{}}}",
            status,
            dtype,
            self.seed_token.0,
            self.steps,
            self.vocab_size,
            crate::common::token::token_ids_to_json(&self.tokens),
            crate::common::token::token_ids_to_json(&self.expected_tokens),
            self.parity,
            self.ledger_count,
            self.cpu_events,
            self.execution_decisions,
            self.total_latency_ns,
            self.hot_path_allocations,
            self.output_hash,
        )
    }
}

pub fn tiny_precision_greedy_decode_smoke(
    dtype: DType,
    steps: usize,
) -> Result<TinyPrecisionGreedyDecodeSummary> {
    let model = tiny_precision_cycle_model(dtype)?;
    let seed_token = TokenId(0);
    let mut scratch = TinyPrecisionGreedyDecodeScratch::new(model.shape(), model.vocab_size())?;
    let output = model.decode_greedy(seed_token, steps, &mut scratch)?;
    let expected_tokens = expected_cycle(seed_token, steps, model.vocab_size());
    let parity = output.tokens == expected_tokens;
    if !parity {
        return Err(NervaError::InvalidArgument {
            reason: "tiny precision greedy decode token parity failed".to_string(),
        });
    }
    let hot_path_allocations = output
        .ledgers
        .iter()
        .map(|ledger| ledger.hot_path_allocations)
        .sum();
    let cpu_events = output
        .ledgers
        .iter()
        .map(|ledger| ledger.event_count(LedgerEventKind::CpuActivity))
        .sum();
    let execution_decisions = output
        .ledgers
        .iter()
        .map(|ledger| ledger.execution_decisions.len() as u64)
        .sum();
    let total_latency_ns = output
        .ledgers
        .iter()
        .map(TokenLedger::total_latency_ns)
        .sum();

    let summary = TinyPrecisionGreedyDecodeSummary {
        status: TinyPrecisionGreedyDecodeStatus::Ok,
        dtype: model.dtype(),
        seed_token,
        steps,
        vocab_size: model.vocab_size(),
        output_hash: hash_tokens(&output.tokens),
        tokens: output.tokens,
        expected_tokens,
        parity,
        ledger_count: output.ledgers.len() as u64,
        cpu_events,
        execution_decisions,
        total_latency_ns,
        hot_path_allocations,
    };
    if summary.passed() {
        Ok(summary)
    } else {
        Err(NervaError::InvalidArgument {
            reason: "tiny precision decode ledger invariants failed".to_string(),
        })
    }
}

pub fn tiny_precision_cycle_model(dtype: DType) -> Result<TinyPrecisionGreedyModel> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let block = PrecisionTransformerBlock::new_from_f32(
        dtype,
        shape,
        &[1.0, 1.0],
        &[1.0, 1.0],
        &[0.0; 4],
        &[0.0; 4],
        &[0.0; 4],
        &[0.0; 4],
        &[0.0; 4],
        &[0.0; 4],
        &[0.0; 4],
        1e-5,
    )?;
    TinyPrecisionGreedyModel::new_from_f32(
        dtype,
        block,
        &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
        &[0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
    )
}

fn encode_slice(dtype: DType, values: &[f32]) -> Result<Vec<u16>> {
    values
        .iter()
        .copied()
        .map(|value| encode_f32_for_dtype(value, dtype))
        .collect()
}

fn copy_encoded_embedding_row(
    embeddings: &[u16],
    hidden: usize,
    token: TokenId,
    output: &mut [u16],
) -> Result<()> {
    require_token_in_vocab(token, embeddings.len() / hidden)?;
    require_len("encoded embedding output", output.len(), hidden)?;
    let start = token.0 as usize * hidden;
    let end = start + hidden;
    output.copy_from_slice(&embeddings[start..end]);
    Ok(())
}

fn decode_slice_into(dtype: DType, values: &[u16], output: &mut [f32]) -> Result<()> {
    require_len("decoded precision output", output.len(), values.len())?;
    for (out, value) in output.iter_mut().zip(values.iter().copied()) {
        *out = decode_f32_for_dtype(value, dtype)?;
    }
    Ok(())
}

fn encoded_lm_head_into(
    dtype: DType,
    lm_head: &[u16],
    input: &[f32],
    logits: &mut [f32],
) -> Result<()> {
    require_len("encoded lm_head", lm_head.len(), logits.len() * input.len())?;
    for (row, logit) in lm_head.chunks_exact(input.len()).zip(logits.iter_mut()) {
        let mut sum = 0.0f32;
        for (weight, value) in row.iter().copied().zip(input.iter().copied()) {
            sum += decode_f32_for_dtype(weight, dtype)? * value;
        }
        *logit = sum;
    }
    Ok(())
}
