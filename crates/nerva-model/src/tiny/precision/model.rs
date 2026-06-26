use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::shape::TransformerBlockShape;
use crate::common::token::{greedy_argmax, require_token_in_vocab};
use crate::common::validate::require_len;
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::tiny::precision::codec::{
    copy_encoded_embedding_row, decode_slice_into, encode_slice, encoded_lm_head_into,
};
use crate::tiny::precision::output::TinyPrecisionGreedyDecodeOutput;
use crate::tiny::precision::scratch::TinyPrecisionGreedyDecodeScratch;

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
