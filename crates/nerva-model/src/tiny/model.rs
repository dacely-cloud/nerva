use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::TokenId;
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::math::mat_vec_row_major;
use crate::common::shape::TransformerBlockShape;
use crate::common::token::{copy_embedding_row, greedy_argmax, require_token_in_vocab};
use crate::common::validate::require_len;
use crate::reference::block::ReferenceTransformerBlock;
use crate::tiny::output::TinyGreedyDecodeOutput;
use crate::tiny::scratch::TinyGreedyDecodeScratch;

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
                scratch.hidden_mut(),
            )?;
            let mut ledger = TokenLedger::new(step as u64);
            let (hidden, block_scratch, block_output) = scratch.block_forward_parts();
            self.block
                .forward_into(hidden, block_scratch, block_output, &mut ledger)?;
            let (block_output, logits) = scratch.logit_parts_mut();
            mat_vec_row_major(&self.lm_head, block_output, logits);
            let next_token = greedy_argmax(scratch.logits())?;
            ledger.record_execution_decision(ExecutionDecision {
                operation: "tiny_greedy_decode",
                executor_selected: ExecutionOwner::Cpu,
                candidate_costs: vec![
                    CandidateCost::estimated("cpu-resident-reference", 1),
                    CandidateCost::estimated("gpu-staged-reference", 3),
                ],
                reason: "tiny reference model is already resident in DRAM",
                predicted_visible_ns: 1,
                actual_visible_ns: Some(1),
                metric_source: MetricSource::EstimatedModel,
            });
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::DeviceActivity,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
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
