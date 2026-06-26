use nerva_core::types::{ExecutionOwner, MemoryTier, NervaError, Result, TokenId};
use nerva_ledger::types::{
    CandidateCost, ExecutionDecision, LedgerEvent, LedgerEventKind, MetricSource, TokenLedger,
};

use crate::common::hash::hash_tokens;
use crate::common::math::mat_vec_row_major;
use crate::common::shape::TransformerBlockShape;
use crate::common::token::{
    copy_embedding_row, expected_cycle, greedy_argmax, require_token_in_vocab, token_ids_to_json,
};
use crate::common::validate::require_len;
use crate::reference::block::ReferenceTransformerBlock;
use crate::reference::scratch::TransformerBlockScratch;

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

pub(crate) fn tiny_cycle_model() -> Result<TinyGreedyModel> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let block = ReferenceTransformerBlock::zero_for_shape(shape)?;
    TinyGreedyModel::new(
        4,
        block,
        vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
        vec![0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
    )
}
