use std::time::Instant;

use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::causal_lm::types::{HfCausalLmDecodeScratch, HfCausalLmModel};
use crate::common::token::{greedy_argmax, require_token_in_vocab};
use crate::precision::block::ops::{
    decode_vec_into, mat_vec_encoded_row_major, rms_norm_encoded_into,
};

impl HfCausalLmModel {
    pub fn decode_greedy(
        &self,
        seed_token: TokenId,
        steps: usize,
        scratch: &mut HfCausalLmDecodeScratch,
    ) -> Result<(Vec<TokenId>, Vec<TokenLedger>)> {
        if steps == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "HF causal LM greedy decode steps must be non-zero".to_string(),
            });
        }
        scratch.require_shape(self)?;
        require_token_in_vocab(seed_token, self.metadata.vocab_size)?;

        let mut current = seed_token;
        let mut tokens = Vec::with_capacity(steps);
        let mut ledgers = Vec::with_capacity(steps);
        for step in 0..steps {
            let start = Instant::now();
            copy_embedding(
                &self.embeddings,
                self.metadata.hidden_size,
                current,
                scratch,
            )?;
            let mut ledger = TokenLedger::new(step as u64);
            for layer in &self.layers {
                layer.forward_into(
                    &scratch.hidden_bits,
                    &mut scratch.block,
                    &mut scratch.next_bits,
                    &mut ledger,
                )?;
                scratch.hidden_bits.copy_from_slice(&scratch.next_bits);
            }
            decode_vec_into(self.dtype, &scratch.hidden_bits, &mut scratch.decoded)?;
            rms_norm_encoded_into(
                self.dtype,
                &scratch.decoded,
                &self.final_norm,
                self.rms_eps,
                &mut scratch.normed,
            )?;
            mat_vec_encoded_row_major(
                self.dtype,
                &self.lm_head,
                &scratch.normed,
                &mut scratch.logits,
            )?;
            let next = greedy_argmax(&scratch.logits)?;
            record_decode_ledger(self, elapsed_ns(start), &mut ledger);
            ledger.require_zero_hot_path_allocations()?;
            tokens.push(next);
            ledgers.push(ledger);
            current = next;
        }
        Ok((tokens, ledgers))
    }
}

impl HfCausalLmDecodeScratch {
    pub(crate) fn require_shape(&self, model: &HfCausalLmModel) -> Result<()> {
        if self.shape == model.shape() && self.vocab_size == model.metadata().vocab_size {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "HF causal LM decode scratch shape does not match model".to_string(),
            })
        }
    }
}

fn copy_embedding(
    embeddings: &[u16],
    hidden: usize,
    token: TokenId,
    scratch: &mut HfCausalLmDecodeScratch,
) -> Result<()> {
    let start = token.0 as usize * hidden;
    let end = start + hidden;
    if end > embeddings.len() {
        return Err(NervaError::InvalidArgument {
            reason: "HF causal LM embedding token is outside vocabulary".to_string(),
        });
    }
    scratch.hidden_bits.copy_from_slice(&embeddings[start..end]);
    Ok(())
}

fn record_decode_ledger(model: &HfCausalLmModel, elapsed_ns: u64, ledger: &mut TokenLedger) {
    let hidden_bytes = model.metadata.hidden_size * model.metadata.num_hidden_layers.max(1) * 2;
    let bytes = (hidden_bytes + model.lm_head.len() * 2) as u64;
    ledger.record_execution_decision(ExecutionDecision {
        operation: "hf_causal_lm_greedy_decode",
        executor_selected: ExecutionOwner::Cpu,
        candidate_costs: vec![
            CandidateCost::measured("cpu-exact-loaded-safetensors", elapsed_ns),
            CandidateCost::estimated(
                "gpu-resident-hf-causal-lm-estimate",
                elapsed_ns.saturating_mul(2),
            ),
        ],
        reason: "exact loaded safetensors CPU path selected for this measured run",
        predicted_visible_ns: elapsed_ns,
        actual_visible_ns: Some(elapsed_ns),
        metric_source: MetricSource::RuntimeTimestamp,
    });
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::CpuActivity,
        sync_class: None,
        metric_source: MetricSource::RuntimeTimestamp,
        block_id: None,
        from_tier: Some(MemoryTier::Dram),
        to_tier: Some(MemoryTier::Dram),
        bytes: bytes as usize,
        latency_ns: elapsed_ns,
        label: "hf_causal_lm_greedy_decode",
    });
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().max(1).min(u64::MAX as u128) as u64
}
