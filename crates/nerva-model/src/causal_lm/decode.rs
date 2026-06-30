use std::time::Instant;

use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::causal_lm::types::{HfCausalLmDecodeScratch, HfCausalLmModel, HfCausalLmStopReason};
use crate::common::token::greedy_argmax;
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
        let output = self.decode_greedy_from_prompt_tokens(&[seed_token], steps, scratch)?;
        Ok((output.generated_tokens, output.ledgers))
    }

    pub(crate) fn decode_greedy_from_seed(
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
                layer.forward_with_token_into(
                    &scratch.hidden_bits,
                    Some(current),
                    &mut scratch.block,
                    &mut scratch.next_bits,
                    &mut ledger,
                )?;
                scratch.hidden_bits.copy_from_slice(&scratch.next_bits);
            }
            let next = self.sample_current_hidden(scratch)?;
            record_decode_ledger(
                self,
                elapsed_ns(start),
                "hf_causal_lm_greedy_decode",
                &mut ledger,
            );
            ledger.require_zero_hot_path_allocations()?;
            tokens.push(next);
            ledgers.push(ledger);
            if self.is_eos_token(next) {
                break;
            }
            current = next;
        }
        Ok((tokens, ledgers))
    }

    pub(crate) fn sample_current_hidden(
        &self,
        scratch: &mut HfCausalLmDecodeScratch,
    ) -> Result<TokenId> {
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
        greedy_argmax(&scratch.logits)
    }

    pub(crate) fn is_eos_token(&self, token: TokenId) -> bool {
        self.metadata.eos_token_id == Some(token.0)
    }

    pub fn stop_reason_for_tokens(&self, tokens: &[TokenId], steps: usize) -> HfCausalLmStopReason {
        match tokens.last().copied() {
            Some(token) if self.is_eos_token(token) => HfCausalLmStopReason::EosToken,
            _ if tokens.len() >= steps => HfCausalLmStopReason::MaxSteps,
            _ => HfCausalLmStopReason::MaxSteps,
        }
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
    copy_embedding_into(embeddings, hidden, token, &mut scratch.hidden_bits)
}

pub(crate) fn copy_embedding_into(
    embeddings: &[u16],
    hidden: usize,
    token: TokenId,
    output: &mut [u16],
) -> Result<()> {
    let start = token.0 as usize * hidden;
    let end = start + hidden;
    if end > embeddings.len() {
        return Err(NervaError::InvalidArgument {
            reason: "HF causal LM embedding token is outside vocabulary".to_string(),
        });
    }
    output.copy_from_slice(&embeddings[start..end]);
    Ok(())
}

pub(crate) fn record_decode_ledger(
    model: &HfCausalLmModel,
    elapsed_ns: u64,
    operation: &'static str,
    ledger: &mut TokenLedger,
) {
    let hidden_bytes = model.metadata.hidden_size * model.metadata.num_hidden_layers.max(1) * 2;
    let bytes = (hidden_bytes + model.lm_head.len() * 2) as u64;
    ledger.record_execution_decision(ExecutionDecision {
        operation,
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
        label: operation,
    });
}

pub(crate) fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().max(1).min(u64::MAX as u128) as u64
}
