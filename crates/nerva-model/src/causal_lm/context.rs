use std::time::Instant;

use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::causal_lm::context_scratch::{
    copy_last_prompt_row, required_context_tokens, reset_layer_kv,
};
use crate::causal_lm::decode::{copy_embedding_into, elapsed_ns, record_decode_ledger};
use crate::causal_lm::types::{
    HfCausalLmContextMode, HfCausalLmDecodeOutput, HfCausalLmDecodeScratch, HfCausalLmModel,
};
use crate::common::token::require_token_in_vocab;

impl HfCausalLmModel {
    pub fn decode_greedy_from_prompt_tokens(
        &self,
        prompt_tokens: &[TokenId],
        steps: usize,
        scratch: &mut HfCausalLmDecodeScratch,
    ) -> Result<HfCausalLmDecodeOutput> {
        let seed_token = self.validate_prompt_tokens(prompt_tokens)?;
        let required_context = required_context_tokens(prompt_tokens.len(), steps)?;
        if scratch.has_context_capacity(self, required_context) {
            return self.decode_greedy_with_prompt_context(prompt_tokens, steps, scratch);
        }
        let (generated_tokens, ledgers) =
            self.decode_greedy_from_seed(seed_token, steps, scratch)?;
        Ok(HfCausalLmDecodeOutput {
            context_mode: HfCausalLmContextMode::LastTokenSeedOnly,
            prompt_tokens: prompt_tokens.to_vec(),
            seed_token,
            generated_tokens,
            ledgers,
        })
    }

    pub fn decode_greedy_with_prompt_context(
        &self,
        prompt_tokens: &[TokenId],
        steps: usize,
        scratch: &mut HfCausalLmDecodeScratch,
    ) -> Result<HfCausalLmDecodeOutput> {
        let seed_token = self.validate_prompt_tokens(prompt_tokens)?;
        let required_context = required_context_tokens(prompt_tokens.len(), steps)?;
        scratch.require_context_capacity(self, required_context)?;
        reset_layer_kv(scratch);

        let mut generated_tokens = Vec::with_capacity(steps);
        let mut ledgers = Vec::with_capacity(steps);
        let mut first_ledger = TokenLedger::new(0);
        let start = Instant::now();
        self.forward_prompt_context(prompt_tokens, scratch, &mut first_ledger)?;
        let mut current = self.sample_current_hidden(scratch)?;
        record_decode_ledger(
            self,
            elapsed_ns(start),
            "hf_causal_lm_prompt_prefill_kv_decode",
            &mut first_ledger,
        );
        first_ledger.require_zero_hot_path_allocations()?;
        generated_tokens.push(current);
        ledgers.push(first_ledger);

        for step in 1..steps {
            let mut ledger = TokenLedger::new(step as u64);
            let start = Instant::now();
            copy_embedding_into(
                &self.embeddings,
                self.metadata.hidden_size,
                current,
                &mut scratch.hidden_bits,
            )?;
            for (layer, kv) in self.layers.iter().zip(scratch.kv_layers.iter_mut()) {
                layer.forward_decode_with_kv_into(
                    &scratch.hidden_bits,
                    kv,
                    &mut scratch.next_bits,
                    &mut ledger,
                )?;
                scratch.hidden_bits.copy_from_slice(&scratch.next_bits);
            }
            current = self.sample_current_hidden(scratch)?;
            record_decode_ledger(
                self,
                elapsed_ns(start),
                "hf_causal_lm_kv_decode",
                &mut ledger,
            );
            ledger.require_zero_hot_path_allocations()?;
            generated_tokens.push(current);
            ledgers.push(ledger);
        }

        Ok(HfCausalLmDecodeOutput {
            context_mode: HfCausalLmContextMode::PromptPrefillKvDecode,
            prompt_tokens: prompt_tokens.to_vec(),
            seed_token,
            generated_tokens,
            ledgers,
        })
    }

    fn validate_prompt_tokens(&self, prompt_tokens: &[TokenId]) -> Result<TokenId> {
        let seed_token = *prompt_tokens
            .last()
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "HF causal LM decode requires at least one prompt token".to_string(),
            })?;
        for token in prompt_tokens {
            require_token_in_vocab(*token, self.metadata.vocab_size)?;
        }
        Ok(seed_token)
    }

    fn forward_prompt_context(
        &self,
        prompt_tokens: &[TokenId],
        scratch: &mut HfCausalLmDecodeScratch,
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let hidden = self.metadata.hidden_size;
        let values = prompt_tokens.len() * hidden;
        for (index, token) in prompt_tokens.iter().copied().enumerate() {
            let start = index * hidden;
            copy_embedding_into(
                &self.embeddings,
                hidden,
                token,
                &mut scratch.sequence_bits[start..start + hidden],
            )?;
        }
        let mut primary = true;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let kv = &mut scratch.kv_layers[layer_index];
            if primary {
                layer.forward_prefill_sequence_into(
                    &scratch.sequence_bits[..values],
                    prompt_tokens.len(),
                    kv,
                    &mut scratch.sequence_next_bits[..values],
                    ledger,
                )?;
            } else {
                layer.forward_prefill_sequence_into(
                    &scratch.sequence_next_bits[..values],
                    prompt_tokens.len(),
                    kv,
                    &mut scratch.sequence_bits[..values],
                    ledger,
                )?;
            }
            primary = !primary;
        }
        copy_last_prompt_row(prompt_tokens.len(), hidden, primary, scratch)
    }
}
