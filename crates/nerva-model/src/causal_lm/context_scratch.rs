use nerva_core::types::error::{NervaError, Result};

use crate::causal_lm::types::{HfCausalLmDecodeScratch, HfCausalLmModel};

impl HfCausalLmDecodeScratch {
    pub(crate) fn has_context_capacity(
        &self,
        model: &HfCausalLmModel,
        required_tokens: usize,
    ) -> bool {
        self.shape == model.shape()
            && self.vocab_size == model.metadata().vocab_size
            && self.kv_layers.len() == model.layer_count()
            && self.max_context_tokens >= required_tokens
    }

    pub(crate) fn require_context_capacity(
        &self,
        model: &HfCausalLmModel,
        required_tokens: usize,
    ) -> Result<()> {
        if self.has_context_capacity(model, required_tokens) {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "HF causal LM decode scratch context capacity is too small".to_string(),
            })
        }
    }
}

pub(crate) fn required_context_tokens(prompt_len: usize, steps: usize) -> Result<usize> {
    if steps == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "HF causal LM greedy decode steps must be non-zero".to_string(),
        });
    }
    prompt_len
        .checked_add(steps)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "HF causal LM context length overflow".to_string(),
        })
}

pub(crate) fn reset_layer_kv(scratch: &mut HfCausalLmDecodeScratch) {
    for kv in &mut scratch.kv_layers {
        kv.reset();
    }
}

pub(crate) fn copy_last_prompt_row(
    prompt_len: usize,
    hidden: usize,
    primary: bool,
    scratch: &mut HfCausalLmDecodeScratch,
) -> Result<()> {
    let start = (prompt_len - 1) * hidden;
    let end = start + hidden;
    if primary {
        scratch
            .hidden_bits
            .copy_from_slice(&scratch.sequence_bits[start..end]);
    } else {
        scratch
            .hidden_bits
            .copy_from_slice(&scratch.sequence_next_bits[start..end]);
    }
    Ok(())
}
