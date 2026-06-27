use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;

use crate::causal_lm::types::{HfCausalLmDecodeScratch, HfCausalLmModel};

impl HfCausalLmModel {
    pub fn sample_encoded_hidden(
        &self,
        hidden_bits: &[u16],
        scratch: &mut HfCausalLmDecodeScratch,
    ) -> Result<TokenId> {
        scratch.require_shape(self)?;
        if hidden_bits.len() != self.metadata().hidden_size {
            return Err(NervaError::InvalidArgument {
                reason: "HF causal LM sampling hidden width does not match model".to_string(),
            });
        }
        scratch.hidden_bits.copy_from_slice(hidden_bits);
        self.sample_current_hidden(scratch)
    }
}
