use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::TokenId;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::math::mat_vec_row_major;
use crate::common::token::{copy_embedding_row, greedy_argmax, require_token_in_vocab};
use crate::tiny::model::ledger::record_tiny_decode_event;
use crate::tiny::model::types::TinyGreedyModel;
use crate::tiny::output::TinyGreedyDecodeOutput;
use crate::tiny::scratch::TinyGreedyDecodeScratch;

impl TinyGreedyModel {
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
            record_tiny_decode_event(self.shape.hidden, self.vocab_size, &mut ledger);
            ledger.require_zero_hot_path_allocations()?;
            tokens.push(next_token);
            ledgers.push(ledger);
            current_token = next_token;
        }

        Ok(TinyGreedyDecodeOutput { tokens, ledgers })
    }
}
