use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::attention::block::KvAttentionBlock;
use crate::attention::exact::run::exact_blockwise_attention_into;
use crate::common::rope::{apply_rotary_to_key, apply_rotary_to_query};
use crate::common::validate::require_len;
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::precision::block::ops::{
    decode_vec_into, mat_vec_encoded_row_major, rms_norm_encoded_into,
};
use crate::precision::scratch::PrecisionTransformerBlockKvScratch;

use super::kv_finish::finish_attention_and_mlp;

impl PrecisionTransformerBlock {
    pub fn forward_prefill_sequence_into(
        &self,
        input: &[u16],
        token_count: usize,
        scratch: &mut PrecisionTransformerBlockKvScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let values = self.require_sequence_io(input, output, token_count, scratch)?;
        scratch.reset();
        for row in 0..token_count {
            let start = row * self.shape.hidden;
            append_kv_from_input(self, &input[start..start + self.shape.hidden], row, scratch)?;
        }
        for row in 0..token_count {
            let start = row * self.shape.hidden;
            forward_with_visible_kv(
                self,
                &input[start..start + self.shape.hidden],
                row + 1,
                row,
                scratch,
                &mut output[start..start + self.shape.hidden],
                ledger,
            )?;
        }
        scratch.set_len(token_count);
        require_len("precision prefill output", output.len(), values)
    }

    pub fn forward_decode_with_kv_into(
        &self,
        input: &[u16],
        scratch: &mut PrecisionTransformerBlockKvScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        require_len("precision decode input", input.len(), self.shape.hidden)?;
        require_len("precision decode output", output.len(), self.shape.hidden)?;
        let next_len = scratch
            .len()
            .checked_add(1)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "precision KV length overflow".to_string(),
            })?;
        scratch.require_capacity(self.shape, next_len)?;
        let position = scratch.len();
        append_kv_from_input(self, input, position, scratch)?;
        forward_with_visible_kv(self, input, next_len, position, scratch, output, ledger)
    }

    fn require_sequence_io(
        &self,
        input: &[u16],
        output: &[u16],
        token_count: usize,
        scratch: &PrecisionTransformerBlockKvScratch,
    ) -> Result<usize> {
        if token_count == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "precision prefill requires at least one token".to_string(),
            });
        }
        scratch.require_capacity(self.shape, token_count)?;
        let values = token_count.checked_mul(self.shape.hidden).ok_or_else(|| {
            NervaError::InvalidArgument {
                reason: "precision prefill token count overflow".to_string(),
            }
        })?;
        require_len("precision prefill input", input.len(), values)?;
        require_len("precision prefill output", output.len(), values)?;
        Ok(values)
    }
}

fn append_kv_from_input(
    block: &PrecisionTransformerBlock,
    input: &[u16],
    position: usize,
    scratch: &mut PrecisionTransformerBlockKvScratch,
) -> Result<()> {
    let start = scratch.len() * block.shape.kv_hidden();
    let end = start + block.shape.kv_hidden();
    decode_vec_into(block.dtype, input, &mut scratch.token.input)?;
    rms_norm_encoded_into(
        block.dtype,
        &scratch.token.input,
        &block.rms_attn_weight,
        block.rms_eps,
        &mut scratch.token.attn_norm,
    )?;
    mat_vec_encoded_row_major(
        block.dtype,
        &block.w_k,
        &scratch.token.attn_norm,
        &mut scratch.token.k,
    )?;
    mat_vec_encoded_row_major(
        block.dtype,
        &block.w_v,
        &scratch.token.attn_norm,
        &mut scratch.token.v,
    )?;
    if let Some(theta) = block.rope_theta {
        apply_rotary_to_key(block.shape, position, theta, &mut scratch.token.k)?;
    }
    scratch.keys[start..end].copy_from_slice(&scratch.token.k);
    scratch.values[start..end].copy_from_slice(&scratch.token.v);
    scratch.set_len(scratch.len() + 1);
    Ok(())
}

fn forward_with_visible_kv(
    block: &PrecisionTransformerBlock,
    input: &[u16],
    visible_tokens: usize,
    position: usize,
    scratch: &mut PrecisionTransformerBlockKvScratch,
    output: &mut [u16],
    ledger: &mut TokenLedger,
) -> Result<()> {
    decode_vec_into(block.dtype, input, &mut scratch.token.input)?;
    rms_norm_encoded_into(
        block.dtype,
        &scratch.token.input,
        &block.rms_attn_weight,
        block.rms_eps,
        &mut scratch.token.attn_norm,
    )?;
    mat_vec_encoded_row_major(
        block.dtype,
        &block.w_q,
        &scratch.token.attn_norm,
        &mut scratch.token.q,
    )?;
    if let Some(theta) = block.rope_theta {
        apply_rotary_to_query(block.shape, position, theta, &mut scratch.token.q)?;
    }
    let values = visible_tokens * block.shape.kv_hidden();
    let kv = [KvAttentionBlock::new(
        &scratch.keys[..values],
        &scratch.values[..values],
        visible_tokens,
        MemoryTier::Dram,
    )];
    exact_blockwise_attention_into(
        block.shape,
        &scratch.token.q,
        &kv,
        &mut scratch.attention,
        &mut scratch.token.attn,
        ledger,
    )?;
    finish_attention_and_mlp(block, scratch, output)
}
