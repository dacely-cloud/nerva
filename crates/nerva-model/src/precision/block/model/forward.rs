use nerva_core::types::error::Result;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::math::{silu, single_token_attention};
use crate::common::validate::require_len;
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::precision::block::ops::{
    decode_vec_into, encode_vec_into, mat_vec_encoded_row_major, rms_norm_encoded_into,
};
use crate::precision::scratch::PrecisionTransformerBlockScratch;

impl PrecisionTransformerBlock {
    pub fn forward_into(
        &self,
        input: &[u16],
        scratch: &mut PrecisionTransformerBlockScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let _ = ledger;
        let shape = self.shape;
        require_len("precision input", input.len(), shape.hidden)?;
        require_len("precision output", output.len(), shape.hidden)?;
        scratch.require_shape(shape)?;

        decode_vec_into(self.dtype, input, &mut scratch.input)?;
        rms_norm_encoded_into(
            self.dtype,
            &scratch.input,
            &self.rms_attn_weight,
            self.rms_eps,
            &mut scratch.attn_norm,
        )?;
        mat_vec_encoded_row_major(self.dtype, &self.w_q, &scratch.attn_norm, &mut scratch.q)?;
        mat_vec_encoded_row_major(self.dtype, &self.w_k, &scratch.attn_norm, &mut scratch.k)?;
        mat_vec_encoded_row_major(self.dtype, &self.w_v, &scratch.attn_norm, &mut scratch.v)?;

        single_token_attention(shape, &scratch.q, &scratch.k, &scratch.v, &mut scratch.attn);
        mat_vec_encoded_row_major(self.dtype, &self.w_o, &scratch.attn, &mut scratch.residual)?;
        for (out, residual) in scratch
            .residual
            .iter_mut()
            .zip(scratch.input.iter().copied())
        {
            *out += residual;
        }

        rms_norm_encoded_into(
            self.dtype,
            &scratch.residual,
            &self.rms_mlp_weight,
            self.rms_eps,
            &mut scratch.mlp_norm,
        )?;
        mat_vec_encoded_row_major(
            self.dtype,
            &self.w_gate,
            &scratch.mlp_norm,
            &mut scratch.gate,
        )?;
        mat_vec_encoded_row_major(self.dtype, &self.w_up, &scratch.mlp_norm, &mut scratch.up)?;
        for ((ff, gate), up) in scratch
            .ff
            .iter_mut()
            .zip(scratch.gate.iter().copied())
            .zip(scratch.up.iter().copied())
        {
            *ff = silu(gate) * up;
        }
        mat_vec_encoded_row_major(self.dtype, &self.w_down, &scratch.ff, &mut scratch.down)?;
        for (out, mlp) in scratch
            .residual
            .iter_mut()
            .zip(scratch.down.iter().copied())
        {
            *out += mlp;
        }
        encode_vec_into(self.dtype, &scratch.residual, output)?;

        Ok(())
    }
}
