use nerva_core::types::error::Result;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::math::{mat_vec_row_major, rms_norm_into, silu, single_token_attention};
use crate::common::validate::require_len;
use crate::reference::block::types::ReferenceTransformerBlock;
use crate::reference::scratch::types::TransformerBlockScratch;

impl ReferenceTransformerBlock {
    pub fn forward_into(
        &self,
        input: &[f32],
        scratch: &mut TransformerBlockScratch,
        output: &mut [f32],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let _ = ledger;
        let shape = self.shape;
        require_len("input", input.len(), shape.hidden)?;
        require_len("output", output.len(), shape.hidden)?;
        scratch.require_shape(shape)?;

        rms_norm_into(
            input,
            &self.rms_attn_weight,
            self.rms_eps,
            &mut scratch.attn_norm,
        );
        mat_vec_row_major(&self.w_q, &scratch.attn_norm, &mut scratch.q);
        mat_vec_row_major(&self.w_k, &scratch.attn_norm, &mut scratch.k);
        mat_vec_row_major(&self.w_v, &scratch.attn_norm, &mut scratch.v);

        single_token_attention(shape, &scratch.q, &scratch.k, &scratch.v, &mut scratch.attn);
        mat_vec_row_major(&self.w_o, &scratch.attn, output);
        for (out, residual) in output.iter_mut().zip(input.iter().copied()) {
            *out += residual;
        }

        rms_norm_into(
            output,
            &self.rms_mlp_weight,
            self.rms_eps,
            &mut scratch.mlp_norm,
        );
        mat_vec_row_major(&self.w_gate, &scratch.mlp_norm, &mut scratch.gate);
        mat_vec_row_major(&self.w_up, &scratch.mlp_norm, &mut scratch.up);
        for ((ff, gate), up) in scratch
            .ff
            .iter_mut()
            .zip(scratch.gate.iter().copied())
            .zip(scratch.up.iter().copied())
        {
            *ff = silu(gate) * up;
        }
        mat_vec_row_major(&self.w_down, &scratch.ff, &mut scratch.down);
        for (out, mlp) in output.iter_mut().zip(scratch.down.iter().copied()) {
            *out += mlp;
        }

        Ok(())
    }
}
