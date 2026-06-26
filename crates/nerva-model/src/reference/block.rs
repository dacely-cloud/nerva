use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::math::{mat_vec_row_major, rms_norm_into, silu, single_token_attention};
use crate::common::shape::TransformerBlockShape;
use crate::common::validate::require_len;
use crate::reference::scratch::TransformerBlockScratch;

#[derive(Clone, Debug)]
pub struct ReferenceTransformerBlock {
    shape: TransformerBlockShape,
    rms_attn_weight: Vec<f32>,
    rms_mlp_weight: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
    rms_eps: f32,
}

impl ReferenceTransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shape: TransformerBlockShape,
        rms_attn_weight: Vec<f32>,
        rms_mlp_weight: Vec<f32>,
        w_q: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
        w_o: Vec<f32>,
        w_gate: Vec<f32>,
        w_up: Vec<f32>,
        w_down: Vec<f32>,
        rms_eps: f32,
    ) -> Result<Self> {
        shape.validate()?;
        require_len("rms_attn_weight", rms_attn_weight.len(), shape.hidden)?;
        require_len("rms_mlp_weight", rms_mlp_weight.len(), shape.hidden)?;
        require_len("w_q", w_q.len(), shape.hidden * shape.hidden)?;
        require_len("w_k", w_k.len(), shape.hidden * shape.hidden)?;
        require_len("w_v", w_v.len(), shape.hidden * shape.hidden)?;
        require_len("w_o", w_o.len(), shape.hidden * shape.hidden)?;
        require_len("w_gate", w_gate.len(), shape.intermediate * shape.hidden)?;
        require_len("w_up", w_up.len(), shape.intermediate * shape.hidden)?;
        require_len("w_down", w_down.len(), shape.hidden * shape.intermediate)?;
        if rms_eps <= 0.0 || !rms_eps.is_finite() {
            return Err(NervaError::InvalidArgument {
                reason: "rms epsilon must be positive and finite".to_string(),
            });
        }
        Ok(Self {
            shape,
            rms_attn_weight,
            rms_mlp_weight,
            w_q,
            w_k,
            w_v,
            w_o,
            w_gate,
            w_up,
            w_down,
            rms_eps,
        })
    }

    pub fn zero_for_shape(shape: TransformerBlockShape) -> Result<Self> {
        shape.validate()?;
        Self::new(
            shape,
            vec![1.0; shape.hidden],
            vec![1.0; shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.intermediate * shape.hidden],
            vec![0.0; shape.intermediate * shape.hidden],
            vec![0.0; shape.hidden * shape.intermediate],
            1e-5,
        )
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

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
