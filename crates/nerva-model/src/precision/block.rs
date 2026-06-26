use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::token::TokenLedger;

use crate::common::math::{silu, single_token_attention};
use crate::common::shape::TransformerBlockShape;
use crate::common::validate::require_len;
use crate::precision::bits::{decode_f32_for_dtype, encode_f32_for_dtype};
use crate::precision::scratch::PrecisionTransformerBlockScratch;

#[derive(Clone, Debug)]
pub struct PrecisionTransformerBlock {
    dtype: DType,
    shape: TransformerBlockShape,
    rms_attn_weight: Vec<u16>,
    rms_mlp_weight: Vec<u16>,
    w_q: Vec<u16>,
    w_k: Vec<u16>,
    w_v: Vec<u16>,
    w_o: Vec<u16>,
    w_gate: Vec<u16>,
    w_up: Vec<u16>,
    w_down: Vec<u16>,
    rms_eps: f32,
}

impl PrecisionTransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new_from_f32(
        dtype: DType,
        shape: TransformerBlockShape,
        rms_attn_weight: &[f32],
        rms_mlp_weight: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        rms_eps: f32,
    ) -> Result<Self> {
        shape.validate()?;
        validate_dtype(dtype)?;
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
            dtype,
            shape,
            rms_attn_weight: encode_vec(dtype, rms_attn_weight)?,
            rms_mlp_weight: encode_vec(dtype, rms_mlp_weight)?,
            w_q: encode_vec(dtype, w_q)?,
            w_k: encode_vec(dtype, w_k)?,
            w_v: encode_vec(dtype, w_v)?,
            w_o: encode_vec(dtype, w_o)?,
            w_gate: encode_vec(dtype, w_gate)?,
            w_up: encode_vec(dtype, w_up)?,
            w_down: encode_vec(dtype, w_down)?,
            rms_eps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_from_encoded(
        dtype: DType,
        shape: TransformerBlockShape,
        rms_attn_weight: Vec<u16>,
        rms_mlp_weight: Vec<u16>,
        w_q: Vec<u16>,
        w_k: Vec<u16>,
        w_v: Vec<u16>,
        w_o: Vec<u16>,
        w_gate: Vec<u16>,
        w_up: Vec<u16>,
        w_down: Vec<u16>,
        rms_eps: f32,
    ) -> Result<Self> {
        shape.validate()?;
        validate_dtype(dtype)?;
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
            dtype,
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

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

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

fn validate_dtype(dtype: DType) -> Result<()> {
    match dtype {
        DType::F16 | DType::BF16 => Ok(()),
        _ => Err(NervaError::InvalidArgument {
            reason: "precision block supports only FP16 and BF16".to_string(),
        }),
    }
}

fn encode_vec(dtype: DType, values: &[f32]) -> Result<Vec<u16>> {
    values
        .iter()
        .copied()
        .map(|value| encode_f32_for_dtype(value, dtype))
        .collect()
}

fn decode_vec_into(dtype: DType, values: &[u16], output: &mut [f32]) -> Result<()> {
    require_len("precision decoded output", output.len(), values.len())?;
    for (out, value) in output.iter_mut().zip(values.iter().copied()) {
        *out = decode_f32_for_dtype(value, dtype)?;
    }
    Ok(())
}

fn encode_vec_into(dtype: DType, values: &[f32], output: &mut [u16]) -> Result<()> {
    require_len("precision encoded output", output.len(), values.len())?;
    for (out, value) in output.iter_mut().zip(values.iter().copied()) {
        *out = encode_f32_for_dtype(value, dtype)?;
    }
    Ok(())
}

fn rms_norm_encoded_into(
    dtype: DType,
    input: &[f32],
    weight: &[u16],
    eps: f32,
    output: &mut [f32],
) -> Result<()> {
    require_len("precision rms weight", weight.len(), input.len())?;
    require_len("precision rms output", output.len(), input.len())?;
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    for ((out, value), weight) in output
        .iter_mut()
        .zip(input.iter().copied())
        .zip(weight.iter().copied())
    {
        *out = value * scale * decode_f32_for_dtype(weight, dtype)?;
    }
    Ok(())
}

fn mat_vec_encoded_row_major(
    dtype: DType,
    matrix: &[u16],
    input: &[f32],
    output: &mut [f32],
) -> Result<()> {
    let cols = input.len();
    require_len("precision matvec matrix", matrix.len(), cols * output.len())?;
    for (row, out) in matrix.chunks_exact(cols).zip(output.iter_mut()) {
        let mut sum = 0.0f32;
        for (weight, value) in row.iter().copied().zip(input.iter().copied()) {
            sum += decode_f32_for_dtype(weight, dtype)? * value;
        }
        *out = sum;
    }
    Ok(())
}
