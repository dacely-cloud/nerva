use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;

use crate::common::shape::TransformerBlockShape;
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::precision::block::ops::encode_vec;
use crate::precision::block::validate::validate_precision_block_layout;

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
        validate_precision_block_layout(
            dtype,
            shape,
            rms_attn_weight.len(),
            rms_mlp_weight.len(),
            w_q.len(),
            w_k.len(),
            w_v.len(),
            w_o.len(),
            w_gate.len(),
            w_up.len(),
            w_down.len(),
            rms_eps,
        )?;

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
        validate_precision_block_layout(
            dtype,
            shape,
            rms_attn_weight.len(),
            rms_mlp_weight.len(),
            w_q.len(),
            w_k.len(),
            w_v.len(),
            w_o.len(),
            w_gate.len(),
            w_up.len(),
            w_down.len(),
            rms_eps,
        )?;

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
}
