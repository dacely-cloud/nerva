use nerva_core::types::dtype::DType;

use crate::common::shape::TransformerBlockShape;

mod constructor;
mod forward;

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
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }
}
