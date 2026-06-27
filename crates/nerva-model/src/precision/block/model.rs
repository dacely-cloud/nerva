use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;

use crate::common::rope::validate_rope;
use crate::common::shape::TransformerBlockShape;

mod constructor;
mod forward;
mod kv;
mod kv_finish;

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
    rope_theta: Option<f32>,
}

impl PrecisionTransformerBlock {
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

    pub const fn rope_theta(&self) -> Option<f32> {
        self.rope_theta
    }

    pub fn with_rope_theta(mut self, rope_theta: Option<f32>) -> Result<Self> {
        if let Some(theta) = rope_theta {
            validate_rope(self.shape, theta)?;
        }
        self.rope_theta = rope_theta;
        Ok(self)
    }
}
