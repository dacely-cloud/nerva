use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;

use crate::common::rope::validate_rope;
use crate::common::shape::TransformerBlockShape;
use crate::common::validate::require_len;

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
    q_bias: Option<Vec<u16>>,
    k_bias: Option<Vec<u16>>,
    v_bias: Option<Vec<u16>>,
    o_bias: Option<Vec<u16>>,
    w_gate: Vec<u16>,
    w_up: Vec<u16>,
    w_down: Vec<u16>,
    rms_eps: f32,
    rope_theta: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct PrecisionTransformerBlockEncodedView<'a> {
    pub dtype: DType,
    pub shape: TransformerBlockShape,
    pub rms_attn_weight: &'a [u16],
    pub rms_mlp_weight: &'a [u16],
    pub w_q: &'a [u16],
    pub w_k: &'a [u16],
    pub w_v: &'a [u16],
    pub w_o: &'a [u16],
    pub q_bias: Option<&'a [u16]>,
    pub k_bias: Option<&'a [u16]>,
    pub v_bias: Option<&'a [u16]>,
    pub o_bias: Option<&'a [u16]>,
    pub w_gate: &'a [u16],
    pub w_up: &'a [u16],
    pub w_down: &'a [u16],
    pub rms_eps: f32,
    pub rope_theta: Option<f32>,
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

    pub fn encoded_view(&self) -> PrecisionTransformerBlockEncodedView<'_> {
        PrecisionTransformerBlockEncodedView {
            dtype: self.dtype,
            shape: self.shape,
            rms_attn_weight: &self.rms_attn_weight,
            rms_mlp_weight: &self.rms_mlp_weight,
            w_q: &self.w_q,
            w_k: &self.w_k,
            w_v: &self.w_v,
            w_o: &self.w_o,
            q_bias: self.q_bias.as_deref(),
            k_bias: self.k_bias.as_deref(),
            v_bias: self.v_bias.as_deref(),
            o_bias: self.o_bias.as_deref(),
            w_gate: &self.w_gate,
            w_up: &self.w_up,
            w_down: &self.w_down,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta,
        }
    }

    pub fn with_rope_theta(mut self, rope_theta: Option<f32>) -> Result<Self> {
        if let Some(theta) = rope_theta {
            validate_rope(self.shape, theta)?;
        }
        self.rope_theta = rope_theta;
        Ok(self)
    }

    pub fn with_attention_biases(
        mut self,
        q_bias: Vec<u16>,
        k_bias: Vec<u16>,
        v_bias: Vec<u16>,
        o_bias: Vec<u16>,
    ) -> Result<Self> {
        require_len("q_proj.bias", q_bias.len(), self.shape.attention_hidden())?;
        require_len("k_proj.bias", k_bias.len(), self.shape.kv_hidden())?;
        require_len("v_proj.bias", v_bias.len(), self.shape.kv_hidden())?;
        require_len("o_proj.bias", o_bias.len(), self.shape.hidden)?;
        self.q_bias = Some(q_bias);
        self.k_bias = Some(k_bias);
        self.v_bias = Some(v_bias);
        self.o_bias = Some(o_bias);
        Ok(self)
    }
}
