use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

use crate::common::math::sigmoid;
use crate::common::rope::validate_rope;
use crate::common::shape::TransformerBlockShape;
use crate::common::validate::require_len;
use crate::precision::block::ops::mat_vec_encoded_row_major;

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
    w_q_gate: Option<Vec<u16>>,
    w_k: Vec<u16>,
    q_norm_weight: Option<Vec<u16>>,
    k_norm_weight: Option<Vec<u16>>,
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
    pub w_q_gate: Option<&'a [u16]>,
    pub w_k: &'a [u16],
    pub q_norm_weight: Option<&'a [u16]>,
    pub k_norm_weight: Option<&'a [u16]>,
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
            w_q_gate: self.w_q_gate.as_deref(),
            w_k: &self.w_k,
            q_norm_weight: self.q_norm_weight.as_deref(),
            k_norm_weight: self.k_norm_weight.as_deref(),
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

    pub fn with_query_gate_projection(mut self, w_q_gate: Vec<u16>) -> Result<Self> {
        require_len(
            "q_proj gate",
            w_q_gate.len(),
            self.shape.attention_hidden() * self.shape.hidden,
        )?;
        if self.shape.intermediate < self.shape.attention_hidden() {
            return Err(NervaError::InvalidArgument {
                reason: "q_proj gate requires attention-hidden scratch capacity".to_string(),
            });
        }
        self.w_q_gate = Some(w_q_gate);
        Ok(self)
    }

    pub(crate) fn apply_query_gate_to_attention(
        &self,
        attn_norm: &[f32],
        attn: &mut [f32],
        scratch_gate: &mut [f32],
    ) -> Result<()> {
        let Some(w_q_gate) = self.w_q_gate.as_deref() else {
            return Ok(());
        };
        let attention_hidden = self.shape.attention_hidden();
        require_len("q_proj gate input", attn_norm.len(), self.shape.hidden)?;
        require_len("q_proj gate attention", attn.len(), attention_hidden)?;
        if scratch_gate.len() < attention_hidden {
            return Err(NervaError::InvalidArgument {
                reason: "q_proj gate scratch is smaller than attention hidden".to_string(),
            });
        }
        let gate = &mut scratch_gate[..attention_hidden];
        mat_vec_encoded_row_major(self.dtype, w_q_gate, attn_norm, gate)?;
        for (attn, gate) in attn.iter_mut().zip(gate.iter().copied()) {
            *attn *= sigmoid(gate);
        }
        Ok(())
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
        self = self.with_optional_attention_biases(
            Some(q_bias),
            Some(k_bias),
            Some(v_bias),
            Some(o_bias),
        )?;
        Ok(self)
    }

    pub fn with_optional_attention_biases(
        mut self,
        q_bias: Option<Vec<u16>>,
        k_bias: Option<Vec<u16>>,
        v_bias: Option<Vec<u16>>,
        o_bias: Option<Vec<u16>>,
    ) -> Result<Self> {
        if let Some(q_bias) = q_bias.as_deref() {
            require_len("q_proj.bias", q_bias.len(), self.shape.attention_hidden())?;
        }
        if let Some(k_bias) = k_bias.as_deref() {
            require_len("k_proj.bias", k_bias.len(), self.shape.kv_hidden())?;
        }
        if let Some(v_bias) = v_bias.as_deref() {
            require_len("v_proj.bias", v_bias.len(), self.shape.kv_hidden())?;
        }
        if let Some(o_bias) = o_bias.as_deref() {
            require_len("o_proj.bias", o_bias.len(), self.shape.hidden)?;
        }
        self.q_bias = q_bias;
        self.k_bias = k_bias;
        self.v_bias = v_bias;
        self.o_bias = o_bias;
        Ok(self)
    }

    pub fn with_qk_norm(
        mut self,
        q_norm_weight: Vec<u16>,
        k_norm_weight: Vec<u16>,
    ) -> Result<Self> {
        require_len("q_norm.weight", q_norm_weight.len(), self.shape.head_dim())?;
        require_len("k_norm.weight", k_norm_weight.len(), self.shape.head_dim())?;
        self.q_norm_weight = Some(q_norm_weight);
        self.k_norm_weight = Some(k_norm_weight);
        Ok(self)
    }
}
