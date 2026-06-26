use nerva_core::types::error::{NervaError, Result};

use crate::common::shape::TransformerBlockShape;
use crate::common::validate::require_len;
use crate::reference::block::types::ReferenceTransformerBlock;

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
}
