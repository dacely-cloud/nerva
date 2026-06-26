use nerva_core::types::error::{NervaError, Result};

use crate::common::shape::TransformerBlockShape;

#[derive(Clone, Debug)]
pub struct PrecisionTransformerBlockScratch {
    shape: TransformerBlockShape,
    pub(crate) input: Vec<f32>,
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) mlp_norm: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) attn: Vec<f32>,
    pub(crate) residual: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) ff: Vec<f32>,
    pub(crate) down: Vec<f32>,
}

impl PrecisionTransformerBlockScratch {
    pub fn new(shape: TransformerBlockShape) -> Result<Self> {
        shape.validate()?;
        Ok(Self {
            shape,
            input: vec![0.0; shape.hidden],
            attn_norm: vec![0.0; shape.hidden],
            mlp_norm: vec![0.0; shape.hidden],
            q: vec![0.0; shape.hidden],
            k: vec![0.0; shape.hidden],
            v: vec![0.0; shape.hidden],
            attn: vec![0.0; shape.hidden],
            residual: vec![0.0; shape.hidden],
            gate: vec![0.0; shape.intermediate],
            up: vec![0.0; shape.intermediate],
            ff: vec![0.0; shape.intermediate],
            down: vec![0.0; shape.hidden],
        })
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

    pub(crate) fn require_shape(&self, shape: TransformerBlockShape) -> Result<()> {
        if self.shape == shape {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "precision block scratch shape does not match block shape".to_string(),
            })
        }
    }
}
