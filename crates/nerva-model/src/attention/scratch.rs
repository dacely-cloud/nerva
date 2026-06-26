use nerva_core::types::error::{NervaError, Result};

use crate::common::shape::TransformerBlockShape;

#[derive(Clone, Debug)]
pub struct BlockwiseAttentionScratch {
    pub(crate) shape: TransformerBlockShape,
    pub(crate) local_output: Vec<f32>,
    pub(crate) global_m: Vec<f32>,
    pub(crate) global_l: Vec<f32>,
}

impl BlockwiseAttentionScratch {
    pub fn new(shape: TransformerBlockShape) -> Result<Self> {
        shape.validate()?;
        Ok(Self {
            shape,
            local_output: vec![0.0; shape.hidden],
            global_m: vec![f32::NEG_INFINITY; shape.heads],
            global_l: vec![0.0; shape.heads],
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
                reason: "blockwise attention scratch shape does not match".to_string(),
            })
        }
    }
}
