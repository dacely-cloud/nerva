use nerva_core::types::error::{NervaError, Result};

use crate::common::shape::TransformerBlockShape;
use crate::reference::scratch::types::TransformerBlockScratch;

impl TransformerBlockScratch {
    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

    pub(crate) fn require_shape(&self, shape: TransformerBlockShape) -> Result<()> {
        if self.shape == shape {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "transformer block scratch shape does not match block shape".to_string(),
            })
        }
    }
}
