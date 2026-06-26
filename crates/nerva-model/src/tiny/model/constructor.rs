use nerva_core::types::error::{NervaError, Result};

use crate::common::shape::TransformerBlockShape;
use crate::common::validate::require_len;
use crate::reference::block::types::ReferenceTransformerBlock;
use crate::tiny::model::types::TinyGreedyModel;

impl TinyGreedyModel {
    pub fn new(
        vocab_size: usize,
        block: ReferenceTransformerBlock,
        embeddings: Vec<f32>,
        lm_head: Vec<f32>,
    ) -> Result<Self> {
        if vocab_size == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "tiny model vocabulary must be non-zero".to_string(),
            });
        }
        let shape = block.shape();
        require_len("embeddings", embeddings.len(), vocab_size * shape.hidden)?;
        require_len("lm_head", lm_head.len(), vocab_size * shape.hidden)?;
        Ok(Self {
            vocab_size,
            shape,
            block,
            embeddings,
            lm_head,
        })
    }

    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }
}
