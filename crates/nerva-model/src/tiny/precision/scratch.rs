use nerva_core::types::error::{NervaError, Result};

use crate::common::shape::TransformerBlockShape;
use crate::precision::scratch::PrecisionTransformerBlockScratch;

#[derive(Clone, Debug)]
pub struct TinyPrecisionGreedyDecodeScratch {
    shape: TransformerBlockShape,
    vocab_size: usize,
    pub(crate) block_scratch: PrecisionTransformerBlockScratch,
    pub(crate) hidden_bits: Vec<u16>,
    pub(crate) block_output_bits: Vec<u16>,
    pub(crate) decoded_output: Vec<f32>,
    pub(crate) logits: Vec<f32>,
}

impl TinyPrecisionGreedyDecodeScratch {
    pub fn new(shape: TransformerBlockShape, vocab_size: usize) -> Result<Self> {
        shape.validate()?;
        if vocab_size == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "tiny precision greedy scratch vocabulary must be non-zero".to_string(),
            });
        }
        Ok(Self {
            shape,
            vocab_size,
            block_scratch: PrecisionTransformerBlockScratch::new(shape)?,
            hidden_bits: vec![0; shape.hidden],
            block_output_bits: vec![0; shape.hidden],
            decoded_output: vec![0.0; shape.hidden],
            logits: vec![0.0; vocab_size],
        })
    }

    pub fn require_shape(&self, shape: TransformerBlockShape, vocab_size: usize) -> Result<()> {
        if self.shape == shape && self.vocab_size == vocab_size {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "tiny precision scratch shape does not match model shape".to_string(),
            })
        }
    }
}
