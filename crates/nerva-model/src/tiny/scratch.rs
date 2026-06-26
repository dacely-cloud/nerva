use nerva_core::types::error::{NervaError, Result};

use crate::common::shape::TransformerBlockShape;
use crate::reference::scratch::types::TransformerBlockScratch;

#[derive(Clone, Debug)]
pub struct TinyGreedyDecodeScratch {
    shape: TransformerBlockShape,
    vocab_size: usize,
    block_scratch: TransformerBlockScratch,
    hidden: Vec<f32>,
    block_output: Vec<f32>,
    logits: Vec<f32>,
}

impl TinyGreedyDecodeScratch {
    pub fn new(shape: TransformerBlockShape, vocab_size: usize) -> Result<Self> {
        shape.validate()?;
        if vocab_size == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "tiny greedy scratch vocabulary must be non-zero".to_string(),
            });
        }
        Ok(Self {
            shape,
            vocab_size,
            block_scratch: TransformerBlockScratch::new(shape)?,
            hidden: vec![0.0; shape.hidden],
            block_output: vec![0.0; shape.hidden],
            logits: vec![0.0; vocab_size],
        })
    }

    pub(crate) fn require_shape(
        &self,
        shape: TransformerBlockShape,
        vocab_size: usize,
    ) -> Result<()> {
        if self.shape == shape && self.vocab_size == vocab_size {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "tiny greedy scratch shape does not match model shape".to_string(),
            })
        }
    }

    pub(crate) fn hidden_mut(&mut self) -> &mut [f32] {
        &mut self.hidden
    }

    pub(crate) fn block_forward_parts(
        &mut self,
    ) -> (&[f32], &mut TransformerBlockScratch, &mut [f32]) {
        (
            &self.hidden,
            &mut self.block_scratch,
            &mut self.block_output,
        )
    }

    pub(crate) fn logit_parts_mut(&mut self) -> (&[f32], &mut [f32]) {
        (&self.block_output, &mut self.logits)
    }

    pub(crate) fn logits(&self) -> &[f32] {
        &self.logits
    }
}
