use nerva_core::types::error::{NervaError, Result};

use crate::attention::scratch::BlockwiseAttentionScratch;
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

#[derive(Clone, Debug)]
pub struct PrecisionTransformerBlockKvScratch {
    shape: TransformerBlockShape,
    capacity_tokens: usize,
    len: usize,
    pub(crate) token: PrecisionTransformerBlockScratch,
    pub(crate) attention: BlockwiseAttentionScratch,
    pub(crate) keys: Vec<f32>,
    pub(crate) values: Vec<f32>,
}

impl PrecisionTransformerBlockKvScratch {
    pub fn new(shape: TransformerBlockShape, capacity_tokens: usize) -> Result<Self> {
        shape.validate()?;
        if capacity_tokens == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "precision KV scratch capacity must be non-zero".to_string(),
            });
        }
        Ok(Self {
            shape,
            capacity_tokens,
            len: 0,
            token: PrecisionTransformerBlockScratch::new(shape)?,
            attention: BlockwiseAttentionScratch::new(shape)?,
            keys: vec![0.0; capacity_tokens * shape.hidden],
            values: vec![0.0; capacity_tokens * shape.hidden],
        })
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }

    pub(crate) fn require_capacity(
        &self,
        shape: TransformerBlockShape,
        required_tokens: usize,
    ) -> Result<()> {
        if self.shape != shape {
            return Err(NervaError::InvalidArgument {
                reason: "precision KV scratch shape does not match block shape".to_string(),
            });
        }
        if required_tokens > self.capacity_tokens {
            return Err(NervaError::InvalidArgument {
                reason: "precision KV scratch capacity is too small".to_string(),
            });
        }
        Ok(())
    }

    pub(crate) fn set_len(&mut self, len: usize) {
        self.len = len;
    }
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
