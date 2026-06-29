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
    pub(crate) gdn: Option<PrecisionGatedDeltaNetScratch>,
}

#[derive(Clone, Debug)]
pub(crate) struct PrecisionGatedDeltaNetScratch {
    conv_dim: usize,
    conv_kernel: usize,
    value_heads: usize,
    value_head_dim: usize,
    key_head_dim: usize,
    pub(crate) conv_state: Vec<f32>,
    pub(crate) recurrent_state: Vec<f32>,
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
            keys: vec![0.0; capacity_tokens * shape.kv_hidden()],
            values: vec![0.0; capacity_tokens * shape.kv_hidden()],
            gdn: None,
        })
    }

    pub fn reset(&mut self) {
        self.len = 0;
        if let Some(gdn) = &mut self.gdn {
            gdn.reset();
        }
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

    pub(crate) fn ensure_gated_delta_net_state(
        &mut self,
        conv_dim: usize,
        conv_kernel: usize,
        value_heads: usize,
        value_head_dim: usize,
        key_head_dim: usize,
    ) -> Result<()> {
        if conv_dim == 0
            || conv_kernel == 0
            || value_heads == 0
            || value_head_dim == 0
            || key_head_dim == 0
        {
            return Err(NervaError::InvalidArgument {
                reason: "precision GatedDeltaNet state dimensions must be non-zero".to_string(),
            });
        }
        let matches = self.gdn.as_ref().is_some_and(|gdn| {
            gdn.conv_dim == conv_dim
                && gdn.conv_kernel == conv_kernel
                && gdn.value_heads == value_heads
                && gdn.value_head_dim == value_head_dim
                && gdn.key_head_dim == key_head_dim
        });
        if !matches {
            self.gdn = Some(PrecisionGatedDeltaNetScratch::new(
                conv_dim,
                conv_kernel,
                value_heads,
                value_head_dim,
                key_head_dim,
            )?);
        }
        Ok(())
    }
}

impl PrecisionGatedDeltaNetScratch {
    fn new(
        conv_dim: usize,
        conv_kernel: usize,
        value_heads: usize,
        value_head_dim: usize,
        key_head_dim: usize,
    ) -> Result<Self> {
        let conv_state_len = conv_dim
            .checked_mul(conv_kernel.saturating_sub(1))
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: conv_dim,
                reason: "precision GatedDeltaNet conv state size overflow".to_string(),
            })?;
        let recurrent_state_len = value_heads
            .checked_mul(value_head_dim)
            .and_then(|value| value.checked_mul(key_head_dim))
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: value_heads,
                reason: "precision GatedDeltaNet recurrent state size overflow".to_string(),
            })?;
        Ok(Self {
            conv_dim,
            conv_kernel,
            value_heads,
            value_head_dim,
            key_head_dim,
            conv_state: vec![0.0; conv_state_len],
            recurrent_state: vec![0.0; recurrent_state_len],
        })
    }

    fn reset(&mut self) {
        self.conv_state.fill(0.0);
        self.recurrent_state.fill(0.0);
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
            q: vec![0.0; shape.attention_hidden()],
            k: vec![0.0; shape.kv_hidden()],
            v: vec![0.0; shape.kv_hidden()],
            attn: vec![0.0; shape.attention_hidden()],
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
