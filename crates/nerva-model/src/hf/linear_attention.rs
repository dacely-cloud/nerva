use nerva_core::types::error::{NervaError, Result};

use crate::hf::metadata::HfModelMetadata;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ConvStateLayout {
    StateLenDim,
    DimStateLen,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Qwen35GatedDeltaNetStateShape {
    pub conv_state: (usize, usize),
    pub recurrent_state: (usize, usize, usize),
}

impl Qwen35GatedDeltaNetStateShape {
    pub const fn conv_elements(self) -> usize {
        self.conv_state.0 * self.conv_state.1
    }

    pub const fn recurrent_elements(self) -> usize {
        self.recurrent_state.0 * self.recurrent_state.1 * self.recurrent_state.2
    }

    pub const fn total_elements(self) -> usize {
        self.conv_elements() + self.recurrent_elements()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Qwen35GatedDeltaNetSpec {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
}

impl Qwen35GatedDeltaNetSpec {
    pub fn from_metadata(metadata: &HfModelMetadata) -> Result<Option<Self>> {
        if !metadata.has_linear_attention_layers() {
            return Ok(None);
        }
        Ok(Some(Self {
            key_heads: required(metadata.linear_num_key_heads, "linear_num_key_heads")?,
            value_heads: required(metadata.linear_num_value_heads, "linear_num_value_heads")?,
            key_head_dim: required(metadata.linear_key_head_dim, "linear_key_head_dim")?,
            value_head_dim: required(metadata.linear_value_head_dim, "linear_value_head_dim")?,
            conv_kernel: required(metadata.linear_conv_kernel_dim, "linear_conv_kernel_dim")?,
        }))
    }

    pub fn conv_dim(self) -> Result<usize> {
        let key_dim = self
            .key_heads
            .checked_mul(self.key_head_dim)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: self.key_heads,
                reason: "Qwen3.5 GDN key dimension overflow".to_string(),
            })?;
        let value_dim = self
            .value_heads
            .checked_mul(self.value_head_dim)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: self.value_heads,
                reason: "Qwen3.5 GDN value dimension overflow".to_string(),
            })?;
        key_dim
            .checked_mul(2)
            .and_then(|dim| dim.checked_add(value_dim))
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: key_dim,
                reason: "Qwen3.5 GDN conv dimension overflow".to_string(),
            })
    }

    pub fn state_shape(
        self,
        tensor_parallel_size: usize,
        speculative_tokens: usize,
        conv_layout: ConvStateLayout,
    ) -> Result<Qwen35GatedDeltaNetStateShape> {
        if tensor_parallel_size == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "Qwen3.5 GDN tensor_parallel_size must be non-zero".to_string(),
            });
        }
        if self.key_heads % tensor_parallel_size != 0
            || self.value_heads % tensor_parallel_size != 0
        {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "Qwen3.5 GDN head counts must divide tensor_parallel_size {tensor_parallel_size}"
                ),
            });
        }
        let conv_state_len = self
            .conv_kernel
            .checked_sub(1)
            .and_then(|len| len.checked_add(speculative_tokens))
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "Qwen3.5 GDN conv state length overflow".to_string(),
            })?;
        let conv_dim = self.conv_dim()? / tensor_parallel_size;
        let conv_state = match conv_layout {
            ConvStateLayout::StateLenDim => (conv_state_len, conv_dim),
            ConvStateLayout::DimStateLen => (conv_dim, conv_state_len),
        };
        Ok(Qwen35GatedDeltaNetStateShape {
            conv_state,
            recurrent_state: (
                self.value_heads / tensor_parallel_size,
                self.value_head_dim,
                self.key_head_dim,
            ),
        })
    }
}

fn required(value: Option<usize>, name: &'static str) -> Result<usize> {
    value.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("Qwen3.5 linear_attention is missing {name}"),
    })
}
