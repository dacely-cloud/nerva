use nerva_core::types::error::{NervaError, Result};

pub(crate) fn validate_hf_metadata(
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
) -> Result<()> {
    if hidden_size == 0
        || num_hidden_layers == 0
        || num_attention_heads == 0
        || num_key_value_heads == 0
        || head_dim == 0
        || intermediate_size == 0
        || vocab_size == 0
    {
        return Err(NervaError::InvalidArgument {
            reason: "HF model metadata dimensions must be non-zero".to_string(),
        });
    }
    if num_key_value_heads > num_attention_heads {
        return Err(NervaError::InvalidArgument {
            reason: "HF KV head count cannot exceed attention head count".to_string(),
        });
    }
    if !num_attention_heads.is_multiple_of(num_key_value_heads) {
        return Err(NervaError::InvalidArgument {
            reason: "HF attention head count must be divisible by KV head count".to_string(),
        });
    }
    validate_attention_width(num_attention_heads, head_dim)?;
    validate_attention_width(num_key_value_heads, head_dim)?;
    Ok(())
}

fn validate_attention_width(heads: usize, head_dim: usize) -> Result<()> {
    heads
        .checked_mul(head_dim)
        .map(|_| ())
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: 0,
            reason: "HF attention width overflow".to_string(),
        })
}
