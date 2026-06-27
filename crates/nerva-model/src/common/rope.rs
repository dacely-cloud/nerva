use nerva_core::types::error::{NervaError, Result};

use crate::common::shape::TransformerBlockShape;
use crate::common::validate::require_len;

pub(crate) fn apply_rotary_to_query_key(
    shape: TransformerBlockShape,
    position: usize,
    theta: f32,
    query: &mut [f32],
    key: &mut [f32],
) -> Result<()> {
    require_len("RoPE query", query.len(), shape.hidden)?;
    require_len("RoPE key", key.len(), shape.kv_hidden())?;
    apply_rotary_to_query(shape, position, theta, query)?;
    apply_rotary_to_key(shape, position, theta, key)
}

pub(crate) fn apply_rotary_to_query(
    shape: TransformerBlockShape,
    position: usize,
    theta: f32,
    query: &mut [f32],
) -> Result<()> {
    require_len("RoPE query", query.len(), shape.hidden)?;
    apply_rotary_to_heads(shape.head_dim(), shape.heads, position, theta, query)
}

pub(crate) fn apply_rotary_to_key(
    shape: TransformerBlockShape,
    position: usize,
    theta: f32,
    key: &mut [f32],
) -> Result<()> {
    require_len("RoPE key", key.len(), shape.kv_hidden())?;
    apply_rotary_to_heads(shape.head_dim(), shape.kv_heads, position, theta, key)
}

pub(crate) fn validate_rope(shape: TransformerBlockShape, theta: f32) -> Result<()> {
    shape.validate()?;
    if shape.head_dim() % 2 != 0 {
        return Err(NervaError::InvalidArgument {
            reason: "RoPE requires an even head dimension".to_string(),
        });
    }
    if theta <= 0.0 || !theta.is_finite() {
        return Err(NervaError::InvalidArgument {
            reason: "RoPE theta must be positive and finite".to_string(),
        });
    }
    Ok(())
}

fn apply_rotary_to_heads(
    head_dim: usize,
    heads: usize,
    position: usize,
    theta: f32,
    values: &mut [f32],
) -> Result<()> {
    let shape = TransformerBlockShape::new(head_dim, 1, 1);
    validate_rope(shape, theta)?;
    let half = head_dim / 2;
    for head in 0..heads {
        let start = head * head_dim;
        for offset in 0..half {
            let first = start + offset;
            let second = first + half;
            let exponent = (2 * offset) as f32 / head_dim as f32;
            let angle = position as f32 / theta.powf(exponent);
            let (sin, cos) = angle.sin_cos();
            let left = values[first];
            let right = values[second];
            values[first] = left * cos - right * sin;
            values[second] = right * cos + left * sin;
        }
    }
    Ok(())
}
