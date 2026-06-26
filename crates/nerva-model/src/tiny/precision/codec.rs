use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;
use nerva_core::types::id::TokenId;

use crate::common::token::require_token_in_vocab;
use crate::common::validate::require_len;
use crate::precision::bits::{decode_f32_for_dtype, encode_f32_for_dtype};

pub fn encode_slice(dtype: DType, values: &[f32]) -> Result<Vec<u16>> {
    values
        .iter()
        .copied()
        .map(|value| encode_f32_for_dtype(value, dtype))
        .collect()
}

pub fn copy_encoded_embedding_row(
    embeddings: &[u16],
    hidden: usize,
    token: TokenId,
    output: &mut [u16],
) -> Result<()> {
    require_token_in_vocab(token, embeddings.len() / hidden)?;
    require_len("encoded embedding output", output.len(), hidden)?;
    let start = token.0 as usize * hidden;
    let end = start + hidden;
    output.copy_from_slice(&embeddings[start..end]);
    Ok(())
}

pub fn decode_slice_into(dtype: DType, values: &[u16], output: &mut [f32]) -> Result<()> {
    require_len("decoded precision output", output.len(), values.len())?;
    for (out, value) in output.iter_mut().zip(values.iter().copied()) {
        *out = decode_f32_for_dtype(value, dtype)?;
    }
    Ok(())
}

pub fn encoded_lm_head_into(
    dtype: DType,
    lm_head: &[u16],
    input: &[f32],
    logits: &mut [f32],
) -> Result<()> {
    require_len("encoded lm_head", lm_head.len(), logits.len() * input.len())?;
    for (row, logit) in lm_head.chunks_exact(input.len()).zip(logits.iter_mut()) {
        let mut sum = 0.0f32;
        for (weight, value) in row.iter().copied().zip(input.iter().copied()) {
            sum += decode_f32_for_dtype(weight, dtype)? * value;
        }
        *logit = sum;
    }
    Ok(())
}
