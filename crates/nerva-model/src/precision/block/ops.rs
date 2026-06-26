use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;

use crate::common::validate::require_len;
use crate::precision::bits::{decode_f32_for_dtype, encode_f32_for_dtype};

pub(crate) fn encode_vec(dtype: DType, values: &[f32]) -> Result<Vec<u16>> {
    values
        .iter()
        .copied()
        .map(|value| encode_f32_for_dtype(value, dtype))
        .collect()
}

pub(crate) fn decode_vec_into(dtype: DType, values: &[u16], output: &mut [f32]) -> Result<()> {
    require_len("precision decoded output", output.len(), values.len())?;
    for (out, value) in output.iter_mut().zip(values.iter().copied()) {
        *out = decode_f32_for_dtype(value, dtype)?;
    }
    Ok(())
}

pub(crate) fn encode_vec_into(dtype: DType, values: &[f32], output: &mut [u16]) -> Result<()> {
    require_len("precision encoded output", output.len(), values.len())?;
    for (out, value) in output.iter_mut().zip(values.iter().copied()) {
        *out = encode_f32_for_dtype(value, dtype)?;
    }
    Ok(())
}

pub(crate) fn rms_norm_encoded_into(
    dtype: DType,
    input: &[f32],
    weight: &[u16],
    eps: f32,
    output: &mut [f32],
) -> Result<()> {
    require_len("precision rms weight", weight.len(), input.len())?;
    require_len("precision rms output", output.len(), input.len())?;
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    for ((out, value), weight) in output
        .iter_mut()
        .zip(input.iter().copied())
        .zip(weight.iter().copied())
    {
        *out = value * scale * decode_f32_for_dtype(weight, dtype)?;
    }
    Ok(())
}

pub(crate) fn mat_vec_encoded_row_major(
    dtype: DType,
    matrix: &[u16],
    input: &[f32],
    output: &mut [f32],
) -> Result<()> {
    let cols = input.len();
    require_len("precision matvec matrix", matrix.len(), cols * output.len())?;
    for (row, out) in matrix.chunks_exact(cols).zip(output.iter_mut()) {
        let mut sum = 0.0f32;
        for (weight, value) in row.iter().copied().zip(input.iter().copied()) {
            sum += decode_f32_for_dtype(weight, dtype)? * value;
        }
        *out = sum;
    }
    Ok(())
}
