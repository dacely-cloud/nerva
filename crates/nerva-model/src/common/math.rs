use crate::common::shape::TransformerBlockShape;

pub(crate) fn rms_norm_into(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    for ((out, value), weight) in output
        .iter_mut()
        .zip(input.iter().copied())
        .zip(weight.iter().copied())
    {
        *out = value * scale * weight;
    }
}

pub(crate) fn mat_vec_row_major(matrix: &[f32], input: &[f32], output: &mut [f32]) {
    let cols = input.len();
    for (row, out) in matrix.chunks_exact(cols).zip(output.iter_mut()) {
        *out = row
            .iter()
            .zip(input.iter())
            .map(|(weight, value)| weight * value)
            .sum();
    }
}

pub(crate) fn mat_vec_row_range(
    matrix: &[f32],
    input: &[f32],
    cols: usize,
    row_start: usize,
    row_end: usize,
    output: &mut [f32],
) -> nerva_core::types::error::Result<()> {
    if row_start > row_end || row_end > output.len() {
        return Err(nerva_core::types::error::NervaError::InvalidArgument {
            reason: "matvec row range is invalid".to_string(),
        });
    }
    for row_index in row_start..row_end {
        let start = row_index * cols;
        let end = start + cols;
        output[row_index] = matrix[start..end]
            .iter()
            .zip(input.iter())
            .map(|(weight, value)| weight * value)
            .sum();
    }
    Ok(())
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| left * right)
        .sum()
}

pub(crate) fn single_token_attention(
    shape: TransformerBlockShape,
    _q: &[f32],
    _k: &[f32],
    v: &[f32],
    output: &mut [f32],
) {
    let head_dim = shape.head_dim();
    for head in 0..shape.heads {
        let out_start = head * head_dim;
        let out_end = out_start + head_dim;
        let kv_head = shape.kv_head_for_attention_head(head);
        let value_start = kv_head * head_dim;
        let value_end = value_start + head_dim;
        output[out_start..out_end].copy_from_slice(&v[value_start..value_end]);
    }
}

pub(crate) fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub(crate) fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}
