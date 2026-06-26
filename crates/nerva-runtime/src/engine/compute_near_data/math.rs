use nerva_core::types::error::{NervaError, Result};

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
    global_row_start: usize,
    global_row_end: usize,
    local_row_offset: usize,
    output: &mut [f32],
) -> Result<()> {
    if global_row_start > global_row_end || global_row_end > output.len() {
        return Err(NervaError::InvalidArgument {
            reason: "compute-near-data row range is invalid".to_string(),
        });
    }
    for global_row in global_row_start..global_row_end {
        let local_row = global_row - local_row_offset;
        let start = local_row * cols;
        let end = start + cols;
        output[global_row] = matrix[start..end]
            .iter()
            .zip(input.iter())
            .map(|(weight, value)| weight * value)
            .sum();
    }
    Ok(())
}

pub(crate) fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

pub(crate) fn hash_f32s(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}
