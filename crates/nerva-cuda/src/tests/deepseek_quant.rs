use crate::deepseek_quant::dequant::{
    deepseek_fp8_e4m3fn_e8m0_dequant, deepseek_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens,
    deepseek_fp8_e4m3fn_f32_scale_encoded_gemm_tokens,
    deepseek_fp8_e4m3fn_f32_scale_encoded_matvec, deepseek_fp8_e4m3fn_f32_scale_matvec,
    deepseek_mxfp4_e2m1_e8m0_dequant,
};
use crate::deepseek_quant::inv_rope::{
    CudaDeepSeekFusedInvRopeFp8QuantSummary, deepseek_fused_inv_rope_fp8_quant,
};
use crate::deepseek_quant::probe::{deepseek_fused_inv_rope_fp8_quant_smoke, deepseek_quant_smoke};
use crate::deepseek_quant::summary::CudaDeepSeekQuantSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn deepseek_quant_summary_serializes_layout_and_mismatches() {
    let summary = CudaDeepSeekQuantSummary {
        status: SmokeStatus::Ok,
        fp8_rows: 3,
        fp8_cols: 4,
        fp8_block_rows: 2,
        fp8_block_cols: 2,
        mxfp4_rows: 2,
        mxfp4_packed_cols: 4,
        mxfp4_scale_packed_cols: 2,
        fp8_output_hash: 11,
        mxfp4_output_hash: 22,
        fp8_mismatches: 0,
        mxfp4_mismatches: 0,
        fp8_max_abs_diff: 0.0,
        mxfp4_max_abs_diff: 0.0,
        device_arena_bytes: 128,
        pinned_host_bytes: 112,
        h2d_bytes: 28,
        d2h_bytes: 112,
        kernel_launches: 2,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"fp8_rows\":3"));
    assert!(json.contains("\"fp8_block_cols\":2"));
    assert!(json.contains("\"mxfp4_packed_cols\":4"));
    assert!(json.contains("\"fp8_mismatches\":0"));
    assert!(json.contains("\"mxfp4_mismatches\":0"));
    assert!(json.contains("\"kernel_launches\":2"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn deepseek_fused_inv_rope_fp8_quant_summary_serializes_outputs() {
    let summary = CudaDeepSeekFusedInvRopeFp8QuantSummary {
        status: SmokeStatus::Ok,
        return_code: 0,
        cuda_error: 0,
        num_tokens: 2,
        n_groups: 1,
        heads_per_group: 2,
        head_dim: 4,
        rope_dim: 2,
        quant_group_size: 2,
        scale_blocks: 4,
        fp8_output: vec![1, 2, 3],
        scale_output: vec![0.5, 1.0],
        packed_scale_output: vec![0x7f80],
        fp8_output_hash: 11,
        scale_output_hash: 22,
        packed_scale_output_hash: 33,
        device_arena_bytes: 64,
        pinned_host_bytes: 32,
        h2d_bytes: 48,
        d2h_bytes: 32,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"num_tokens\":2"));
    assert!(json.contains("\"heads_per_group\":2"));
    assert!(json.contains("\"fp8_output\":[1,2,3]"));
    assert!(json.contains("\"scale_output\":[0.5,1]"));
    assert!(json.contains("\"packed_scale_output\":[32640]"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn deepseek_fused_inv_rope_fp8_quant_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_fused_inv_rope_fp8_quant_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_fused_inv_rope_fp8_quant_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.num_tokens, 2);
    assert_eq!(second.n_groups, 1);
    assert_eq!(second.heads_per_group, 2);
    assert_eq!(second.head_dim, 4);
    assert_eq!(second.rope_dim, 2);
    assert_eq!(second.quant_group_size, 2);
    assert_eq!(second.fp8_output_hash, first.fp8_output_hash);
    assert_eq!(second.scale_output_hash, first.scale_output_hash);
    assert_eq!(
        second.packed_scale_output_hash,
        first.packed_scale_output_hash
    );
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
}

#[test]
fn deepseek_fused_inv_rope_fp8_quant_api_matches_vllm_math() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let input = inv_rope_fixture_input();
    let positions = [0i64, 1i64];
    let cos_sin_cache = [1.0, 0.0, 0.6, 0.8];
    let summary = deepseek_fused_inv_rope_fp8_quant(
        &input,
        &positions,
        &cos_sin_cache,
        2,
        1,
        2,
        4,
        2,
        2,
        2,
        448.0,
        1e-10,
    );
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected = reference_inv_rope_fp8_quant(&input, &positions, &cos_sin_cache);
    assert_eq!(summary.fp8_output, expected.0);
    assert_eq!(summary.packed_scale_output, expected.2);
    for (actual, expected) in summary.scale_output.iter().zip(expected.1.iter()) {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "scale actual={actual} expected={expected}"
        );
    }
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.fp8_output_hash != 0);
    assert!(summary.scale_output_hash != 0);
    assert!(summary.packed_scale_output_hash != 0);
}

#[test]
fn deepseek_quant_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_quant_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_quant_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.fp8_mismatches, 0);
    assert_eq!(second.mxfp4_mismatches, 0);
    assert_eq!(second.fp8_max_abs_diff, 0.0);
    assert_eq!(second.mxfp4_max_abs_diff, 0.0);
    assert_eq!(second.kernel_launches, 2);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
    assert_eq!(second.fp8_output_hash, first.fp8_output_hash);
    assert_eq!(second.mxfp4_output_hash, first.mxfp4_output_hash);
}

fn inv_rope_fixture_input() -> [f32; 16] {
    [
        1.0, -2.0, 3.0, -4.0, -0.5, 1.5, -2.5, 3.5, 0.25, -0.75, 1.25, -1.5, -2.0, 2.25, -2.5, 2.75,
    ]
}

fn reference_inv_rope_fp8_quant(
    input: &[f32],
    positions: &[i64],
    cos_sin_cache: &[f32],
) -> (Vec<u8>, Vec<f32>, Vec<u32>) {
    let num_tokens = 2usize;
    let heads_per_group = 2usize;
    let head_dim = 4usize;
    let rope_dim = 2usize;
    let quant_group_size = 2usize;
    let chunks_per_head = head_dim / quant_group_size;
    let scale_blocks = heads_per_group * chunks_per_head;
    let mut fp8 = vec![0u8; num_tokens * heads_per_group * head_dim];
    let mut scales = vec![0.0f32; num_tokens * scale_blocks];
    let mut packed = vec![0u32; num_tokens * heads_per_group];
    for token in 0..num_tokens {
        for head in 0..heads_per_group {
            for chunk in 0..chunks_per_head {
                let mut rotated = [0.0f32; 2];
                let mut absmax = 0.0f32;
                for (offset, value) in rotated.iter_mut().enumerate() {
                    let dim = chunk * quant_group_size + offset;
                    *value = rotated_value(
                        input,
                        cos_sin_cache,
                        positions[token],
                        token,
                        head,
                        dim,
                        heads_per_group,
                        head_dim,
                        rope_dim,
                        quant_group_size,
                    );
                    absmax = absmax.max(value.abs());
                }
                let scale = ((absmax.max(1e-10) / 448.0).log2().ceil()).exp2();
                let scale_idx = token * scale_blocks + head * chunks_per_head + chunk;
                scales[scale_idx] = scale;
                packed[token * heads_per_group + head] |=
                    ((scale.to_bits() >> 23) & 0xff) << (chunk * 8);
                for (offset, value) in rotated.iter().enumerate() {
                    let dim = chunk * quant_group_size + offset;
                    fp8[token * heads_per_group * head_dim + head * head_dim + dim] =
                        f32_to_f8_e4m3fn_bits_nearest((value / scale).clamp(-448.0, 448.0));
                }
            }
        }
    }
    (fp8, scales, packed)
}

#[allow(clippy::too_many_arguments)]
fn rotated_value(
    input: &[f32],
    cos_sin_cache: &[f32],
    position: i64,
    token: usize,
    head: usize,
    dim: usize,
    heads_per_group: usize,
    head_dim: usize,
    rope_dim: usize,
    quant_group_size: usize,
) -> f32 {
    let chunks_per_head = head_dim / quant_group_size;
    let nope_dim = head_dim - rope_dim;
    let rope_abs_start = (chunks_per_head - 1) * quant_group_size + (nope_dim % quant_group_size);
    let input_base = (token * heads_per_group + head) * head_dim;
    let value = input[input_base + dim];
    if dim < rope_abs_start {
        return value;
    }
    let rope_local = dim - rope_abs_start;
    let partner = input[input_base + (dim ^ 1)];
    let cache_base = position.max(0) as usize * rope_dim;
    let cos = cos_sin_cache[cache_base + (rope_local >> 1)];
    let sin = cos_sin_cache[cache_base + rope_dim / 2 + (rope_local >> 1)];
    if rope_local & 1 == 0 {
        value * cos + partner * sin
    } else {
        value * cos - partner * sin
    }
}

fn f32_to_f8_e4m3fn_bits_nearest(value: f32) -> u8 {
    let mut best_bits = 0u8;
    let mut best_diff = value.abs();
    for bits in 0..=0xfeu8 {
        let candidate = f8_e4m3fn_bits_to_f32(bits);
        if !candidate.is_finite() {
            continue;
        }
        let diff = (value - candidate).abs();
        if diff < best_diff {
            best_diff = diff;
            best_bits = bits;
        }
    }
    best_bits
}

fn f8_e4m3fn_bits_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (bits >> 3) & 0x0f;
    let frac = bits & 0x07;
    if exp == 0 {
        if frac == 0 {
            return sign * 0.0;
        }
        return sign * ((frac as f32) * 0.125) * 2f32.powi(-6);
    }
    if exp == 0x0f && frac == 0x07 {
        return f32::NAN;
    }
    sign * (1.0 + (frac as f32) * 0.125) * 2f32.powi(exp as i32 - 7)
}

#[test]
fn deepseek_quant_dequant_apis_match_reference_values() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let fp8_weights = [
        0x38, 0x40, 0x30, 0xb8, 0x70, 0x77, 0x78, 0x7e, 0x20, 0x28, 0x30, 0x38,
    ];
    let fp8_scales = [0x7f, 0x80, 0x7e, 0x81];
    let fp8 = deepseek_fp8_e4m3fn_e8m0_dequant(&fp8_weights, &fp8_scales, 3, 4, 2, 2);
    if fp8.status != SmokeStatus::Ok {
        return;
    }
    assert_eq!(fp8.rows, 3);
    assert_eq!(fp8.cols, 4);
    assert_eq!(fp8.block_rows, 2);
    assert_eq!(fp8.block_cols, 2);
    assert_eq!(
        fp8.output,
        [
            1.0, 2.0, 1.0, -2.0, 128.0, 240.0, 512.0, 896.0, 0.0625, 0.125, 2.0, 4.0
        ]
    );
    assert_eq!(fp8.kernel_launches, 1);
    assert_eq!(fp8.sync_calls, 1);
    assert_eq!(fp8.hot_path_allocations, 0);
    assert!(fp8.output_hash != 0);

    let mxfp4_packed = [0x21, 0x76, 0xa9, 0xfe, 0x10, 0x54, 0x98, 0xdc];
    let mxfp4_scales = [0x7f, 0x80, 0x7e, 0x81];
    let mxfp4 = deepseek_mxfp4_e2m1_e8m0_dequant(&mxfp4_packed, &mxfp4_scales, 2, 4, 2);
    assert_eq!(mxfp4.status, SmokeStatus::Ok, "mxfp4 dequant: {mxfp4:?}");
    assert_eq!(mxfp4.rows, 2);
    assert_eq!(mxfp4.cols, 8);
    assert_eq!(mxfp4.block_rows, 1);
    assert_eq!(mxfp4.block_cols, 4);
    assert_eq!(
        mxfp4.output,
        [
            0.5, 1.0, 4.0, 6.0, -1.0, -2.0, -8.0, -12.0, 0.0, 0.25, 1.0, 1.5, -0.0, -2.0, -8.0,
            -12.0,
        ]
    );
    assert_eq!(mxfp4.kernel_launches, 1);
    assert_eq!(mxfp4.sync_calls, 1);
    assert_eq!(mxfp4.hot_path_allocations, 0);
    assert!(mxfp4.output_hash != 0);
}

#[test]
fn deepseek_fp8_f32_scale_matvec_matches_reference_values() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let weights = [
        0x38, 0x40, 0x30, 0xb8, 0x70, 0x77, 0x78, 0x7e, 0x20, 0x28, 0x30, 0x38,
    ];
    let scales = [1.0, 2.0, 0.5, 4.0];
    let input = [0.5, -1.0, 2.0, 0.25];

    let summary = deepseek_fp8_e4m3fn_f32_scale_matvec(&weights, &scales, &input, 3, 4, 2, 2);
    if summary.status != SmokeStatus::Ok {
        return;
    }

    assert_eq!(summary.rows, 3);
    assert_eq!(summary.cols, 4);
    assert_eq!(summary.block_rows, 2);
    assert_eq!(summary.block_cols, 2);
    assert_eq!(summary.output, [0.0, 1072.0, 4.90625]);
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.output_hash != 0);
}

#[test]
fn deepseek_fp8_f32_scale_encoded_matvec_matches_reference_values() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let weights = [
        0x38, 0x40, 0x30, 0xb8, 0x70, 0x77, 0x78, 0x7e, 0x20, 0x28, 0x30, 0x38,
    ];
    let scales = [1.0, 2.0, 0.5, 4.0];
    let input = [0.5, -1.0, 2.0, 0.25].map(f32_to_bf16_bits);
    const BF16: u32 = 1;

    let summary =
        deepseek_fp8_e4m3fn_f32_scale_encoded_matvec(&weights, &scales, &input, BF16, 3, 4, 2, 2);
    if summary.status != SmokeStatus::Ok {
        return;
    }

    assert_eq!(summary.rows, 3);
    assert_eq!(summary.cols, 4);
    assert_eq!(summary.block_rows, 2);
    assert_eq!(summary.block_cols, 2);
    assert_eq!(summary.output, [0.0, 1072.0, 4.90625]);
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.output_hash != 0);
}

#[test]
fn deepseek_fp8_f32_scale_encoded_gemm_tokens_matches_reference_values() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let weights = [
        0x38, 0x40, 0x30, 0xb8, 0x70, 0x77, 0x78, 0x7e, 0x20, 0x28, 0x30, 0x38,
    ];
    let scales = [1.0, 2.0, 0.5, 4.0];
    let input_f32 = (0..40)
        .map(|idx| ((idx % 13) as f32 - 6.0) * 0.25)
        .collect::<Vec<_>>();
    let input = input_f32
        .iter()
        .copied()
        .map(f32_to_bf16_bits)
        .collect::<Vec<_>>();
    const BF16: u32 = 1;

    let summary = deepseek_fp8_e4m3fn_f32_scale_encoded_gemm_tokens(
        &weights, &scales, &input, BF16, 3, 4, 10, 2, 2,
    );
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected =
        reference_fp8_f32_scale_encoded_gemm_tokens(&weights, &scales, &input, 3, 4, 10, 2, 2);
    assert_eq!(summary.rows, 3);
    assert_eq!(summary.cols, 4);
    assert_eq!(summary.tokens, 10);
    assert_eq!(summary.block_rows, 2);
    assert_eq!(summary.block_cols, 2);
    for (idx, (actual, expected)) in summary.output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "output[{idx}] actual={actual} expected={expected}"
        );
    }
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.output_hash != 0);
}

#[test]
fn deepseek_fp8_e8m0_scale_encoded_gemm_tokens_matches_reference_values() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let weights = [
        0x38, 0x40, 0x30, 0xb8, 0x70, 0x77, 0x78, 0x7e, 0x20, 0x28, 0x30, 0x38,
    ];
    let scales = [0x7f, 0x80, 0x7e, 0x81];
    let input_f32 = (0..40)
        .map(|idx| ((idx % 13) as f32 - 6.0) * 0.25)
        .collect::<Vec<_>>();
    let input = input_f32
        .iter()
        .copied()
        .map(f32_to_bf16_bits)
        .collect::<Vec<_>>();
    const BF16: u32 = 1;

    let summary = deepseek_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens(
        &weights, &scales, &input, BF16, 3, 4, 10, 2, 2,
    );
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected =
        reference_fp8_e8m0_scale_encoded_gemm_tokens(&weights, &scales, &input, 3, 4, 10, 2, 2);
    assert_eq!(summary.rows, 3);
    assert_eq!(summary.cols, 4);
    assert_eq!(summary.tokens, 10);
    assert_eq!(summary.block_rows, 2);
    assert_eq!(summary.block_cols, 2);
    for (idx, (actual, expected)) in summary.output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "output[{idx}] actual={actual} expected={expected}"
        );
    }
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.output_hash != 0);
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn reference_fp8_f32_scale_encoded_gemm_tokens(
    weights: &[u8],
    scales: &[f32],
    input: &[u16],
    rows: usize,
    cols: usize,
    tokens: usize,
    block_rows: usize,
    block_cols: usize,
) -> Vec<f32> {
    let scale_cols = cols.div_ceil(block_cols);
    let mut output = vec![0.0f32; rows * tokens];
    for token in 0..tokens {
        for row in 0..rows {
            let mut sum = 0.0f32;
            for col in 0..cols {
                let scale_idx = (row / block_rows) * scale_cols + (col / block_cols);
                let weight = f8_e4m3fn_bits_to_f32(weights[row * cols + col]) * scales[scale_idx];
                sum += weight * bf16_bits_to_f32(input[token * cols + col]);
            }
            output[token * rows + row] = sum;
        }
    }
    output
}

fn reference_fp8_e8m0_scale_encoded_gemm_tokens(
    weights: &[u8],
    scales: &[u8],
    input: &[u16],
    rows: usize,
    cols: usize,
    tokens: usize,
    block_rows: usize,
    block_cols: usize,
) -> Vec<f32> {
    let scale_cols = cols.div_ceil(block_cols);
    let mut output = vec![0.0f32; rows * tokens];
    for token in 0..tokens {
        for row in 0..rows {
            let mut sum = 0.0f32;
            for col in 0..cols {
                let scale_idx = (row / block_rows) * scale_cols + (col / block_cols);
                let weight = f8_e4m3fn_bits_to_f32(weights[row * cols + col])
                    * e8m0_exponent_bits_to_f32(scales[scale_idx]);
                sum += weight * bf16_bits_to_f32(input[token * cols + col]);
            }
            output[token * rows + row] = sum;
        }
    }
    output
}

fn e8m0_exponent_bits_to_f32(bits: u8) -> f32 {
    if bits == 0 {
        0.0
    } else {
        2f32.powi(bits as i32 - 127)
    }
}
