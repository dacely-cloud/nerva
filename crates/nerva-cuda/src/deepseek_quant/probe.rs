use crate::deepseek_quant::ffi::{NervaCudaDeepSeekQuantSmokeResult, run_deepseek_quant_smoke};
use crate::deepseek_quant::fp8::f32_to_f8_e4m3fn_bits;
use crate::deepseek_quant::inv_rope::{
    CudaDeepSeekFusedInvRopeFp8QuantSummary, deepseek_fused_inv_rope_fp8_quant,
};
use crate::deepseek_quant::summary::CudaDeepSeekQuantSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

pub fn deepseek_quant_smoke() -> CudaDeepSeekQuantSummary {
    let mut out = NervaCudaDeepSeekQuantSmokeResult::default();
    let return_code = run_deepseek_quant_smoke(&mut out);

    if return_code == 0
        && out.status == 0
        && out.fp8_rows == 3
        && out.fp8_cols == 4
        && out.fp8_block_rows == 2
        && out.fp8_block_cols == 2
        && out.mxfp4_rows == 2
        && out.mxfp4_packed_cols == 4
        && out.mxfp4_scale_packed_cols == 2
        && out.fp8_output_hash != 0
        && out.mxfp4_output_hash != 0
        && out.fp8_mismatches == 0
        && out.mxfp4_mismatches == 0
        && out.fp8_max_abs_diff == 0.0
        && out.mxfp4_max_abs_diff == 0.0
        && out.h2d_bytes > 0
        && out.d2h_bytes > 0
        && out.kernel_launches == 2
        && out.sync_calls == 1
        && out.hot_path_allocations == 0
    {
        return CudaDeepSeekQuantSummary {
            status: SmokeStatus::Ok,
            fp8_rows: out.fp8_rows,
            fp8_cols: out.fp8_cols,
            fp8_block_rows: out.fp8_block_rows,
            fp8_block_cols: out.fp8_block_cols,
            mxfp4_rows: out.mxfp4_rows,
            mxfp4_packed_cols: out.mxfp4_packed_cols,
            mxfp4_scale_packed_cols: out.mxfp4_scale_packed_cols,
            fp8_output_hash: out.fp8_output_hash,
            mxfp4_output_hash: out.mxfp4_output_hash,
            fp8_mismatches: out.fp8_mismatches,
            mxfp4_mismatches: out.mxfp4_mismatches,
            fp8_max_abs_diff: out.fp8_max_abs_diff,
            mxfp4_max_abs_diff: out.mxfp4_max_abs_diff,
            device_arena_bytes: out.device_arena_bytes,
            pinned_host_bytes: out.pinned_host_bytes,
            h2d_bytes: out.h2d_bytes,
            d2h_bytes: out.d2h_bytes,
            kernel_launches: out.kernel_launches,
            sync_calls: out.sync_calls,
            hot_path_allocations: out.hot_path_allocations,
            error: None,
        };
    }

    let reason = format!(
        "CUDA DeepSeek quant smoke failed: return_code={} status={} cuda_error={} device_count={} fp8_hash={} mxfp4_hash={} fp8_mismatches={} mxfp4_mismatches={} fp8_max_abs_diff={} mxfp4_max_abs_diff={} kernel_launches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.fp8_output_hash,
        out.mxfp4_output_hash,
        out.fp8_mismatches,
        out.mxfp4_mismatches,
        out.fp8_max_abs_diff,
        out.mxfp4_max_abs_diff,
        out.kernel_launches,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaDeepSeekQuantSummary::unavailable(reason)
    } else {
        CudaDeepSeekQuantSummary::failed(reason)
    }
}

pub fn deepseek_fused_inv_rope_fp8_quant_smoke() -> CudaDeepSeekFusedInvRopeFp8QuantSummary {
    let input = inv_rope_fixture_input();
    let positions = [0i64, 1i64];
    let cos_sin_cache = [
        1.0, 0.0, // position 0: cos, sin
        0.6, 0.8, // position 1
    ];
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
        return summary;
    }

    let expected = reference_inv_rope_fp8_quant(&input, &positions, &cos_sin_cache);
    if summary.fp8_output == expected.0
        && scale_slices_close(&summary.scale_output, &expected.1, 1e-6)
        && summary.packed_scale_output == expected.2
        && summary.fp8_output_hash != 0
        && summary.scale_output_hash != 0
        && summary.packed_scale_output_hash != 0
        && summary.kernel_launches == 1
        && summary.sync_calls == 1
        && summary.hot_path_allocations == 0
    {
        return summary;
    }

    let mut failed = summary;
    failed.status = SmokeStatus::Failed;
    failed.error = Some(format!(
        "CUDA DeepSeek fused inverse RoPE FP8 quant smoke mismatch: fp8_match={} scale_match={} packed_match={}",
        failed.fp8_output == expected.0,
        scale_slices_close(&failed.scale_output, &expected.1, 1e-6),
        failed.packed_scale_output == expected.2
    ));
    failed
}

fn inv_rope_fixture_input() -> [f32; 16] {
    [
        1.0, -2.0, 3.0, -4.0, // token 0, head 0
        -0.5, 1.5, -2.5, 3.5, // token 0, head 1
        0.25, -0.75, 1.25, -1.5, // token 1, head 0
        -2.0, 2.25, -2.5, 2.75, // token 1, head 1
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
                let mut rotated = vec![0.0f32; quant_group_size];
                let mut absmax = 0.0f32;
                for offset in 0..quant_group_size {
                    let dim = chunk * quant_group_size + offset;
                    let value = rotated_value(
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
                    rotated[offset] = value;
                    absmax = absmax.max(value.abs());
                }
                let scale = ((absmax.max(1e-10) / 448.0).log2().ceil()).exp2();
                let scale_idx = token * scale_blocks + head * chunks_per_head + chunk;
                scales[scale_idx] = scale;
                let scale_byte = (scale.to_bits() >> 23) & 0xff;
                packed[token * heads_per_group + head] |= scale_byte << (chunk * 8);
                for (offset, value) in rotated.iter().enumerate() {
                    let dim = chunk * quant_group_size + offset;
                    let quantized = (value / scale).clamp(-448.0, 448.0);
                    fp8[token * heads_per_group * head_dim + head * head_dim + dim] =
                        f32_to_f8_e4m3fn_bits(quantized);
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
    let cs_idx = rope_local >> 1;
    let cache_base = position.max(0) as usize * rope_dim;
    let cos = cos_sin_cache[cache_base + cs_idx];
    let sin = cos_sin_cache[cache_base + rope_dim / 2 + cs_idx];
    if rope_local & 1 == 0 {
        value * cos + partner * sin
    } else {
        value * cos - partner * sin
    }
}

fn scale_slices_close(actual: &[f32], expected: &[f32], tolerance: f32) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| (actual - expected).abs() <= tolerance)
}
