use crate::deepseek_mla::ffi::{NervaCudaDeepSeekMlaSmokeResult, run_deepseek_mla_smoke};
use crate::deepseek_mla::qkv_norm::{CudaDeepSeekQKvRmsNormSummary, deepseek_qkv_rmsnorm};
use crate::deepseek_mla::summary::CudaDeepSeekMlaSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

pub fn deepseek_mla_smoke() -> CudaDeepSeekMlaSummary {
    let mut out = NervaCudaDeepSeekMlaSmokeResult::default();
    let return_code = run_deepseek_mla_smoke(&mut out);

    if return_code == 0
        && out.status == 0
        && out.heads == 2
        && out.tokens == 3
        && out.kv_lora_rank == 3
        && out.qk_nope_head_dim == 2
        && out.qk_rope_head_dim == 1
        && out.v_head_dim == 2
        && (out.softmax_scale - 0.7).abs() < 1e-6
        && out.output_hash != 0
        && out.mismatches == 0
        && out.max_abs_diff <= 1e-6
        && out.output.iter().all(|value| value.is_finite())
        && out.d2h_bytes > 0
        && out.kernel_launches == 1
        && out.sync_calls == 1
        && out.hot_path_allocations == 0
    {
        return CudaDeepSeekMlaSummary {
            status: SmokeStatus::Ok,
            heads: out.heads,
            tokens: out.tokens,
            kv_lora_rank: out.kv_lora_rank,
            qk_nope_head_dim: out.qk_nope_head_dim,
            qk_rope_head_dim: out.qk_rope_head_dim,
            v_head_dim: out.v_head_dim,
            softmax_scale: out.softmax_scale,
            output: out.output,
            output_hash: out.output_hash,
            mismatches: out.mismatches,
            max_abs_diff: out.max_abs_diff,
            device_arena_bytes: out.device_arena_bytes,
            pinned_host_bytes: out.pinned_host_bytes,
            d2h_bytes: out.d2h_bytes,
            kernel_launches: out.kernel_launches,
            sync_calls: out.sync_calls,
            hot_path_allocations: out.hot_path_allocations,
            error: None,
        };
    }

    let reason = format!(
        "CUDA DeepSeek MLA smoke failed: return_code={} status={} cuda_error={} device_count={} heads={} tokens={} output_hash={} mismatches={} max_abs_diff={} kernel_launches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.heads,
        out.tokens,
        out.output_hash,
        out.mismatches,
        out.max_abs_diff,
        out.kernel_launches,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaDeepSeekMlaSummary::unavailable(reason)
    } else {
        CudaDeepSeekMlaSummary::failed(reason)
    }
}

pub fn deepseek_qkv_rmsnorm_smoke() -> CudaDeepSeekQKvRmsNormSummary {
    let q = [
        1.0, -2.0, 3.0, -4.0, // token 0
        -0.5, 1.5, -2.5, 3.5, // token 1
    ];
    let kv = [
        0.25, -0.75, 1.25, // token 0
        -1.5, 2.0, -2.5, // token 1
    ];
    let q_weight = [0.5, 1.0, -1.5, 2.0];
    let kv_weight = [1.25, -0.5, 0.75];
    let summary = deepseek_qkv_rmsnorm(&q, &kv, &q_weight, &kv_weight, 2, 4, 3, 1e-5);
    if summary.status != SmokeStatus::Ok {
        return summary;
    }

    let expected_q = reference_rmsnorm_rows(&q, &q_weight, 2, 4, 1e-5);
    let expected_kv = reference_rmsnorm_rows(&kv, &kv_weight, 2, 3, 1e-5);
    let matches_q = summary
        .q_out
        .iter()
        .zip(expected_q.iter())
        .all(|(actual, expected)| (actual - expected).abs() <= 1e-5);
    let matches_kv = summary
        .kv_out
        .iter()
        .zip(expected_kv.iter())
        .all(|(actual, expected)| (actual - expected).abs() <= 1e-5);
    if matches_q
        && matches_kv
        && summary.output_hash != 0
        && summary.kernel_launches == 1
        && summary.sync_calls == 1
        && summary.hot_path_allocations == 0
    {
        return summary;
    }

    let mut failed = summary;
    failed.status = SmokeStatus::Failed;
    failed.error = Some(format!(
        "CUDA DeepSeek Q/KV RMSNorm smoke mismatch: matches_q={} matches_kv={} output_hash={} kernel_launches={}",
        matches_q, matches_kv, failed.output_hash, failed.kernel_launches
    ));
    failed
}

fn reference_rmsnorm_rows(
    values: &[f32],
    weight: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let row_values = &values[row * cols..(row + 1) * cols];
        let variance = row_values.iter().map(|value| value * value).sum::<f32>() / cols as f32;
        let rrms = 1.0 / (variance + eps).sqrt();
        for col in 0..cols {
            out[row * cols + col] = row_values[col] * rrms * weight[col];
        }
    }
    out
}
