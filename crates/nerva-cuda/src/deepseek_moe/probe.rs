use crate::deepseek_moe::ffi::{NervaCudaDeepSeekMoeSmokeResult, run_deepseek_moe_smoke};
use crate::deepseek_moe::summary::CudaDeepSeekMoeSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

pub fn deepseek_moe_smoke() -> CudaDeepSeekMoeSummary {
    let mut out = NervaCudaDeepSeekMoeSmokeResult::default();
    let return_code = run_deepseek_moe_smoke(&mut out);

    if return_code == 0
        && out.status == 0
        && out.hidden_size == 3
        && out.intermediate_size == 2
        && out.num_experts == 2
        && out.top_k == 2
        && (out.swiglu_limit - 1.0).abs() <= f32::EPSILON
        && out.expert_ids == [1, 0]
        && out.expert_weights.iter().all(|value| value.is_finite())
        && out.output.iter().all(|value| value.is_finite())
        && out.output_hash != 0
        && out.mismatches == 0
        && out.max_abs_diff <= 1e-6
        && out.d2h_bytes > 0
        && out.kernel_launches == 1
        && out.sync_calls == 1
        && out.hot_path_allocations == 0
    {
        return CudaDeepSeekMoeSummary {
            status: SmokeStatus::Ok,
            hidden_size: out.hidden_size,
            intermediate_size: out.intermediate_size,
            num_experts: out.num_experts,
            top_k: out.top_k,
            swiglu_limit: out.swiglu_limit,
            expert_ids: out.expert_ids,
            expert_weights: out.expert_weights,
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
        "CUDA DeepSeek MoE smoke failed: return_code={} status={} cuda_error={} device_count={} expert_ids={:?} mismatches={} max_abs_diff={} kernel_launches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.expert_ids,
        out.mismatches,
        out.max_abs_diff,
        out.kernel_launches,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaDeepSeekMoeSummary::unavailable(reason)
    } else {
        CudaDeepSeekMoeSummary::failed(reason)
    }
}
