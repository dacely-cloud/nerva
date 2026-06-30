use crate::deepseek_router::ffi::{NervaCudaDeepSeekRouterSmokeResult, run_deepseek_router_smoke};
use crate::deepseek_router::summary::CudaDeepSeekRouterSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

pub fn deepseek_router_smoke() -> CudaDeepSeekRouterSummary {
    let mut out = NervaCudaDeepSeekRouterSmokeResult::default();
    let return_code = run_deepseek_router_smoke(&mut out);

    if return_code == 0
        && out.status == 0
        && out.v3_num_experts == 8
        && out.v3_num_groups == 2
        && out.v3_top_k_groups == 1
        && out.v3_top_k == 2
        && out.v4_num_experts == 4
        && out.v4_top_k == 2
        && out.v4_hash_top_k == 3
        && out.v3_expert_ids == [3, 2]
        && out.v4_expert_ids == [1, 2]
        && out.v4_hash_expert_ids == [2, 1, 3]
        && out.v3_mismatches == 0
        && out.v4_mismatches == 0
        && out.v4_hash_mismatches == 0
        && out.v3_max_abs_diff <= 1e-6
        && out.v4_max_abs_diff <= 1e-6
        && out.v4_hash_max_abs_diff <= 1e-6
        && out.v3_output_hash != 0
        && out.v4_output_hash != 0
        && out.v4_hash_output_hash != 0
        && out.v3_weights.iter().all(|value| value.is_finite())
        && out.v4_weights.iter().all(|value| value.is_finite())
        && out.v4_hash_weights.iter().all(|value| value.is_finite())
        && out.d2h_bytes > 0
        && out.kernel_launches == 1
        && out.sync_calls == 1
        && out.hot_path_allocations == 0
    {
        return CudaDeepSeekRouterSummary {
            status: SmokeStatus::Ok,
            v3_num_experts: out.v3_num_experts,
            v3_num_groups: out.v3_num_groups,
            v3_top_k_groups: out.v3_top_k_groups,
            v3_top_k: out.v3_top_k,
            v4_num_experts: out.v4_num_experts,
            v4_top_k: out.v4_top_k,
            v4_hash_top_k: out.v4_hash_top_k,
            v3_expert_ids: out.v3_expert_ids,
            v4_expert_ids: out.v4_expert_ids,
            v4_hash_expert_ids: out.v4_hash_expert_ids,
            v3_weights: out.v3_weights,
            v4_weights: out.v4_weights,
            v4_hash_weights: out.v4_hash_weights,
            v3_output_hash: out.v3_output_hash,
            v4_output_hash: out.v4_output_hash,
            v4_hash_output_hash: out.v4_hash_output_hash,
            v3_mismatches: out.v3_mismatches,
            v4_mismatches: out.v4_mismatches,
            v4_hash_mismatches: out.v4_hash_mismatches,
            v3_max_abs_diff: out.v3_max_abs_diff,
            v4_max_abs_diff: out.v4_max_abs_diff,
            v4_hash_max_abs_diff: out.v4_hash_max_abs_diff,
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
        "CUDA DeepSeek router smoke failed: return_code={} status={} cuda_error={} device_count={} v3_ids={:?} v4_ids={:?} v4_hash_ids={:?} v3_mismatches={} v4_mismatches={} v4_hash_mismatches={} kernel_launches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.v3_expert_ids,
        out.v4_expert_ids,
        out.v4_hash_expert_ids,
        out.v3_mismatches,
        out.v4_mismatches,
        out.v4_hash_mismatches,
        out.kernel_launches,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaDeepSeekRouterSummary::unavailable(reason)
    } else {
        CudaDeepSeekRouterSummary::failed(reason)
    }
}
