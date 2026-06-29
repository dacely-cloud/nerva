use crate::sampler::ffi::{
    run_greedy_sampler_smoke, NervaCudaGreedySamplerResult, CUDA_ERROR_NO_DEVICE,
};
use crate::sampler::summary::CudaGreedySamplerSummary;
use crate::smoke::status::SmokeStatus;

pub fn greedy_sampler_smoke() -> CudaGreedySamplerSummary {
    let mut out = NervaCudaGreedySamplerResult::default();
    let return_code = run_greedy_sampler_smoke(&mut out);

    if return_code == 0
        && out.status == 0
        && out.vocab_size == 4
        && out.token == 2
        && out.slot_version == 1
        && out.completion == 1
        && out.kernel_launches == 1
        && out.hot_path_allocations == 0
    {
        return CudaGreedySamplerSummary {
            status: SmokeStatus::Ok,
            vocab_size: out.vocab_size,
            token_index: out.token_index,
            token: out.token,
            slot_version: out.slot_version,
            completion: out.completion,
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
        "CUDA greedy sampler smoke failed: return_code={} status={} cuda_error={} device_count={} vocab_size={} token={} slot_version={} completion={} kernel_launches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.vocab_size,
        out.token,
        out.slot_version,
        out.completion,
        out.kernel_launches,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaGreedySamplerSummary::unavailable(reason)
    } else {
        CudaGreedySamplerSummary::failed(reason)
    }
}
