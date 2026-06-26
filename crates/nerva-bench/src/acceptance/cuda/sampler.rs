use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_device_sampler(report: &mut AcceptanceReport) {
    let cuda_sampler = nerva_cuda::sampler::probe::greedy_sampler_smoke();
    let cuda_sampler_passed = format!("{:?}", cuda_sampler.status) == "Ok"
        && cuda_sampler.vocab_size == 4
        && cuda_sampler.token_index == 0
        && cuda_sampler.token == 2
        && cuda_sampler.slot_version == 1
        && cuda_sampler.completion == 1
        && cuda_sampler.h2d_bytes == 16
        && cuda_sampler.d2h_bytes > 0
        && cuda_sampler.device_arena_bytes > 0
        && cuda_sampler.pinned_host_bytes > 0
        && cuda_sampler.kernel_launches == 1
        && cuda_sampler.sync_calls == 2
        && cuda_sampler.hot_path_allocations == 0;
    report.push(
        "cuda_device_sampler",
        cuda_sampler_passed,
        format!(
            "status={:?} vocab_size={} token_index={} token={} slot_version={} completion={} H2D_bytes={} D2H_bytes={} device_arena_bytes={} pinned_host_bytes={} kernel_launches={} sync_calls={} hot_path_allocations={} error={}",
            cuda_sampler.status,
            cuda_sampler.vocab_size,
            cuda_sampler.token_index,
            cuda_sampler.token,
            cuda_sampler.slot_version,
            cuda_sampler.completion,
            cuda_sampler.h2d_bytes,
            cuda_sampler.d2h_bytes,
            cuda_sampler.device_arena_bytes,
            cuda_sampler.pinned_host_bytes,
            cuda_sampler.kernel_launches,
            cuda_sampler.sync_calls,
            cuda_sampler.hot_path_allocations,
            cuda_sampler.error.as_deref().unwrap_or("none"),
        ),
    );
}
