use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_backend_contract(report: &mut AcceptanceReport) {
    let summary = nerva_runtime::engine::cuda::cuda_backend_contract_smoke(4096, 4096);
    report.push(
        "cuda_backend_contract",
        summary.passed(),
        format!(
            "status={:?} gpu={} cc={}.{} device_bytes={}/{} pinned_bytes={}/{} streams={}/{} events={}/{} device_allocs={}/{} pinned_allocs={}/{} memset_bytes={} D2H_bytes={} sync_calls={} observed_word={} hot_path_allocations={} error={}",
            summary.status,
            summary.gpu_name.as_deref().unwrap_or("none"),
            summary
                .compute_capability_major
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            summary
                .compute_capability_minor
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            summary.allocated_device_bytes,
            summary.requested_device_bytes,
            summary.allocated_pinned_bytes,
            summary.requested_pinned_bytes,
            summary.stream_creations,
            summary.stream_destroys,
            summary.event_creations,
            summary.event_destroys,
            summary.device_allocations,
            summary.device_frees,
            summary.pinned_allocations,
            summary.pinned_frees,
            summary.memset_bytes,
            summary.d2h_bytes,
            summary.sync_calls,
            summary
                .observed_word
                .map_or_else(|| "none".to_string(), |value| format!("0x{value:08x}")),
            summary.hot_path_allocations,
            summary.error.as_deref().unwrap_or("none"),
        ),
    );
}
