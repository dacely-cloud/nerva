use nerva_runtime::backend::contract::status::BackendContractProbeStatus;
use nerva_runtime::engine::runtime::Runtime;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_backend_contract(report: &mut AcceptanceReport, runtime: &Runtime) {
    let summary = runtime.run_backend_contract_probe();
    report.push(
        "runtime_backend_contract",
        summary.status == BackendContractProbeStatus::Ok
            && summary.passed()
            && summary.supports_device_allocations
            && summary.supports_pinned_host_allocations
            && summary.supports_streams
            && summary.supports_events
            && summary.supports_graph_capture
            && summary.supports_async_copies
            && summary.supports_device_sampling
            && summary.validation.bootstrap_decode_ready
            && summary.validation.device_allocation_ready
            && summary.validation.pinned_allocation_ready
            && summary.validation.queue_ready
            && summary.validation.graph_ready
            && summary.hot_path_allocations == 0,
        format!(
            "status={:?} backend={} device={} dtypes={} device_bytes={}/{} pinned_bytes={}/{} graph_replays={} graph_nodes={} sampler_tokens={} queue_ready={} event_ready={} graph_ready={} submission_id={} bootstrap_decode_ready={} hot_path_allocations={} error={}",
            summary.status,
            summary.backend.as_str(),
            summary.device_ordinal,
            summary.exact_dtypes.len(),
            summary.allocated_device_bytes,
            summary.requested_device_bytes,
            summary.allocated_pinned_bytes,
            summary.requested_pinned_bytes,
            summary.graph_replays,
            summary.graph_nodes,
            summary.sampler_tokens,
            summary.queue_ready,
            summary.event_ready,
            summary.graph_ready,
            summary
                .submission_id
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            summary.validation.bootstrap_decode_ready,
            summary.hot_path_allocations,
            summary.error.as_deref().unwrap_or("none"),
        ),
    );
}
