use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::transport::matrix::types::TransportCapabilityMatrixStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_transport_matrix(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_transport_capability_matrix_probe() {
        Ok(summary) => report.push(
            "transport_capability_matrix",
            matches!(summary.status, TransportCapabilityMatrixStatus::Ok)
                && summary.sizes == 6
                && summary.entries.len() == 24
                && summary.degraded_to_pinned_host_entries > 0
                && summary.pageable_copies == 0
                && summary.per_token_registrations == 0
                && summary.registration_cache_hits == summary.entries.len() as u64
                && summary.estimated_cpu_core_ns > 0
                && summary.pcie_tx_bytes > 0
                && summary.pcie_rx_bytes > 0
                && summary.credit_stall_ns == 0
                && summary.hot_path_allocations == 0,
            format!(
                "sizes={} entries={} host_staged={} gpu_direct={} degraded_to_pinned_host={} p95_estimated_visible_ns={} cpu_core_ns={} pcie_tx_bytes={} pcie_rx_bytes={} registration_cache_hits={} pageable_copies={} per_token_registrations={} credit_stall_ns={} hot_path_allocations={}",
                summary.sizes,
                summary.entries.len(),
                summary.host_staged_entries,
                summary.gpu_direct_entries,
                summary.degraded_to_pinned_host_entries,
                summary.p95_estimated_visible_ns,
                summary.estimated_cpu_core_ns,
                summary.pcie_tx_bytes,
                summary.pcie_rx_bytes,
                summary.registration_cache_hits,
                summary.pageable_copies,
                summary.per_token_registrations,
                summary.credit_stall_ns,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("transport_capability_matrix", false, format!("{err:?}")),
    }
}
