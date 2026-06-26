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
                && summary.registration_cache_hit_rate_per_mille == 1_000
                && summary.estimated_nic_utilization_per_mille == 1_000
                && summary.host_event_wait_ns == summary.visible_non_overlapped_ns
                && summary.host_event_wait_ns > 0
                && summary.gpu_idle_ns == 0
                && summary.max_queue_depth >= 4
                && summary.estimated_cpu_core_ns > 0
                && summary.pcie_tx_bytes > 0
                && summary.pcie_rx_bytes > 0
                && summary.false_gpu_direct_claims == 0
                && summary.gpu_export_without_nic_direct_entries
                    <= summary.gpu_memory_export_verified_entries
                && summary.credit_stall_ns == 0
                && summary.hot_path_allocations == 0,
            format!(
                "sizes={} entries={} host_staged={} gpu_direct={} degraded_to_pinned_host={} gpu_memory_export_verified={} cuda_vmm_posix_fd_export_verified={} gpu_direct_rdma_verified={} gpu_export_without_nic_direct={} false_gpu_direct_claims={} p95_estimated_visible_ns={} visible_non_overlapped_ns={} host_event_wait_ns={} gpu_idle_ns={} cpu_core_ns={} pcie_tx_bytes={} pcie_rx_bytes={} registration_cache_hits={} registration_cache_hit_rate_per_mille={} estimated_nic_utilization_per_mille={} max_queue_depth={} pageable_copies={} per_token_registrations={} credit_stall_ns={} hot_path_allocations={}",
                summary.sizes,
                summary.entries.len(),
                summary.host_staged_entries,
                summary.gpu_direct_entries,
                summary.degraded_to_pinned_host_entries,
                summary.gpu_memory_export_verified_entries,
                summary.cuda_vmm_posix_fd_export_verified_entries,
                summary.gpu_direct_rdma_verified_entries,
                summary.gpu_export_without_nic_direct_entries,
                summary.false_gpu_direct_claims,
                summary.p95_estimated_visible_ns,
                summary.visible_non_overlapped_ns,
                summary.host_event_wait_ns,
                summary.gpu_idle_ns,
                summary.estimated_cpu_core_ns,
                summary.pcie_tx_bytes,
                summary.pcie_rx_bytes,
                summary.registration_cache_hits,
                summary.registration_cache_hit_rate_per_mille,
                summary.estimated_nic_utilization_per_mille,
                summary.max_queue_depth,
                summary.pageable_copies,
                summary.per_token_registrations,
                summary.credit_stall_ns,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("transport_capability_matrix", false, format!("{err:?}")),
    }
}
