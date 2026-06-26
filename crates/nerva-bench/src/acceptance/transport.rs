use nerva_runtime::engine::kv_probe::{KvResidencyProbeConfig, KvResidencyProbeStatus};
use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::transport::matrix::TransportCapabilityMatrixStatus;
use nerva_runtime::transport::probe::TransportPathProbeStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_kv_residency(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_kv_residency_probe(KvResidencyProbeConfig::default()) {
        Ok(summary) => report.push(
            "kv_residency_tiering",
            matches!(summary.status, KvResidencyProbeStatus::Ok)
                && summary.decisions > 0
                && summary.prefetches > 0
                && summary.demotions > 0
                && summary.evictions > 0
                && summary.stall_events > 0
                && summary.hot_path_allocations == 0,
            format!(
                "pages={} decisions={} prefetches={} demotions={} evictions={} stall_events={} hot_path_allocations={}",
                summary.pages,
                summary.decisions,
                summary.prefetches,
                summary.demotions,
                summary.evictions,
                summary.stall_events,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("kv_residency_tiering", false, format!("{err:?}")),
    }
}

pub(crate) fn push_transport_path(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_transport_path_probe() {
        Ok(summary) => report.push(
            "transport_pinned_fallback",
            matches!(summary.status, TransportPathProbeStatus::Ok)
                && summary.requests > 0
                && summary.pinned_host_paths > 0
                && summary.fallback_decisions > 0
                && summary.phase_handoff_syncs > 0
                && summary.pageable_copies == 0
                && summary.per_token_registrations == 0
                && summary.hot_path_allocations == 0,
            format!(
                "requests={} pinned_host_paths={} fallback_decisions={} phase_handoff_syncs={} pageable_copies={} per_token_registrations={} hot_path_allocations={}",
                summary.requests,
                summary.pinned_host_paths,
                summary.fallback_decisions,
                summary.phase_handoff_syncs,
                summary.pageable_copies,
                summary.per_token_registrations,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("transport_pinned_fallback", false, format!("{err:?}")),
    }
}

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
