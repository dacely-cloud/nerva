use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::transport::probe::TransportPathProbeStatus;

use crate::acceptance::report::AcceptanceReport;

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
