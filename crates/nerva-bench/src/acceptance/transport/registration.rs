use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::transport::registration::summary::TransportRegistrationStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_transport_registration(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_transport_registration_probe() {
        Ok(summary) => report.push(
            "transport_registration_cache",
            matches!(summary.status, TransportRegistrationStatus::Ok)
                && summary.passed()
                && summary.bootstrap_registrations == summary.registered_entries
                && summary.cache_hits > 0
                && summary.cache_misses > 0
                && summary.stale_address_rejections > 0
                && summary.hot_path_registration_attempts
                    == summary.hot_path_registration_rejections
                && summary.per_token_registrations == 0
                && summary.hot_path_allocations == 0,
            format!(
                "capacity={} registered_entries={} bootstrap_registrations={} cache_hits={} cache_misses={} stale_address_rejections={} stale_version_rejections={} hot_path_registration_attempts={} hot_path_registration_rejections={} per_token_registrations={} pinned_host_registrations={} gpu_direct_registrations={} transport_events={} phase_handoff_syncs={} registration_cache_hit_rate_per_mille={} hot_path_allocations={}",
                summary.cache_capacity,
                summary.registered_entries,
                summary.bootstrap_registrations,
                summary.cache_hits,
                summary.cache_misses,
                summary.stale_address_rejections,
                summary.stale_version_rejections,
                summary.hot_path_registration_attempts,
                summary.hot_path_registration_rejections,
                summary.per_token_registrations,
                summary.pinned_host_registrations,
                summary.gpu_direct_registrations,
                summary.transport_events,
                summary.phase_handoff_syncs,
                summary.registration_cache_hit_rate_per_mille,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push(
            "transport_registration_cache",
            false,
            format!("transport registration probe failed: {err:?}"),
        ),
    }
}
