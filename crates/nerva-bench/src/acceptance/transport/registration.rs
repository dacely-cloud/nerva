use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::transport::registration::lifetime::summary::TransportRegistrationLifecycleStatus;
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

pub(crate) fn push_transport_registration_lifecycle(
    report: &mut AcceptanceReport,
    runtime: &Runtime,
) {
    match runtime.run_transport_registration_lifecycle_probe() {
        Ok(summary) => report.push(
            "transport_registration_lifecycle",
            matches!(summary.status, TransportRegistrationLifecycleStatus::Ok)
                && summary.passed()
                && summary.total_revocations == summary.bootstrap_registrations
                && summary.final_registered_entries == 0
                && summary.post_revoke_misses > 0
                && summary.safe_move_post_revoke_misses > 0
                && summary.stale_mapping_reuse_rejections > 0
                && summary.revocation_syncs == summary.total_revocations
                && summary.stale_handle_reuse_prevented
                && summary.hot_path_allocations == 0,
            format!(
                "bootstrap_registrations={} registered_before_revoke={} explicit_key_revocations={} block_lifetime_revocations={} destroy_revocations={} total_revocations={} final_registered_entries={} lookup_hits_before_revoke={} post_revoke_misses={} safe_move_post_revoke_misses={} stale_mapping_reuse_rejections={} revocation_syncs={} phase_handoff_syncs={} transport_events={} stale_handle_reuse_prevented={} hot_path_allocations={}",
                summary.bootstrap_registrations,
                summary.registered_before_revoke,
                summary.explicit_key_revocations,
                summary.block_lifetime_revocations,
                summary.destroy_revocations,
                summary.total_revocations,
                summary.final_registered_entries,
                summary.lookup_hits_before_revoke,
                summary.post_revoke_misses,
                summary.safe_move_post_revoke_misses,
                summary.stale_mapping_reuse_rejections,
                summary.revocation_syncs,
                summary.phase_handoff_syncs,
                summary.transport_events,
                summary.stale_handle_reuse_prevented,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push(
            "transport_registration_lifecycle",
            false,
            format!("transport registration lifecycle probe failed: {err:?}"),
        ),
    }
}
