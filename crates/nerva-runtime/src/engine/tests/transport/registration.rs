use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::registration::lifetime::summary::TransportRegistrationLifecycleStatus;
use crate::transport::registration::summary::TransportRegistrationStatus;

#[test]
fn transport_registration_probe_reuses_stable_block_registrations() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_transport_registration_probe().unwrap();

    assert_eq!(summary.status, TransportRegistrationStatus::Ok);
    assert_eq!(summary.bootstrap_registrations, 4);
    assert_eq!(summary.registered_entries, 4);
    assert_eq!(summary.cache_hits, 4);
    assert_eq!(summary.cache_misses, 1);
    assert_eq!(summary.stale_address_rejections, 1);
    assert_eq!(summary.stale_version_rejections, 1);
    assert_eq!(summary.hot_path_registration_attempts, 1);
    assert_eq!(summary.hot_path_registration_rejections, 1);
    assert_eq!(summary.per_token_registrations, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(summary.transport_events, 4);
    assert_eq!(summary.phase_handoff_syncs, 2);
    assert!(summary.registration_cache_hit_rate_per_mille > 0);
    assert!(summary.passed());
    let json = summary.to_json();
    assert!(json.contains("\"per_token_registrations\":0"));
    assert!(json.contains("\"stale_address_rejections\":1"));
    assert!(json.contains("\"hot_path_registration_rejections\":1"));
}

#[test]
fn transport_registration_lifecycle_revokes_before_allocation_lifetime_ends() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_transport_registration_lifecycle_probe()
        .unwrap();

    assert_eq!(summary.status, TransportRegistrationLifecycleStatus::Ok);
    assert_eq!(summary.bootstrap_registrations, 4);
    assert_eq!(summary.registered_before_revoke, 4);
    assert!(summary.explicit_key_revocations > 0);
    assert!(summary.block_lifetime_revocations > 0);
    assert!(summary.destroy_revocations > 0);
    assert_eq!(summary.total_revocations, summary.bootstrap_registrations);
    assert_eq!(summary.final_registered_entries, 0);
    assert!(summary.lookup_hits_before_revoke > 0);
    assert!(summary.post_revoke_misses > 0);
    assert!(summary.safe_move_post_revoke_misses > 0);
    assert!(summary.stale_mapping_reuse_rejections > 0);
    assert_eq!(summary.revocation_syncs, summary.total_revocations);
    assert!(summary.phase_handoff_syncs > summary.revocation_syncs);
    assert!(summary.transport_events > 0);
    assert!(summary.stale_handle_reuse_prevented);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.passed());
    let json = summary.to_json();
    assert!(json.contains("\"final_registered_entries\":0"));
    assert!(json.contains("\"stale_handle_reuse_prevented\":true"));
}
