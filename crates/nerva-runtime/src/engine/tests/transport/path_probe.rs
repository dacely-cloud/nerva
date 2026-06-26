use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::probe::status::TransportPathProbeStatus;

#[test]
fn transport_path_probe_reports_explicit_fallback_without_hot_allocations() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_transport_path_probe().unwrap();

    assert_eq!(summary.status, TransportPathProbeStatus::Ok);
    assert_eq!(summary.requests, 7);
    assert_eq!(summary.decode_requests, 4);
    assert_eq!(summary.prefill_requests, 3);
    assert_eq!(summary.pinned_host_paths, 6);
    assert_eq!(summary.cpu_produced_paths, 1);
    assert_eq!(summary.transport_events, 7);
    assert_eq!(summary.copy_events, 6);
    assert_eq!(summary.sync_events, 7);
    assert_eq!(summary.phase_handoff_syncs, 7);
    assert_eq!(summary.fallback_decisions, 6);
    assert_eq!(summary.estimated_events, 20);
    assert_eq!(summary.estimated_latency_ns, summary.total_latency_ns);
    assert_eq!(summary.pageable_copies, 0);
    assert_eq!(summary.per_token_registrations, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.explicit_copy_bytes > 0);
    assert!(summary.to_json().contains("\"pinned_host_paths\":6"));
}
