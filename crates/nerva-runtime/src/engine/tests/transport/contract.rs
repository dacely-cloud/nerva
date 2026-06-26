use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::contract::summary::TransportContractStatus;

#[test]
fn transport_contract_probe_requires_registered_preposted_transfer() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_transport_contract_probe().unwrap();

    assert_eq!(summary.status, TransportContractStatus::Ok);
    assert!(summary.passed());
    assert_eq!(summary.backend, "rdma_pinned_host");
    assert_eq!(summary.registrations, 2);
    assert_eq!(summary.registered_entries, 2);
    assert_eq!(summary.preposted_receives, 0);
    assert_eq!(summary.sends, 1);
    assert_eq!(summary.completions, 1);
    assert_eq!(summary.bytes_completed, 32 * 1024);
    assert_eq!(summary.unposted_send_rejections, 1);
    assert_eq!(summary.stale_version_rejections, 1);
    assert_eq!(summary.descriptor_rejections, 1);
    assert_eq!(summary.per_transfer_registrations, 0);
    assert_eq!(summary.transport_events, 1);
    assert_eq!(summary.phase_handoff_syncs, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(
        summary
            .to_json()
            .contains("\"per_transfer_registrations\":0")
    );
    assert!(summary.to_json().contains("\"unposted_send_rejections\":1"));
}
