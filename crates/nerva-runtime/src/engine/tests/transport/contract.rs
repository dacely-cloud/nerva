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
    assert_eq!(summary.receive_queue_capacity, 4);
    assert_eq!(summary.completion_queue_capacity, 2);
    assert_eq!(summary.preposted_receives, 2);
    assert_eq!(summary.pending_completions, 1);
    assert_eq!(summary.sends, 2);
    assert_eq!(summary.completions, 1);
    assert_eq!(summary.bytes_completed, 32 * 1024);
    assert_eq!(summary.receive_queue_full_rejections, 1);
    assert_eq!(summary.completion_queue_full_rejections, 1);
    assert_eq!(summary.unposted_send_rejections, 1);
    assert_eq!(summary.stale_version_rejections, 1);
    assert_eq!(summary.descriptor_rejections, 1);
    assert_eq!(summary.pre_visibility_consume_rejections, 1);
    assert_eq!(summary.visibility_fences, 1);
    assert_eq!(summary.visible_consumes, 1);
    assert_eq!(summary.per_transfer_registrations, 0);
    assert_eq!(summary.transport_events, 1);
    assert_eq!(summary.phase_handoff_syncs, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(
        summary
            .to_json()
            .contains("\"per_transfer_registrations\":0")
    );
    assert!(summary.to_json().contains("\"preposted_receives\":2"));
    assert!(summary.to_json().contains("\"pending_completions\":1"));
    assert!(
        summary
            .to_json()
            .contains("\"receive_queue_full_rejections\":1")
    );
    assert!(
        summary
            .to_json()
            .contains("\"completion_queue_full_rejections\":1")
    );
    assert!(summary.to_json().contains("\"unposted_send_rejections\":1"));
    assert!(
        summary
            .to_json()
            .contains("\"pre_visibility_consume_rejections\":1")
    );
}
