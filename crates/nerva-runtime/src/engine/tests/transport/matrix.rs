use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::matrix::types::TransportCapabilityMatrixStatus;

#[test]
fn transport_capability_matrix_reports_required_sizes_and_degradation() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_transport_capability_matrix_probe().unwrap();

    assert_eq!(summary.status, TransportCapabilityMatrixStatus::Ok);
    assert_eq!(summary.sizes, 6);
    assert_eq!(summary.entries.len(), 24);
    assert_eq!(summary.decode_entries, 12);
    assert_eq!(summary.prefill_entries, 12);
    assert_eq!(summary.host_staged_entries, 18);
    assert_eq!(summary.cpu_produced_entries, 6);
    assert_eq!(summary.gpu_direct_entries, 0);
    assert_eq!(summary.mapped_pinned_entries, 0);
    assert_eq!(summary.degraded_to_pinned_host_entries, 12);
    assert_eq!(summary.supported_unverified_entries, 6);
    assert_eq!(summary.supported_verified_entries, 6);
    assert_eq!(summary.unsupported_entries, 0);
    assert_eq!(summary.pageable_copies, 0);
    assert_eq!(summary.per_token_registrations, 0);
    assert_eq!(
        summary.registration_cache_hits,
        summary.entries.len() as u64
    );
    assert_eq!(summary.credit_stall_ns, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.explicit_copy_bytes > 0);
    assert!(summary.estimated_cpu_core_ns > 0);
    assert!(summary.dram_read_bytes > 0);
    assert!(summary.dram_write_bytes > 0);
    assert!(summary.pcie_tx_bytes > 0);
    assert!(summary.pcie_rx_bytes > 0);
    assert!(summary.total_estimated_visible_ns > 0);
    assert!(summary.p50_estimated_visible_ns > 0);
    assert!(summary.p95_estimated_visible_ns >= summary.p50_estimated_visible_ns);
    assert!(summary.p99_estimated_visible_ns >= summary.p95_estimated_visible_ns);
    assert!(
        summary
            .entries
            .iter()
            .all(|entry| entry.effective_payload_bandwidth_bps > 0)
    );
    assert!(summary.entries.iter().all(|entry| entry.queue_depth > 0));
    assert!(
        summary
            .entries
            .iter()
            .all(|entry| entry.registration_cache_hit)
    );
    let json = summary.to_json();
    assert!(json.contains("\"requested_path\":\"A_GPU_DIRECT_RDMA\""));
    assert!(json.contains("\"size_bytes\":32768"));
    assert!(json.contains("\"capability_result\":\"DEGRADED_TO_PINNED_HOST\""));
    assert!(json.contains("\"metric_source\":\"estimated_model\""));
    assert!(json.contains("\"p95_estimated_visible_ns\""));
    assert!(json.contains("\"effective_payload_bandwidth_bps\""));
    assert!(json.contains("\"estimated_cpu_core_ns\""));
    assert!(json.contains("\"dram_read_bytes\""));
    assert!(json.contains("\"dram_write_bytes\""));
    assert!(json.contains("\"pcie_tx_bytes\""));
    assert!(json.contains("\"pcie_rx_bytes\""));
    assert!(json.contains("\"registration_cache_hits\""));
    assert!(json.contains("\"credit_stall_ns\""));
}
