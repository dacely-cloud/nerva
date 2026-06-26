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
    assert_eq!(summary.false_gpu_direct_claims, 0);
    assert!(
        summary.gpu_memory_export_verified_entries >= summary.gpu_export_without_nic_direct_entries
    );
    assert!(
        summary.cuda_vmm_posix_fd_export_verified_entries
            <= summary.gpu_memory_export_verified_entries
    );
    assert!(summary.gpu_direct_rdma_verified_entries <= summary.gpu_memory_export_verified_entries);
    assert!(summary.total_payload_bytes > 0);
    assert_eq!(summary.pageable_copies, 0);
    assert_eq!(summary.per_token_registrations, 0);
    assert_eq!(
        summary.registration_cache_hits,
        summary.entries.len() as u64
    );
    assert_eq!(summary.registration_cache_hit_rate_per_mille, 1_000);
    assert_eq!(summary.estimated_nic_utilization_per_mille, 1_000);
    assert_eq!(
        summary.visible_non_overlapped_ns,
        summary.total_estimated_visible_ns
    );
    assert_eq!(
        summary.host_event_wait_ns,
        summary.total_estimated_visible_ns
    );
    assert_eq!(summary.gpu_idle_ns, 0);
    assert!(summary.max_queue_depth >= 4);
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
    assert!(
        summary
            .entries
            .iter()
            .all(|entry| entry.visible_non_overlapped_ns == entry.estimated_visible_ns)
    );
    assert!(
        summary
            .entries
            .iter()
            .all(|entry| entry.host_event_wait_ns == entry.estimated_visible_ns)
    );
    assert!(summary.entries.iter().all(|entry| entry.gpu_idle_ns == 0));
    assert!(summary.entries.iter().all(|entry| entry.queue_depth > 0));
    assert!(
        summary
            .entries
            .iter()
            .all(|entry| entry.registration_cache_hit)
    );
    assert!(summary.entries.iter().all(|entry| {
        !entry.gpu_export_without_nic_direct
            || (entry.gpu_memory_export_verified && !entry.gpu_direct_rdma_verified)
    }));
    let json = summary.to_json();
    assert!(json.contains("\"requested_path\":\"A_GPU_DIRECT_RDMA\""));
    assert!(json.contains("\"size_bytes\":32768"));
    assert!(json.contains("\"capability_result\":\"DEGRADED_TO_PINNED_HOST\""));
    assert!(json.contains("\"metric_source\":\"estimated_model\""));
    assert!(json.contains("\"p95_estimated_visible_ns\""));
    assert!(json.contains("\"effective_payload_bandwidth_bps\""));
    assert!(json.contains("\"visible_non_overlapped_ns\""));
    assert!(json.contains("\"host_event_wait_ns\""));
    assert!(json.contains("\"gpu_idle_ns\""));
    assert!(json.contains("\"estimated_cpu_core_ns\""));
    assert!(json.contains("\"dram_read_bytes\""));
    assert!(json.contains("\"dram_write_bytes\""));
    assert!(json.contains("\"pcie_tx_bytes\""));
    assert!(json.contains("\"pcie_rx_bytes\""));
    assert!(json.contains("\"registration_cache_hits\""));
    assert!(json.contains("\"registration_cache_hit_rate_per_mille\""));
    assert!(json.contains("\"estimated_nic_utilization_per_mille\""));
    assert!(json.contains("\"credit_stall_ns\""));
    assert!(json.contains("\"gpu_memory_export_verified\""));
    assert!(json.contains("\"cuda_vmm_posix_fd_export_verified\""));
    assert!(json.contains("\"gpu_direct_rdma_verified\""));
    assert!(json.contains("\"gpu_export_without_nic_direct\""));
    assert!(json.contains("\"gpu_memory_export_verified_entries\""));
    assert!(json.contains("\"false_gpu_direct_claims\""));
}
