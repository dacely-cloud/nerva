use crate::transport::matrix::aggregate::aggregate_entries;
use crate::transport::matrix::types::{
    TransportCapabilityMatrixEntry, TransportCapabilityMatrixStatus,
    TransportCapabilityMatrixSummary,
};

pub(crate) fn transport_capability_matrix_summary(
    sizes: u64,
    entries: Vec<TransportCapabilityMatrixEntry>,
    hot_path_allocations: u64,
) -> TransportCapabilityMatrixSummary {
    let counters = aggregate_entries(&entries);

    TransportCapabilityMatrixSummary {
        status: TransportCapabilityMatrixStatus::Ok,
        sizes,
        entries,
        decode_entries: counters.decode_entries,
        prefill_entries: counters.prefill_entries,
        gpu_direct_entries: counters.gpu_direct_entries,
        host_staged_entries: counters.host_staged_entries,
        cpu_produced_entries: counters.cpu_produced_entries,
        mapped_pinned_entries: counters.mapped_pinned_entries,
        total_payload_bytes: counters.total_payload_bytes,
        supported_verified_entries: counters.supported_verified_entries,
        supported_unverified_entries: counters.supported_unverified_entries,
        degraded_to_pinned_host_entries: counters.degraded_to_pinned_host_entries,
        unsupported_entries: counters.unsupported_entries,
        gpu_memory_export_verified_entries: counters.gpu_memory_export_verified_entries,
        cuda_vmm_posix_fd_export_verified_entries: counters
            .cuda_vmm_posix_fd_export_verified_entries,
        gpu_direct_rdma_verified_entries: counters.gpu_direct_rdma_verified_entries,
        gpu_export_without_nic_direct_entries: counters.gpu_export_without_nic_direct_entries,
        false_gpu_direct_claims: counters.false_gpu_direct_claims,
        total_estimated_visible_ns: counters.total_estimated_visible_ns,
        visible_non_overlapped_ns: counters.visible_non_overlapped_ns,
        host_event_wait_ns: counters.host_event_wait_ns,
        gpu_idle_ns: counters.gpu_idle_ns,
        p50_estimated_visible_ns: counters.p50_estimated_visible_ns,
        p95_estimated_visible_ns: counters.p95_estimated_visible_ns,
        p99_estimated_visible_ns: counters.p99_estimated_visible_ns,
        explicit_copy_bytes: counters.explicit_copy_bytes,
        nic_tx_bytes: counters.nic_tx_bytes,
        nic_rx_bytes: counters.nic_rx_bytes,
        estimated_cpu_core_ns: counters.estimated_cpu_core_ns,
        dram_read_bytes: counters.dram_read_bytes,
        dram_write_bytes: counters.dram_write_bytes,
        pcie_tx_bytes: counters.pcie_tx_bytes,
        pcie_rx_bytes: counters.pcie_rx_bytes,
        pageable_copies: counters.pageable_copies,
        per_token_registrations: counters.per_token_registrations,
        registration_cache_hits: counters.registration_cache_hits,
        registration_cache_hit_rate_per_mille: counters.registration_cache_hit_rate_per_mille,
        max_queue_depth: counters.max_queue_depth,
        estimated_nic_utilization_per_mille: counters.estimated_nic_utilization_per_mille,
        credit_stall_ns: counters.credit_stall_ns,
        hot_path_allocations,
        error: None,
    }
}
