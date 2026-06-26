use crate::transport::json::{json_opt_static_str, memory_tier_to_str};
use crate::transport::matrix::types::{
    TransportCapabilityMatrixEntry, TransportCapabilityMatrixStatus,
    TransportCapabilityMatrixSummary, TransportMatrixRequestedPath,
};

impl TransportMatrixRequestedPath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GpuDirectRdma => "A_GPU_DIRECT_RDMA",
            Self::PinnedHostBounce => "B_PINNED_HOST_BOUNCE",
            Self::CpuProducedBoundary => "C_CPU_PRODUCED_BOUNDARY",
            Self::MappedPinnedWrite => "D_MAPPED_PINNED_WRITE",
        }
    }
}

impl TransportCapabilityMatrixEntry {
    pub fn to_json(self) -> String {
        format!(
            "{{\"requested_path\":\"{}\",\"size_bytes\":{},\"mode\":\"{}\",\"source_tier\":\"{}\",\"destination_tier\":\"{}\",\"selected_path\":\"{}\",\"class\":\"{}\",\"capability_result\":\"{}\",\"estimated_visible_ns\":{},\"metric_source\":\"estimated_model\",\"effective_payload_bandwidth_bps\":{},\"estimated_cpu_core_ns\":{},\"dram_read_bytes\":{},\"dram_write_bytes\":{},\"pcie_tx_bytes\":{},\"pcie_rx_bytes\":{},\"explicit_copy_bytes\":{},\"nic_tx_bytes\":{},\"nic_rx_bytes\":{},\"pageable_copy\":{},\"per_token_registration\":{},\"registration_cache_hit\":{},\"queue_depth\":{},\"credit_stall_ns\":{}}}",
            self.requested_path.as_str(),
            self.size_bytes,
            self.mode.as_str(),
            memory_tier_to_str(self.source_tier),
            memory_tier_to_str(self.destination_tier),
            self.selected_path.as_str(),
            self.class.as_str(),
            self.capability_result.as_str(),
            self.estimated_visible_ns,
            self.effective_payload_bandwidth_bps,
            self.estimated_cpu_core_ns,
            self.dram_read_bytes,
            self.dram_write_bytes,
            self.pcie_tx_bytes,
            self.pcie_rx_bytes,
            self.explicit_copy_bytes,
            self.nic_tx_bytes,
            self.nic_rx_bytes,
            self.pageable_copy,
            self.per_token_registration,
            self.registration_cache_hit,
            self.queue_depth,
            self.credit_stall_ns,
        )
    }
}

impl TransportCapabilityMatrixSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            TransportCapabilityMatrixStatus::Ok => "ok",
            TransportCapabilityMatrixStatus::Failed => "failed",
        };
        let mut entries = String::from("[");
        for (index, entry) in self.entries.iter().enumerate() {
            if index != 0 {
                entries.push(',');
            }
            entries.push_str(&entry.to_json());
        }
        entries.push(']');
        format!(
            "{{\"status\":\"{}\",\"sizes\":{},\"entries_count\":{},\"decode_entries\":{},\"prefill_entries\":{},\"gpu_direct_entries\":{},\"host_staged_entries\":{},\"cpu_produced_entries\":{},\"mapped_pinned_entries\":{},\"supported_verified_entries\":{},\"supported_unverified_entries\":{},\"degraded_to_pinned_host_entries\":{},\"unsupported_entries\":{},\"total_estimated_visible_ns\":{},\"p50_estimated_visible_ns\":{},\"p95_estimated_visible_ns\":{},\"p99_estimated_visible_ns\":{},\"explicit_copy_bytes\":{},\"nic_tx_bytes\":{},\"nic_rx_bytes\":{},\"estimated_cpu_core_ns\":{},\"dram_read_bytes\":{},\"dram_write_bytes\":{},\"pcie_tx_bytes\":{},\"pcie_rx_bytes\":{},\"pageable_copies\":{},\"per_token_registrations\":{},\"registration_cache_hits\":{},\"credit_stall_ns\":{},\"hot_path_allocations\":{},\"error\":{},\"entries\":{}}}",
            status,
            self.sizes,
            self.entries.len(),
            self.decode_entries,
            self.prefill_entries,
            self.gpu_direct_entries,
            self.host_staged_entries,
            self.cpu_produced_entries,
            self.mapped_pinned_entries,
            self.supported_verified_entries,
            self.supported_unverified_entries,
            self.degraded_to_pinned_host_entries,
            self.unsupported_entries,
            self.total_estimated_visible_ns,
            self.p50_estimated_visible_ns,
            self.p95_estimated_visible_ns,
            self.p99_estimated_visible_ns,
            self.explicit_copy_bytes,
            self.nic_tx_bytes,
            self.nic_rx_bytes,
            self.estimated_cpu_core_ns,
            self.dram_read_bytes,
            self.dram_write_bytes,
            self.pcie_tx_bytes,
            self.pcie_rx_bytes,
            self.pageable_copies,
            self.per_token_registrations,
            self.registration_cache_hits,
            self.credit_stall_ns,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
            entries,
        )
    }
}
