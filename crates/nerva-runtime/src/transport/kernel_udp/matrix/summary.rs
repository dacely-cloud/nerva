use crate::transport::kernel_udp::matrix::entry::KernelUdpBaselineMatrixEntry;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelUdpBaselineMatrixStatus {
    Ok,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelUdpBaselineMatrixSummary {
    pub status: KernelUdpBaselineMatrixStatus,
    pub backend: &'static str,
    pub measured_sizes: u64,
    pub entries: Vec<KernelUdpBaselineMatrixEntry>,
    pub total_payload_bytes: usize,
    pub total_wire_bytes: usize,
    pub total_runtime_timestamp_events: u64,
    pub total_transport_events: u64,
    pub p50_max_ns: u64,
    pub p95_max_ns: u64,
    pub p99_max_ns: u64,
    pub min_effective_payload_bandwidth_bps: u64,
    pub packet_loss: u64,
    pub checksum_failures: u64,
    pub baseline_only: bool,
    pub production_tensor_data_plane: bool,
    pub pageable_copies: u64,
    pub per_token_registrations: u64,
    pub hot_path_allocations: u64,
}

impl KernelUdpBaselineMatrixSummary {
    pub(crate) fn from_entries(
        entries: Vec<KernelUdpBaselineMatrixEntry>,
        hot_path_allocations: u64,
    ) -> Self {
        Self {
            status: KernelUdpBaselineMatrixStatus::Ok,
            backend: "kernel_udp_test",
            measured_sizes: entries.len() as u64,
            total_payload_bytes: entries.iter().map(|entry| entry.payload_bytes).sum(),
            total_wire_bytes: entries.iter().map(|entry| entry.total_wire_bytes).sum(),
            total_runtime_timestamp_events: entries
                .iter()
                .map(|entry| entry.runtime_timestamp_events)
                .sum(),
            total_transport_events: entries.iter().map(|entry| entry.transport_events).sum(),
            p50_max_ns: entries
                .iter()
                .map(|entry| entry.p50_completion_latency_ns)
                .max()
                .unwrap_or(0),
            p95_max_ns: entries
                .iter()
                .map(|entry| entry.p95_completion_latency_ns)
                .max()
                .unwrap_or(0),
            p99_max_ns: entries
                .iter()
                .map(|entry| entry.p99_completion_latency_ns)
                .max()
                .unwrap_or(0),
            min_effective_payload_bandwidth_bps: entries
                .iter()
                .map(|entry| entry.effective_payload_bandwidth_bps)
                .min()
                .unwrap_or(0),
            packet_loss: entries.iter().map(|entry| entry.packet_loss).sum(),
            checksum_failures: entries.iter().map(|entry| entry.checksum_failures).sum(),
            baseline_only: true,
            production_tensor_data_plane: false,
            pageable_copies: 0,
            per_token_registrations: 0,
            hot_path_allocations,
            entries,
        }
    }

    pub fn passed(&self) -> bool {
        matches!(self.status, KernelUdpBaselineMatrixStatus::Ok)
            && self.backend == "kernel_udp_test"
            && self.measured_sizes >= 3
            && self.entries.len() as u64 == self.measured_sizes
            && self.total_payload_bytes > 0
            && self.total_wire_bytes > self.total_payload_bytes
            && self.total_runtime_timestamp_events > 0
            && self.total_transport_events == self.total_runtime_timestamp_events
            && self.p50_max_ns > 0
            && self.p95_max_ns >= self.p50_max_ns
            && self.p99_max_ns >= self.p95_max_ns
            && self.min_effective_payload_bandwidth_bps > 0
            && self.packet_loss == 0
            && self.checksum_failures == 0
            && self.baseline_only
            && !self.production_tensor_data_plane
            && self.pageable_copies == 0
            && self.per_token_registrations == 0
            && self.hot_path_allocations == 0
            && self.entries.iter().all(entry_passed)
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            KernelUdpBaselineMatrixStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"backend\":\"{}\",\"measured_sizes\":{},\"total_payload_bytes\":{},\"total_wire_bytes\":{},\"total_runtime_timestamp_events\":{},\"total_transport_events\":{},\"p50_max_ns\":{},\"p95_max_ns\":{},\"p99_max_ns\":{},\"min_effective_payload_bandwidth_bps\":{},\"packet_loss\":{},\"checksum_failures\":{},\"baseline_only\":{},\"production_tensor_data_plane\":{},\"pageable_copies\":{},\"per_token_registrations\":{},\"hot_path_allocations\":{},\"entries\":{}}}",
            status,
            self.backend,
            self.measured_sizes,
            self.total_payload_bytes,
            self.total_wire_bytes,
            self.total_runtime_timestamp_events,
            self.total_transport_events,
            self.p50_max_ns,
            self.p95_max_ns,
            self.p99_max_ns,
            self.min_effective_payload_bandwidth_bps,
            self.packet_loss,
            self.checksum_failures,
            self.baseline_only,
            self.production_tensor_data_plane,
            self.pageable_copies,
            self.per_token_registrations,
            self.hot_path_allocations,
            entries_json(&self.entries),
        )
    }
}

fn entry_passed(entry: &KernelUdpBaselineMatrixEntry) -> bool {
    entry.payload_bytes > 0
        && entry.chunk_payload_bytes > 0
        && entry.chunks > 0
        && entry.total_wire_bytes > entry.payload_bytes
        && entry.p50_completion_latency_ns > 0
        && entry.p95_completion_latency_ns >= entry.p50_completion_latency_ns
        && entry.p99_completion_latency_ns >= entry.p95_completion_latency_ns
        && entry.total_completion_latency_ns >= entry.p99_completion_latency_ns
        && entry.effective_payload_bandwidth_bps > 0
        && entry.runtime_timestamp_events == entry.chunks
        && entry.transport_events == entry.chunks
        && entry.packet_loss == 0
        && entry.checksum_failures == 0
}

fn entries_json(entries: &[KernelUdpBaselineMatrixEntry]) -> String {
    let items = entries
        .iter()
        .map(KernelUdpBaselineMatrixEntry::to_json)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}
