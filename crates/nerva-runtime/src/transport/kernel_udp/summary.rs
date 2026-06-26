#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelUdpBaselineStatus {
    Ok,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelUdpBaselineSummary {
    pub status: KernelUdpBaselineStatus,
    pub backend: &'static str,
    pub protocol_version: u16,
    pub request_id: u64,
    pub sequence_id: u64,
    pub block_id: u64,
    pub block_version: u64,
    pub payload_bytes: usize,
    pub chunk_payload_bytes: usize,
    pub chunks: u64,
    pub protocol_header_bytes: usize,
    pub total_wire_bytes: usize,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub validated_packets: u64,
    pub bytes_received: usize,
    pub p50_completion_latency_ns: u64,
    pub p95_completion_latency_ns: u64,
    pub p99_completion_latency_ns: u64,
    pub total_completion_latency_ns: u64,
    pub effective_payload_bandwidth_bps: u64,
    pub runtime_timestamp_events: u64,
    pub transport_events: u64,
    pub packet_loss: u64,
    pub checksum_failures: u64,
    pub baseline_only: bool,
    pub production_tensor_data_plane: bool,
    pub pageable_copies: u64,
    pub per_token_registrations: u64,
    pub hot_path_allocations: u64,
}

impl KernelUdpBaselineSummary {
    pub fn passed(&self) -> bool {
        matches!(self.status, KernelUdpBaselineStatus::Ok)
            && self.backend == "kernel_udp_test"
            && self.protocol_version > 0
            && self.request_id > 0
            && self.sequence_id > 0
            && self.block_id > 0
            && self.block_version > 0
            && self.payload_bytes > 0
            && self.chunk_payload_bytes > 0
            && self.chunks > 0
            && self.total_wire_bytes > self.payload_bytes
            && self.packets_sent == self.chunks
            && self.packets_received == self.chunks
            && self.validated_packets == self.chunks
            && self.bytes_received == self.payload_bytes
            && self.p50_completion_latency_ns > 0
            && self.p95_completion_latency_ns >= self.p50_completion_latency_ns
            && self.p99_completion_latency_ns >= self.p95_completion_latency_ns
            && self.total_completion_latency_ns >= self.p99_completion_latency_ns
            && self.effective_payload_bandwidth_bps > 0
            && self.runtime_timestamp_events == self.chunks
            && self.transport_events == self.chunks
            && self.packet_loss == 0
            && self.checksum_failures == 0
            && self.baseline_only
            && !self.production_tensor_data_plane
            && self.pageable_copies == 0
            && self.per_token_registrations == 0
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            KernelUdpBaselineStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"backend\":\"{}\",\"protocol_version\":{},\"request_id\":{},\"sequence_id\":{},\"block_id\":{},\"block_version\":{},\"payload_bytes\":{},\"chunk_payload_bytes\":{},\"chunks\":{},\"protocol_header_bytes\":{},\"total_wire_bytes\":{},\"packets_sent\":{},\"packets_received\":{},\"validated_packets\":{},\"bytes_received\":{},\"p50_completion_latency_ns\":{},\"p95_completion_latency_ns\":{},\"p99_completion_latency_ns\":{},\"total_completion_latency_ns\":{},\"effective_payload_bandwidth_bps\":{},\"runtime_timestamp_events\":{},\"transport_events\":{},\"packet_loss\":{},\"checksum_failures\":{},\"baseline_only\":{},\"production_tensor_data_plane\":{},\"pageable_copies\":{},\"per_token_registrations\":{},\"hot_path_allocations\":{}}}",
            status,
            self.backend,
            self.protocol_version,
            self.request_id,
            self.sequence_id,
            self.block_id,
            self.block_version,
            self.payload_bytes,
            self.chunk_payload_bytes,
            self.chunks,
            self.protocol_header_bytes,
            self.total_wire_bytes,
            self.packets_sent,
            self.packets_received,
            self.validated_packets,
            self.bytes_received,
            self.p50_completion_latency_ns,
            self.p95_completion_latency_ns,
            self.p99_completion_latency_ns,
            self.total_completion_latency_ns,
            self.effective_payload_bandwidth_bps,
            self.runtime_timestamp_events,
            self.transport_events,
            self.packet_loss,
            self.checksum_failures,
            self.baseline_only,
            self.production_tensor_data_plane,
            self.pageable_copies,
            self.per_token_registrations,
            self.hot_path_allocations,
        )
    }
}
