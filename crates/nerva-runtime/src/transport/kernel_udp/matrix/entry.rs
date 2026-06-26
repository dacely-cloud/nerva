use crate::transport::kernel_udp::summary::KernelUdpBaselineSummary;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelUdpBaselineMatrixEntry {
    pub payload_bytes: usize,
    pub chunk_payload_bytes: usize,
    pub chunks: u64,
    pub total_wire_bytes: usize,
    pub p50_completion_latency_ns: u64,
    pub p95_completion_latency_ns: u64,
    pub p99_completion_latency_ns: u64,
    pub total_completion_latency_ns: u64,
    pub effective_payload_bandwidth_bps: u64,
    pub runtime_timestamp_events: u64,
    pub transport_events: u64,
    pub packet_loss: u64,
    pub checksum_failures: u64,
}

impl KernelUdpBaselineMatrixEntry {
    pub(crate) fn from_summary(summary: &KernelUdpBaselineSummary) -> Self {
        Self {
            payload_bytes: summary.payload_bytes,
            chunk_payload_bytes: summary.chunk_payload_bytes,
            chunks: summary.chunks,
            total_wire_bytes: summary.total_wire_bytes,
            p50_completion_latency_ns: summary.p50_completion_latency_ns,
            p95_completion_latency_ns: summary.p95_completion_latency_ns,
            p99_completion_latency_ns: summary.p99_completion_latency_ns,
            total_completion_latency_ns: summary.total_completion_latency_ns,
            effective_payload_bandwidth_bps: summary.effective_payload_bandwidth_bps,
            runtime_timestamp_events: summary.runtime_timestamp_events,
            transport_events: summary.transport_events,
            packet_loss: summary.packet_loss,
            checksum_failures: summary.checksum_failures,
        }
    }

    pub(crate) fn to_json(&self) -> String {
        format!(
            "{{\"payload_bytes\":{},\"chunk_payload_bytes\":{},\"chunks\":{},\"total_wire_bytes\":{},\"p50_completion_latency_ns\":{},\"p95_completion_latency_ns\":{},\"p99_completion_latency_ns\":{},\"total_completion_latency_ns\":{},\"effective_payload_bandwidth_bps\":{},\"runtime_timestamp_events\":{},\"transport_events\":{},\"packet_loss\":{},\"checksum_failures\":{}}}",
            self.payload_bytes,
            self.chunk_payload_bytes,
            self.chunks,
            self.total_wire_bytes,
            self.p50_completion_latency_ns,
            self.p95_completion_latency_ns,
            self.p99_completion_latency_ns,
            self.total_completion_latency_ns,
            self.effective_payload_bandwidth_bps,
            self.runtime_timestamp_events,
            self.transport_events,
            self.packet_loss,
            self.checksum_failures,
        )
    }
}
