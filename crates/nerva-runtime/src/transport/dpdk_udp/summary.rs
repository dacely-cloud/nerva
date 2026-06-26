use crate::capabilities::snapshot::CapabilityState;
use crate::transport::dpdk_udp::protocol::DpdkUdpMemoryPath;
use crate::transport::json::json_opt_static_str;
use crate::transport::path::TransferMode;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DpdkUdpProtocolStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DpdkUdpProtocolSummary {
    pub status: DpdkUdpProtocolStatus,
    pub protocol_version: u32,
    pub request_id: u64,
    pub sequence_id: u64,
    pub block_id: u64,
    pub block_version: u64,
    pub mode: TransferMode,
    pub selected_path: DpdkUdpMemoryPath,
    pub capability_result: CapabilityState,
    pub payload_bytes: usize,
    pub chunk_payload_bytes: usize,
    pub chunks: u32,
    pub protocol_header_bytes: usize,
    pub total_wire_bytes: usize,
    pub preposted_receives: u32,
    pub credit_window_chunks: u32,
    pub credit_windows: u32,
    pub credit_stalls: u32,
    pub sender_retention_chunks: u32,
    pub receiver_bitmap_words: u32,
    pub nack_ranges: u32,
    pub selective_retransmits: u32,
    pub ack_packets: u32,
    pub mbufs_preallocated: u32,
    pub rings_preallocated: u32,
    pub direct_gpu_memory_claimed: bool,
    pub pinned_host_required: bool,
    pub fallback_decisions: u64,
    pub transport_events: u64,
    pub phase_handoff_syncs: u64,
    pub pageable_copies: u64,
    pub per_token_registrations: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl DpdkUdpProtocolSummary {
    pub fn passed(&self) -> bool {
        let legal_memory_path = if self.direct_gpu_memory_claimed {
            !self.pinned_host_required && self.fallback_decisions == 0
        } else {
            self.pinned_host_required && self.fallback_decisions >= 1
        };
        matches!(self.status, DpdkUdpProtocolStatus::Ok)
            && self.chunks > 0
            && self.preposted_receives == self.chunks
            && self.sender_retention_chunks >= self.chunks
            && self.receiver_bitmap_words > 0
            && self.ack_packets == 0
            && self.selective_retransmits == self.nack_ranges
            && self.mbufs_preallocated >= self.chunks
            && self.rings_preallocated >= 2
            && legal_memory_path
            && self.transport_events >= u64::from(self.chunks)
            && self.phase_handoff_syncs >= 1
            && self.pageable_copies == 0
            && self.per_token_registrations == 0
            && self.hot_path_allocations == 0
    }

    pub fn to_json(self) -> String {
        let status = match self.status {
            DpdkUdpProtocolStatus::Ok => "ok",
            DpdkUdpProtocolStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"protocol_version\":{},\"request_id\":{},\"sequence_id\":{},\"block_id\":{},\"block_version\":{},\"mode\":\"{}\",\"selected_path\":\"{}\",\"capability_result\":\"{}\",\"payload_bytes\":{},\"chunk_payload_bytes\":{},\"chunks\":{},\"protocol_header_bytes\":{},\"total_wire_bytes\":{},\"preposted_receives\":{},\"credit_window_chunks\":{},\"credit_windows\":{},\"credit_stalls\":{},\"sender_retention_chunks\":{},\"receiver_bitmap_words\":{},\"nack_ranges\":{},\"selective_retransmits\":{},\"ack_packets\":{},\"mbufs_preallocated\":{},\"rings_preallocated\":{},\"direct_gpu_memory_claimed\":{},\"pinned_host_required\":{},\"fallback_decisions\":{},\"transport_events\":{},\"phase_handoff_syncs\":{},\"pageable_copies\":{},\"per_token_registrations\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.protocol_version,
            self.request_id,
            self.sequence_id,
            self.block_id,
            self.block_version,
            self.mode.as_str(),
            self.selected_path.as_str(),
            self.capability_result.as_str(),
            self.payload_bytes,
            self.chunk_payload_bytes,
            self.chunks,
            self.protocol_header_bytes,
            self.total_wire_bytes,
            self.preposted_receives,
            self.credit_window_chunks,
            self.credit_windows,
            self.credit_stalls,
            self.sender_retention_chunks,
            self.receiver_bitmap_words,
            self.nack_ranges,
            self.selective_retransmits,
            self.ack_packets,
            self.mbufs_preallocated,
            self.rings_preallocated,
            self.direct_gpu_memory_claimed,
            self.pinned_host_required,
            self.fallback_decisions,
            self.transport_events,
            self.phase_handoff_syncs,
            self.pageable_copies,
            self.per_token_registrations,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}
