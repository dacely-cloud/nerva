#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TcpControlStatus {
    Ok,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TcpControlSummary {
    pub status: TcpControlStatus,
    pub backend: &'static str,
    pub protocol_version: u16,
    pub request_id: u64,
    pub sequence_id: u64,
    pub control_bytes_sent: usize,
    pub control_bytes_received: usize,
    pub tensor_payload_bytes: usize,
    pub total_wire_bytes: usize,
    pub connection_count: u64,
    pub control_messages: u64,
    pub completion_latency_ns: u64,
    pub effective_control_bandwidth_bps: u64,
    pub runtime_timestamp_events: u64,
    pub transport_events: u64,
    pub control_plane_only: bool,
    pub debug_only: bool,
    pub production_tensor_data_plane: bool,
    pub pageable_copies: u64,
    pub per_token_registrations: u64,
    pub hot_path_allocations: u64,
}

impl TcpControlSummary {
    pub fn passed(&self) -> bool {
        matches!(self.status, TcpControlStatus::Ok)
            && self.backend == "tcp_control_only"
            && self.protocol_version > 0
            && self.request_id > 0
            && self.sequence_id > 0
            && self.control_bytes_sent > 0
            && self.control_bytes_received > 0
            && self.tensor_payload_bytes == 0
            && self.total_wire_bytes == self.control_bytes_sent + self.control_bytes_received
            && self.connection_count == 1
            && self.control_messages == 2
            && self.completion_latency_ns > 0
            && self.effective_control_bandwidth_bps > 0
            && self.runtime_timestamp_events == 1
            && self.transport_events == 1
            && self.control_plane_only
            && self.debug_only
            && !self.production_tensor_data_plane
            && self.pageable_copies == 0
            && self.per_token_registrations == 0
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            TcpControlStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"backend\":\"{}\",\"protocol_version\":{},\"request_id\":{},\"sequence_id\":{},\"control_bytes_sent\":{},\"control_bytes_received\":{},\"tensor_payload_bytes\":{},\"total_wire_bytes\":{},\"connection_count\":{},\"control_messages\":{},\"completion_latency_ns\":{},\"effective_control_bandwidth_bps\":{},\"runtime_timestamp_events\":{},\"transport_events\":{},\"control_plane_only\":{},\"debug_only\":{},\"production_tensor_data_plane\":{},\"pageable_copies\":{},\"per_token_registrations\":{},\"hot_path_allocations\":{}}}",
            status,
            self.backend,
            self.protocol_version,
            self.request_id,
            self.sequence_id,
            self.control_bytes_sent,
            self.control_bytes_received,
            self.tensor_payload_bytes,
            self.total_wire_bytes,
            self.connection_count,
            self.control_messages,
            self.completion_latency_ns,
            self.effective_control_bandwidth_bps,
            self.runtime_timestamp_events,
            self.transport_events,
            self.control_plane_only,
            self.debug_only,
            self.production_tensor_data_plane,
            self.pageable_copies,
            self.per_token_registrations,
            self.hot_path_allocations,
        )
    }
}
