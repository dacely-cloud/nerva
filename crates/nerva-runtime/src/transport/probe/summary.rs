use crate::transport::json::json_opt_static_str;
use crate::transport::probe::status::TransportPathProbeStatus;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportPathProbeSummary {
    pub status: TransportPathProbeStatus,
    pub requests: u64,
    pub decode_requests: u64,
    pub prefill_requests: u64,
    pub gpu_direct_paths: u64,
    pub pinned_host_paths: u64,
    pub cpu_produced_paths: u64,
    pub mapped_pinned_paths: u64,
    pub transport_events: u64,
    pub copy_events: u64,
    pub sync_events: u64,
    pub phase_handoff_syncs: u64,
    pub fallback_decisions: u64,
    pub nic_tx_bytes: usize,
    pub nic_rx_bytes: usize,
    pub explicit_copy_bytes: usize,
    pub pageable_copies: u64,
    pub per_token_registrations: u64,
    pub estimated_events: u64,
    pub estimated_latency_ns: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl TransportPathProbeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            TransportPathProbeStatus::Ok => "ok",
            TransportPathProbeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"requests\":{},\"decode_requests\":{},\"prefill_requests\":{},\"gpu_direct_paths\":{},\"pinned_host_paths\":{},\"cpu_produced_paths\":{},\"mapped_pinned_paths\":{},\"transport_events\":{},\"copy_events\":{},\"sync_events\":{},\"phase_handoff_syncs\":{},\"fallback_decisions\":{},\"nic_tx_bytes\":{},\"nic_rx_bytes\":{},\"explicit_copy_bytes\":{},\"pageable_copies\":{},\"per_token_registrations\":{},\"estimated_events\":{},\"estimated_latency_ns\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.requests,
            self.decode_requests,
            self.prefill_requests,
            self.gpu_direct_paths,
            self.pinned_host_paths,
            self.cpu_produced_paths,
            self.mapped_pinned_paths,
            self.transport_events,
            self.copy_events,
            self.sync_events,
            self.phase_handoff_syncs,
            self.fallback_decisions,
            self.nic_tx_bytes,
            self.nic_rx_bytes,
            self.explicit_copy_bytes,
            self.pageable_copies,
            self.per_token_registrations,
            self.estimated_events,
            self.estimated_latency_ns,
            self.total_latency_ns,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}
