use crate::transport::json::json_opt_static_str;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportContractStatus {
    Ok,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportContractSummary {
    pub status: TransportContractStatus,
    pub backend: &'static str,
    pub registrations: u64,
    pub registered_entries: u64,
    pub preposted_receives: u64,
    pub sends: u64,
    pub completions: u64,
    pub bytes_completed: usize,
    pub unposted_send_rejections: u64,
    pub stale_version_rejections: u64,
    pub descriptor_rejections: u64,
    pub pre_visibility_consume_rejections: u64,
    pub visibility_fences: u64,
    pub visible_consumes: u64,
    pub per_transfer_registrations: u64,
    pub transport_events: u64,
    pub phase_handoff_syncs: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl TransportContractSummary {
    pub fn passed(&self) -> bool {
        matches!(self.status, TransportContractStatus::Ok)
            && self.registrations == 2
            && self.registered_entries == 2
            && self.preposted_receives == 0
            && self.sends == 1
            && self.completions == 1
            && self.bytes_completed > 0
            && self.unposted_send_rejections == 1
            && self.stale_version_rejections == 1
            && self.descriptor_rejections == 1
            && self.pre_visibility_consume_rejections == 1
            && self.visibility_fences == 1
            && self.visible_consumes == 1
            && self.per_transfer_registrations == 0
            && self.transport_events == 1
            && self.phase_handoff_syncs == 1
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            TransportContractStatus::Ok => "ok",
            TransportContractStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"backend\":\"{}\",\"registrations\":{},\"registered_entries\":{},\"preposted_receives\":{},\"sends\":{},\"completions\":{},\"bytes_completed\":{},\"unposted_send_rejections\":{},\"stale_version_rejections\":{},\"descriptor_rejections\":{},\"pre_visibility_consume_rejections\":{},\"visibility_fences\":{},\"visible_consumes\":{},\"per_transfer_registrations\":{},\"transport_events\":{},\"phase_handoff_syncs\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.backend,
            self.registrations,
            self.registered_entries,
            self.preposted_receives,
            self.sends,
            self.completions,
            self.bytes_completed,
            self.unposted_send_rejections,
            self.stale_version_rejections,
            self.descriptor_rejections,
            self.pre_visibility_consume_rejections,
            self.visibility_fences,
            self.visible_consumes,
            self.per_transfer_registrations,
            self.transport_events,
            self.phase_handoff_syncs,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}
