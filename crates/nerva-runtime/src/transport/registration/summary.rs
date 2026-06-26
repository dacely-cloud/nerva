use crate::capabilities::snapshot::CapabilityState;
use crate::transport::json::json_opt_static_str;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportRegistrationStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportRegistrationSummary {
    pub status: TransportRegistrationStatus,
    pub cache_capacity: u64,
    pub registered_entries: u64,
    pub bootstrap_registrations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub stale_address_rejections: u64,
    pub stale_version_rejections: u64,
    pub hot_path_registration_attempts: u64,
    pub hot_path_registration_rejections: u64,
    pub per_token_registrations: u64,
    pub pinned_host_registrations: u64,
    pub gpu_direct_registrations: u64,
    pub gpu_direct_rdma_capability: CapabilityState,
    pub gpu_direct_registration_skips: u64,
    pub false_gpu_direct_registrations: u64,
    pub transport_events: u64,
    pub sync_events: u64,
    pub phase_handoff_syncs: u64,
    pub registration_cache_hit_rate_per_mille: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl TransportRegistrationSummary {
    pub fn passed(self) -> bool {
        let direct_verified =
            self.gpu_direct_rdma_capability == CapabilityState::SupportedAndVerified;
        let direct_policy_ok = if direct_verified {
            self.gpu_direct_registrations > 0 && self.gpu_direct_registration_skips == 0
        } else {
            self.gpu_direct_registrations == 0 && self.gpu_direct_registration_skips > 0
        };

        matches!(self.status, TransportRegistrationStatus::Ok)
            && self.bootstrap_registrations > 0
            && self.registered_entries == self.bootstrap_registrations
            && self.cache_hits > 0
            && self.cache_misses > 0
            && self.stale_address_rejections > 0
            && self.hot_path_registration_attempts == self.hot_path_registration_rejections
            && self.per_token_registrations == 0
            && self.false_gpu_direct_registrations == 0
            && direct_policy_ok
            && self.hot_path_allocations == 0
            && self.registration_cache_hit_rate_per_mille > 0
    }

    pub fn to_json(self) -> String {
        let status = match self.status {
            TransportRegistrationStatus::Ok => "ok",
            TransportRegistrationStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"cache_capacity\":{},\"registered_entries\":{},\"bootstrap_registrations\":{},\"cache_hits\":{},\"cache_misses\":{},\"stale_address_rejections\":{},\"stale_version_rejections\":{},\"hot_path_registration_attempts\":{},\"hot_path_registration_rejections\":{},\"per_token_registrations\":{},\"pinned_host_registrations\":{},\"gpu_direct_registrations\":{},\"gpu_direct_rdma_capability\":\"{}\",\"gpu_direct_registration_skips\":{},\"false_gpu_direct_registrations\":{},\"transport_events\":{},\"sync_events\":{},\"phase_handoff_syncs\":{},\"registration_cache_hit_rate_per_mille\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.cache_capacity,
            self.registered_entries,
            self.bootstrap_registrations,
            self.cache_hits,
            self.cache_misses,
            self.stale_address_rejections,
            self.stale_version_rejections,
            self.hot_path_registration_attempts,
            self.hot_path_registration_rejections,
            self.per_token_registrations,
            self.pinned_host_registrations,
            self.gpu_direct_registrations,
            self.gpu_direct_rdma_capability.as_str(),
            self.gpu_direct_registration_skips,
            self.false_gpu_direct_registrations,
            self.transport_events,
            self.sync_events,
            self.phase_handoff_syncs,
            self.registration_cache_hit_rate_per_mille,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}
