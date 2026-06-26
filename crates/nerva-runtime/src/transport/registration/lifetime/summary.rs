use crate::transport::json::json_opt_static_str;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportRegistrationLifecycleStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportRegistrationLifecycleSummary {
    pub status: TransportRegistrationLifecycleStatus,
    pub bootstrap_registrations: u64,
    pub registered_before_revoke: u64,
    pub explicit_key_revocations: u64,
    pub block_lifetime_revocations: u64,
    pub destroy_revocations: u64,
    pub total_revocations: u64,
    pub final_registered_entries: u64,
    pub lookup_hits_before_revoke: u64,
    pub post_revoke_misses: u64,
    pub safe_move_post_revoke_misses: u64,
    pub stale_mapping_reuse_rejections: u64,
    pub revocation_syncs: u64,
    pub phase_handoff_syncs: u64,
    pub transport_events: u64,
    pub stale_handle_reuse_prevented: bool,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl TransportRegistrationLifecycleSummary {
    pub fn passed(self) -> bool {
        matches!(self.status, TransportRegistrationLifecycleStatus::Ok)
            && self.bootstrap_registrations == 4
            && self.registered_before_revoke == 4
            && self.explicit_key_revocations > 0
            && self.block_lifetime_revocations > 0
            && self.destroy_revocations > 0
            && self.total_revocations == self.bootstrap_registrations
            && self.final_registered_entries == 0
            && self.lookup_hits_before_revoke > 0
            && self.post_revoke_misses > 0
            && self.safe_move_post_revoke_misses > 0
            && self.stale_mapping_reuse_rejections > 0
            && self.revocation_syncs == self.total_revocations
            && self.phase_handoff_syncs
                == self.total_revocations + self.stale_mapping_reuse_rejections
            && self.transport_events > 0
            && self.stale_handle_reuse_prevented
            && self.hot_path_allocations == 0
    }

    pub fn to_json(self) -> String {
        let status = match self.status {
            TransportRegistrationLifecycleStatus::Ok => "ok",
            TransportRegistrationLifecycleStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"bootstrap_registrations\":{},\"registered_before_revoke\":{},\"explicit_key_revocations\":{},\"block_lifetime_revocations\":{},\"destroy_revocations\":{},\"total_revocations\":{},\"final_registered_entries\":{},\"lookup_hits_before_revoke\":{},\"post_revoke_misses\":{},\"safe_move_post_revoke_misses\":{},\"stale_mapping_reuse_rejections\":{},\"revocation_syncs\":{},\"phase_handoff_syncs\":{},\"transport_events\":{},\"stale_handle_reuse_prevented\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.bootstrap_registrations,
            self.registered_before_revoke,
            self.explicit_key_revocations,
            self.block_lifetime_revocations,
            self.destroy_revocations,
            self.total_revocations,
            self.final_registered_entries,
            self.lookup_hits_before_revoke,
            self.post_revoke_misses,
            self.safe_move_post_revoke_misses,
            self.stale_mapping_reuse_rejections,
            self.revocation_syncs,
            self.phase_handoff_syncs,
            self.transport_events,
            self.stale_handle_reuse_prevented,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}
