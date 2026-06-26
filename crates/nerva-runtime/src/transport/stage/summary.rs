use crate::transport::json::json_opt_static_str;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StagePipelineStatus {
    Ok,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagePipelineSummary {
    pub status: StagePipelineStatus,
    pub stages: u32,
    pub layers: u32,
    pub boundaries: u32,
    pub activation_bytes_per_boundary: usize,
    pub total_activation_tx_bytes: usize,
    pub stage_local_weight_bytes: usize,
    pub stage_local_kv_bytes: usize,
    pub inter_stage_weight_bytes: usize,
    pub all_reduce_bytes: usize,
    pub activation_only_boundaries: u32,
    pub gpu_direct_boundaries: u32,
    pub host_staged_boundaries: u32,
    pub cpu_produced_boundaries: u32,
    pub mapped_pinned_boundaries: u32,
    pub fallback_decisions: u64,
    pub transport_events: u64,
    pub copy_events: u64,
    pub phase_handoff_syncs: u64,
    pub explicit_copy_bytes: usize,
    pub nic_tx_bytes: usize,
    pub nic_rx_bytes: usize,
    pub pageable_copies: u64,
    pub per_token_registrations: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl StagePipelineSummary {
    pub fn passed(&self) -> bool {
        matches!(self.status, StagePipelineStatus::Ok)
            && self.stages >= 2
            && self.boundaries == self.stages.saturating_sub(1)
            && self.activation_only_boundaries == self.boundaries
            && self.total_activation_tx_bytes > 0
            && self.stage_local_weight_bytes > 0
            && self.stage_local_kv_bytes > 0
            && self.inter_stage_weight_bytes == 0
            && self.all_reduce_bytes == 0
            && self.transport_events == u64::from(self.boundaries)
            && self.phase_handoff_syncs == u64::from(self.boundaries)
            && self.pageable_copies == 0
            && self.per_token_registrations == 0
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            StagePipelineStatus::Ok => "ok",
            StagePipelineStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"stages\":{},\"layers\":{},\"boundaries\":{},\"activation_bytes_per_boundary\":{},\"total_activation_tx_bytes\":{},\"stage_local_weight_bytes\":{},\"stage_local_kv_bytes\":{},\"inter_stage_weight_bytes\":{},\"all_reduce_bytes\":{},\"activation_only_boundaries\":{},\"gpu_direct_boundaries\":{},\"host_staged_boundaries\":{},\"cpu_produced_boundaries\":{},\"mapped_pinned_boundaries\":{},\"fallback_decisions\":{},\"transport_events\":{},\"copy_events\":{},\"phase_handoff_syncs\":{},\"explicit_copy_bytes\":{},\"nic_tx_bytes\":{},\"nic_rx_bytes\":{},\"pageable_copies\":{},\"per_token_registrations\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.stages,
            self.layers,
            self.boundaries,
            self.activation_bytes_per_boundary,
            self.total_activation_tx_bytes,
            self.stage_local_weight_bytes,
            self.stage_local_kv_bytes,
            self.inter_stage_weight_bytes,
            self.all_reduce_bytes,
            self.activation_only_boundaries,
            self.gpu_direct_boundaries,
            self.host_staged_boundaries,
            self.cpu_produced_boundaries,
            self.mapped_pinned_boundaries,
            self.fallback_decisions,
            self.transport_events,
            self.copy_events,
            self.phase_handoff_syncs,
            self.explicit_copy_bytes,
            self.nic_tx_bytes,
            self.nic_rx_bytes,
            self.pageable_copies,
            self.per_token_registrations,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}
