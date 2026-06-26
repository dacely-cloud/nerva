use crate::transport::json::json_opt_static_str;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MultiGpuNodeStatus {
    Ok,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiGpuNodeSummary {
    pub status: MultiGpuNodeStatus,
    pub gpu_count: u32,
    pub gpu_islands: u32,
    pub compute_gpu_count: u32,
    pub egress_gpu: i32,
    pub nic_near_egress: bool,
    pub local_vram_bytes_per_gpu: usize,
    pub aggregate_vram_bytes: usize,
    pub aggregate_vram_pool_claimed: bool,
    pub coherent_vram_allocation_claims: u32,
    pub max_single_allocation_bytes: usize,
    pub stage_layers: u32,
    pub stage_weight_bytes: usize,
    pub hot_weight_cache_bytes: usize,
    pub dram_weight_backing_bytes: usize,
    pub stage_kv_bytes: usize,
    pub kv_owner_count: u32,
    pub activation_bytes_per_boundary: usize,
    pub local_boundaries: u32,
    pub activation_only_boundaries: u32,
    pub activation_bytes_moved: usize,
    pub inter_gpu_weight_bytes: usize,
    pub all_reduce_bytes: usize,
    pub execution_decisions: u64,
    pub device_events: u64,
    pub copy_events: u64,
    pub phase_handoff_syncs: u64,
    pub pageable_copies: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl MultiGpuNodeSummary {
    pub fn passed(&self) -> bool {
        matches!(self.status, MultiGpuNodeStatus::Ok)
            && self.gpu_count >= 2
            && self.gpu_islands == self.gpu_count
            && self.compute_gpu_count == self.gpu_count
            && self.egress_gpu >= 0
            && self.nic_near_egress
            && self.aggregate_vram_bytes > self.local_vram_bytes_per_gpu
            && !self.aggregate_vram_pool_claimed
            && self.coherent_vram_allocation_claims == 0
            && self.max_single_allocation_bytes <= self.local_vram_bytes_per_gpu
            && self.stage_weight_bytes > self.hot_weight_cache_bytes
            && self.dram_weight_backing_bytes
                == self.stage_weight_bytes - self.hot_weight_cache_bytes
            && self.kv_owner_count == self.gpu_count
            && self.local_boundaries == self.gpu_count.saturating_sub(1)
            && self.activation_only_boundaries == self.local_boundaries
            && self.activation_bytes_moved
                == self
                    .activation_bytes_per_boundary
                    .saturating_mul(self.local_boundaries as usize)
            && self.inter_gpu_weight_bytes == 0
            && self.all_reduce_bytes == 0
            && self.execution_decisions == u64::from(self.gpu_count)
            && self.device_events == u64::from(self.gpu_count)
            && self.copy_events == u64::from(self.local_boundaries)
            && self.phase_handoff_syncs == u64::from(self.local_boundaries)
            && self.pageable_copies == 0
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            MultiGpuNodeStatus::Ok => "ok",
            MultiGpuNodeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"gpu_count\":{},\"gpu_islands\":{},\"compute_gpu_count\":{},\"egress_gpu\":{},\"nic_near_egress\":{},\"local_vram_bytes_per_gpu\":{},\"aggregate_vram_bytes\":{},\"aggregate_vram_pool_claimed\":{},\"coherent_vram_allocation_claims\":{},\"max_single_allocation_bytes\":{},\"stage_layers\":{},\"stage_weight_bytes\":{},\"hot_weight_cache_bytes\":{},\"dram_weight_backing_bytes\":{},\"stage_kv_bytes\":{},\"kv_owner_count\":{},\"activation_bytes_per_boundary\":{},\"local_boundaries\":{},\"activation_only_boundaries\":{},\"activation_bytes_moved\":{},\"inter_gpu_weight_bytes\":{},\"all_reduce_bytes\":{},\"execution_decisions\":{},\"device_events\":{},\"copy_events\":{},\"phase_handoff_syncs\":{},\"pageable_copies\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.gpu_count,
            self.gpu_islands,
            self.compute_gpu_count,
            self.egress_gpu,
            self.nic_near_egress,
            self.local_vram_bytes_per_gpu,
            self.aggregate_vram_bytes,
            self.aggregate_vram_pool_claimed,
            self.coherent_vram_allocation_claims,
            self.max_single_allocation_bytes,
            self.stage_layers,
            self.stage_weight_bytes,
            self.hot_weight_cache_bytes,
            self.dram_weight_backing_bytes,
            self.stage_kv_bytes,
            self.kv_owner_count,
            self.activation_bytes_per_boundary,
            self.local_boundaries,
            self.activation_only_boundaries,
            self.activation_bytes_moved,
            self.inter_gpu_weight_bytes,
            self.all_reduce_bytes,
            self.execution_decisions,
            self.device_events,
            self.copy_events,
            self.phase_handoff_syncs,
            self.pageable_copies,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}
