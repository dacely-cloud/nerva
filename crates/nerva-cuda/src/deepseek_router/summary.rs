use crate::json::json_opt_str;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekRouterSummary {
    pub status: SmokeStatus,
    pub v3_num_experts: u32,
    pub v3_num_groups: u32,
    pub v3_top_k_groups: u32,
    pub v3_top_k: u32,
    pub v4_num_experts: u32,
    pub v4_top_k: u32,
    pub v4_hash_top_k: u32,
    pub v3_expert_ids: [u32; 2],
    pub v4_expert_ids: [u32; 2],
    pub v4_hash_expert_ids: [u32; 3],
    pub v3_weights: [f32; 2],
    pub v4_weights: [f32; 2],
    pub v4_hash_weights: [f32; 3],
    pub v3_output_hash: u64,
    pub v4_output_hash: u64,
    pub v4_hash_output_hash: u64,
    pub v3_mismatches: u64,
    pub v4_mismatches: u64,
    pub v4_hash_mismatches: u64,
    pub v3_max_abs_diff: f32,
    pub v4_max_abs_diff: f32,
    pub v4_hash_max_abs_diff: f32,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaDeepSeekRouterSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"v3_num_experts\":{},\"v3_num_groups\":{},\"v3_top_k_groups\":{},\"v3_top_k\":{},\"v4_num_experts\":{},\"v4_top_k\":{},\"v4_hash_top_k\":{},\"v3_expert_ids\":[{},{}],\"v4_expert_ids\":[{},{}],\"v4_hash_expert_ids\":[{},{},{}],\"v3_weights\":[{},{}],\"v4_weights\":[{},{}],\"v4_hash_weights\":[{},{},{}],\"v3_output_hash\":{},\"v4_output_hash\":{},\"v4_hash_output_hash\":{},\"v3_mismatches\":{},\"v4_mismatches\":{},\"v4_hash_mismatches\":{},\"v3_max_abs_diff\":{},\"v4_max_abs_diff\":{},\"v4_hash_max_abs_diff\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.v3_num_experts,
            self.v3_num_groups,
            self.v3_top_k_groups,
            self.v3_top_k,
            self.v4_num_experts,
            self.v4_top_k,
            self.v4_hash_top_k,
            self.v3_expert_ids[0],
            self.v3_expert_ids[1],
            self.v4_expert_ids[0],
            self.v4_expert_ids[1],
            self.v4_hash_expert_ids[0],
            self.v4_hash_expert_ids[1],
            self.v4_hash_expert_ids[2],
            self.v3_weights[0],
            self.v3_weights[1],
            self.v4_weights[0],
            self.v4_weights[1],
            self.v4_hash_weights[0],
            self.v4_hash_weights[1],
            self.v4_hash_weights[2],
            self.v3_output_hash,
            self.v4_output_hash,
            self.v4_hash_output_hash,
            self.v3_mismatches,
            self.v4_mismatches,
            self.v4_hash_mismatches,
            self.v3_max_abs_diff,
            self.v4_max_abs_diff,
            self.v4_hash_max_abs_diff,
            self.device_arena_bytes,
            self.pinned_host_bytes,
            self.d2h_bytes,
            self.kernel_launches,
            self.sync_calls,
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
    }

    pub(crate) fn unavailable(error: impl Into<String>) -> Self {
        Self::empty(SmokeStatus::Unavailable, error)
    }

    pub(crate) fn failed(error: impl Into<String>) -> Self {
        Self::empty(SmokeStatus::Failed, error)
    }

    fn empty(status: SmokeStatus, error: impl Into<String>) -> Self {
        Self {
            status,
            v3_num_experts: 8,
            v3_num_groups: 2,
            v3_top_k_groups: 1,
            v3_top_k: 2,
            v4_num_experts: 4,
            v4_top_k: 2,
            v4_hash_top_k: 3,
            v3_expert_ids: [0; 2],
            v4_expert_ids: [0; 2],
            v4_hash_expert_ids: [0; 3],
            v3_weights: [0.0; 2],
            v4_weights: [0.0; 2],
            v4_hash_weights: [0.0; 3],
            v3_output_hash: 0,
            v4_output_hash: 0,
            v4_hash_output_hash: 0,
            v3_mismatches: 0,
            v4_mismatches: 0,
            v4_hash_mismatches: 0,
            v3_max_abs_diff: 0.0,
            v4_max_abs_diff: 0.0,
            v4_hash_max_abs_diff: 0.0,
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            d2h_bytes: 0,
            kernel_launches: 0,
            sync_calls: 0,
            hot_path_allocations: 0,
            error: Some(error.into()),
        }
    }
}
