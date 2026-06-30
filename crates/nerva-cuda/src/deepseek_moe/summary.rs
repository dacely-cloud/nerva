use crate::json::json_opt_str;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekMoeSummary {
    pub status: SmokeStatus,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub swiglu_limit: f32,
    pub expert_ids: [u32; 2],
    pub expert_weights: [f32; 2],
    pub output: [f32; 3],
    pub output_hash: u64,
    pub mismatches: u64,
    pub max_abs_diff: f32,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaDeepSeekMoeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden_size\":{},\"intermediate_size\":{},\"num_experts\":{},\"top_k\":{},\"swiglu_limit\":{},\"expert_ids\":[{},{}],\"expert_weights\":[{},{}],\"output\":[{},{},{}],\"output_hash\":{},\"mismatches\":{},\"max_abs_diff\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.hidden_size,
            self.intermediate_size,
            self.num_experts,
            self.top_k,
            self.swiglu_limit,
            self.expert_ids[0],
            self.expert_ids[1],
            self.expert_weights[0],
            self.expert_weights[1],
            self.output[0],
            self.output[1],
            self.output[2],
            self.output_hash,
            self.mismatches,
            self.max_abs_diff,
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
            hidden_size: 3,
            intermediate_size: 2,
            num_experts: 2,
            top_k: 2,
            swiglu_limit: 1.0,
            expert_ids: [0; 2],
            expert_weights: [0.0; 2],
            output: [0.0; 3],
            output_hash: 0,
            mismatches: 0,
            max_abs_diff: 0.0,
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
