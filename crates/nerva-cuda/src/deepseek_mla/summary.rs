use crate::json::json_opt_str;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekMlaSummary {
    pub status: SmokeStatus,
    pub heads: u32,
    pub tokens: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub softmax_scale: f32,
    pub output: [f32; 4],
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

impl CudaDeepSeekMlaSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"heads\":{},\"tokens\":{},\"kv_lora_rank\":{},\"qk_nope_head_dim\":{},\"qk_rope_head_dim\":{},\"v_head_dim\":{},\"softmax_scale\":{},\"output\":[{},{},{},{}],\"output_hash\":{},\"mismatches\":{},\"max_abs_diff\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.heads,
            self.tokens,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.softmax_scale,
            self.output[0],
            self.output[1],
            self.output[2],
            self.output[3],
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
            heads: 2,
            tokens: 3,
            kv_lora_rank: 3,
            qk_nope_head_dim: 2,
            qk_rope_head_dim: 1,
            v_head_dim: 2,
            softmax_scale: 0.7,
            output: [0.0; 4],
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
