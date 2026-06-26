use crate::json::json_opt_str;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaTieredAttentionSummary {
    pub status: SmokeStatus,
    pub hidden: u32,
    pub heads: u32,
    pub blocks: u32,
    pub tokens: u32,
    pub output: [f32; 2],
    pub output_hash: u64,
    pub cpu_block_events: u64,
    pub device_block_events: u64,
    pub resident_kv_bytes: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaTieredAttentionSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"heads\":{},\"blocks\":{},\"tokens\":{},\"output\":[{},{}],\"output_hash\":{},\"cpu_block_events\":{},\"device_block_events\":{},\"resident_kv_bytes\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.hidden,
            self.heads,
            self.blocks,
            self.tokens,
            self.output[0],
            self.output[1],
            self.output_hash,
            self.cpu_block_events,
            self.device_block_events,
            self.resident_kv_bytes,
            self.device_arena_bytes,
            self.pinned_host_bytes,
            self.h2d_bytes,
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
            hidden: 2,
            heads: 1,
            blocks: 2,
            tokens: 4,
            output: [0.0, 0.0],
            output_hash: 0,
            cpu_block_events: 0,
            device_block_events: 0,
            resident_kv_bytes: 0,
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            h2d_bytes: 0,
            d2h_bytes: 0,
            kernel_launches: 0,
            sync_calls: 0,
            hot_path_allocations: 0,
            error: Some(error.into()),
        }
    }
}
