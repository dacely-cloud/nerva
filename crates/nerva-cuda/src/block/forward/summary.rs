use crate::block::ffi::NervaCudaBlockForwardResult;
use crate::block::forward::request::CudaBlockForwardRequest;
use crate::json::json_opt_str;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaBlockForwardSummary {
    pub status: SmokeStatus,
    pub dtype: u32,
    pub hidden: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub output: Vec<u16>,
    pub output_hash: u64,
    pub resident_weight_bytes: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaBlockForwardSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"status\":\"{}\",\"dtype\":{},\"hidden\":{},\"heads\":{},\"kv_heads\":{},\"head_dim\":{},\"intermediate\":{},\"output_hash\":{},\"resident_weight_bytes\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status_str(&self.status),
            self.dtype,
            self.hidden,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.intermediate,
            self.output_hash,
            self.resident_weight_bytes,
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

    pub(crate) fn failed_from_request(
        request: &CudaBlockForwardRequest<'_>,
        output: Vec<u16>,
        error: String,
    ) -> Self {
        Self {
            status: SmokeStatus::Failed,
            dtype: request.dtype,
            hidden: request.hidden as u32,
            heads: request.heads as u32,
            kv_heads: request.kv_heads as u32,
            head_dim: request.head_dim as u32,
            intermediate: request.intermediate as u32,
            output,
            output_hash: 0,
            resident_weight_bytes: 0,
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            h2d_bytes: 0,
            d2h_bytes: 0,
            kernel_launches: 0,
            sync_calls: 0,
            hot_path_allocations: 0,
            error: Some(error),
        }
    }

    pub(crate) fn from_ffi(
        status: SmokeStatus,
        output: Vec<u16>,
        out: NervaCudaBlockForwardResult,
        error: Option<String>,
    ) -> Self {
        Self {
            status,
            dtype: out.dtype,
            hidden: out.hidden,
            heads: out.heads,
            kv_heads: out.kv_heads,
            head_dim: out.head_dim,
            intermediate: out.intermediate,
            output,
            output_hash: out.output_hash,
            resident_weight_bytes: out.resident_weight_bytes,
            device_arena_bytes: out.device_arena_bytes,
            pinned_host_bytes: out.pinned_host_bytes,
            h2d_bytes: out.h2d_bytes,
            d2h_bytes: out.d2h_bytes,
            kernel_launches: out.kernel_launches,
            sync_calls: out.sync_calls,
            hot_path_allocations: out.hot_path_allocations,
            error,
        }
    }
}

fn status_str(status: &SmokeStatus) -> &'static str {
    match status {
        SmokeStatus::Ok => "ok",
        SmokeStatus::Unavailable => "unavailable",
        SmokeStatus::Failed => "failed",
    }
}
