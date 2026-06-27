use crate::json::json_opt_str;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaHfSamplerSummary {
    pub status: SmokeStatus,
    pub dtype: u32,
    pub hidden: u32,
    pub vocab_size: u32,
    pub token_index: u64,
    pub token: u32,
    pub slot_version: u64,
    pub completion: u32,
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

impl CudaHfSamplerSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"status\":\"{}\",\"dtype\":{},\"hidden\":{},\"vocab_size\":{},\"token_index\":{},\"token\":{},\"slot_version\":{},\"completion\":{},\"output_hash\":{},\"resident_weight_bytes\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status_str(&self.status),
            self.dtype,
            self.hidden,
            self.vocab_size,
            self.token_index,
            self.token,
            self.slot_version,
            self.completion,
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
}

pub(crate) fn empty_summary(
    status: SmokeStatus,
    dtype: u32,
    hidden: usize,
    vocab_size: usize,
    token_index: u64,
    error: String,
) -> CudaHfSamplerSummary {
    CudaHfSamplerSummary {
        status,
        dtype,
        hidden: hidden as u32,
        vocab_size: vocab_size as u32,
        token_index,
        token: 0,
        slot_version: 0,
        completion: 0,
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

pub(crate) fn status_str(status: &SmokeStatus) -> &'static str {
    match status {
        SmokeStatus::Ok => "ok",
        SmokeStatus::Unavailable => "unavailable",
        SmokeStatus::Failed => "failed",
    }
}
