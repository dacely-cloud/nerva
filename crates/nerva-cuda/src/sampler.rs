//! CUDA-backed greedy sampler smoke.

use std::os::raw::c_int;

use crate::smoke::{SmokeStatus, escape_json};

const CUDA_ERROR_NO_DEVICE: i32 = 100;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct NervaCudaGreedySamplerResult {
    status: i32,
    cuda_error: i32,
    device_count: i32,
    vocab_size: u32,
    token_index: u64,
    token: u32,
    slot_version: u64,
    completion: u32,
    device_arena_bytes: u64,
    pinned_host_bytes: u64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    kernel_launches: u64,
    sync_calls: u64,
    hot_path_allocations: u64,
}

unsafe extern "C" {
    fn nerva_cuda_greedy_sampler_smoke(out: *mut NervaCudaGreedySamplerResult) -> c_int;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaGreedySamplerSummary {
    pub status: SmokeStatus,
    pub vocab_size: u32,
    pub token_index: u64,
    pub token: u32,
    pub slot_version: u64,
    pub completion: u32,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaGreedySamplerSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"vocab_size\":{},\"token_index\":{},\"token\":{},\"slot_version\":{},\"completion\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.vocab_size,
            self.token_index,
            self.token,
            self.slot_version,
            self.completion,
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

    fn unavailable(error: impl Into<String>) -> Self {
        Self {
            status: SmokeStatus::Unavailable,
            vocab_size: 4,
            token_index: 0,
            token: 0,
            slot_version: 0,
            completion: 0,
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

    fn failed(error: impl Into<String>) -> Self {
        Self {
            status: SmokeStatus::Failed,
            vocab_size: 4,
            token_index: 0,
            token: 0,
            slot_version: 0,
            completion: 0,
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

pub fn greedy_sampler_smoke() -> CudaGreedySamplerSummary {
    let mut out = NervaCudaGreedySamplerResult::default();
    let return_code = unsafe { nerva_cuda_greedy_sampler_smoke(&mut out) };

    if return_code == 0
        && out.status == 0
        && out.vocab_size == 4
        && out.token == 2
        && out.slot_version == 1
        && out.completion == 1
        && out.kernel_launches == 1
        && out.hot_path_allocations == 0
    {
        return CudaGreedySamplerSummary {
            status: SmokeStatus::Ok,
            vocab_size: out.vocab_size,
            token_index: out.token_index,
            token: out.token,
            slot_version: out.slot_version,
            completion: out.completion,
            device_arena_bytes: out.device_arena_bytes,
            pinned_host_bytes: out.pinned_host_bytes,
            h2d_bytes: out.h2d_bytes,
            d2h_bytes: out.d2h_bytes,
            kernel_launches: out.kernel_launches,
            sync_calls: out.sync_calls,
            hot_path_allocations: out.hot_path_allocations,
            error: None,
        };
    }

    let reason = format!(
        "CUDA greedy sampler smoke failed: return_code={} status={} cuda_error={} device_count={} vocab_size={} token={} slot_version={} completion={} kernel_launches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.vocab_size,
        out.token,
        out.slot_version,
        out.completion,
        out.kernel_launches,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaGreedySamplerSummary::unavailable(reason)
    } else {
        CudaGreedySamplerSummary::failed(reason)
    }
}

fn json_opt_str(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("\"{}\"", escape_json(value)),
        None => "null".to_string(),
    }
}
