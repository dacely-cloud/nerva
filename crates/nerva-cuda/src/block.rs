//! CUDA-backed exact tiny Transformer-block component smoke.

use std::os::raw::c_int;

use crate::smoke::{SmokeStatus, escape_json};

const CUDA_ERROR_NO_DEVICE: i32 = 100;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct NervaCudaTinyBlockResult {
    status: i32,
    cuda_error: i32,
    device_count: i32,
    hidden: u32,
    intermediate: u32,
    output: [u16; 2],
    output_hash: u64,
    device_arena_bytes: u64,
    pinned_host_bytes: u64,
    kernel_launches: u64,
    sync_calls: u64,
    d2h_bytes: u64,
    hot_path_allocations: u64,
}

unsafe extern "C" {
    fn nerva_cuda_tiny_block_smoke(out: *mut NervaCudaTinyBlockResult) -> c_int;
    fn nerva_cuda_loaded_tiny_block_smoke(out: *mut NervaCudaLoadedTinyBlockResult) -> c_int;
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct NervaCudaLoadedTinyBlockResult {
    status: i32,
    cuda_error: i32,
    device_count: i32,
    hidden: u32,
    intermediate: u32,
    output: [u16; 2],
    output_hash: u64,
    resident_weight_bytes: u64,
    device_arena_bytes: u64,
    pinned_host_bytes: u64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    kernel_launches: u64,
    sync_calls: u64,
    hot_path_allocations: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaTinyBlockSummary {
    pub status: SmokeStatus,
    pub hidden: u32,
    pub intermediate: u32,
    pub output: [u16; 2],
    pub output_hash: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub d2h_bytes: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaLoadedTinyBlockSummary {
    pub status: SmokeStatus,
    pub hidden: u32,
    pub intermediate: u32,
    pub output: [u16; 2],
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

impl CudaTinyBlockSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"intermediate\":{},\"output_bits\":[{},{}],\"output_hash\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"D2H_bytes\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.hidden,
            self.intermediate,
            self.output[0],
            self.output[1],
            self.output_hash,
            self.device_arena_bytes,
            self.pinned_host_bytes,
            self.kernel_launches,
            self.sync_calls,
            self.d2h_bytes,
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
    }

    fn unavailable(error: impl Into<String>) -> Self {
        Self {
            status: SmokeStatus::Unavailable,
            hidden: 2,
            intermediate: 2,
            output: [0, 0],
            output_hash: 0,
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            kernel_launches: 0,
            sync_calls: 0,
            d2h_bytes: 0,
            hot_path_allocations: 0,
            error: Some(error.into()),
        }
    }

    fn failed(error: impl Into<String>) -> Self {
        Self {
            status: SmokeStatus::Failed,
            hidden: 2,
            intermediate: 2,
            output: [0, 0],
            output_hash: 0,
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            kernel_launches: 0,
            sync_calls: 0,
            d2h_bytes: 0,
            hot_path_allocations: 0,
            error: Some(error.into()),
        }
    }
}

impl CudaLoadedTinyBlockSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"intermediate\":{},\"output_bits\":[{},{}],\"output_hash\":{},\"resident_weight_bytes\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.hidden,
            self.intermediate,
            self.output[0],
            self.output[1],
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

    fn unavailable(error: impl Into<String>) -> Self {
        Self {
            status: SmokeStatus::Unavailable,
            hidden: 2,
            intermediate: 2,
            output: [0, 0],
            output_hash: 0,
            resident_weight_bytes: 0,
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
            hidden: 2,
            intermediate: 2,
            output: [0, 0],
            output_hash: 0,
            resident_weight_bytes: 0,
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

pub fn tiny_block_smoke() -> CudaTinyBlockSummary {
    let mut out = NervaCudaTinyBlockResult::default();
    let return_code = unsafe { nerva_cuda_tiny_block_smoke(&mut out) };

    if return_code == 0
        && out.status == 0
        && out.hidden == 2
        && out.intermediate == 2
        && out.output_hash != 0
        && out.kernel_launches == 1
        && out.hot_path_allocations == 0
    {
        return CudaTinyBlockSummary {
            status: SmokeStatus::Ok,
            hidden: out.hidden,
            intermediate: out.intermediate,
            output: out.output,
            output_hash: out.output_hash,
            device_arena_bytes: out.device_arena_bytes,
            pinned_host_bytes: out.pinned_host_bytes,
            kernel_launches: out.kernel_launches,
            sync_calls: out.sync_calls,
            d2h_bytes: out.d2h_bytes,
            hot_path_allocations: out.hot_path_allocations,
            error: None,
        };
    }

    let reason = format!(
        "CUDA tiny block smoke failed: return_code={} status={} cuda_error={} device_count={} hidden={} intermediate={} output_hash={} kernel_launches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.hidden,
        out.intermediate,
        out.output_hash,
        out.kernel_launches,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaTinyBlockSummary::unavailable(reason)
    } else {
        CudaTinyBlockSummary::failed(reason)
    }
}

pub fn loaded_tiny_block_smoke() -> CudaLoadedTinyBlockSummary {
    let mut out = NervaCudaLoadedTinyBlockResult::default();
    let return_code = unsafe { nerva_cuda_loaded_tiny_block_smoke(&mut out) };

    if return_code == 0
        && out.status == 0
        && out.hidden == 2
        && out.intermediate == 2
        && out.output_hash != 0
        && out.resident_weight_bytes > 0
        && out.h2d_bytes >= out.resident_weight_bytes
        && out.kernel_launches == 1
        && out.hot_path_allocations == 0
    {
        return CudaLoadedTinyBlockSummary {
            status: SmokeStatus::Ok,
            hidden: out.hidden,
            intermediate: out.intermediate,
            output: out.output,
            output_hash: out.output_hash,
            resident_weight_bytes: out.resident_weight_bytes,
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
        "CUDA loaded tiny block smoke failed: return_code={} status={} cuda_error={} device_count={} hidden={} intermediate={} output_hash={} resident_weight_bytes={} h2d_bytes={} kernel_launches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.hidden,
        out.intermediate,
        out.output_hash,
        out.resident_weight_bytes,
        out.h2d_bytes,
        out.kernel_launches,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaLoadedTinyBlockSummary::unavailable(reason)
    } else {
        CudaLoadedTinyBlockSummary::failed(reason)
    }
}

fn json_opt_str(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("\"{}\"", escape_json(value)),
        None => "null".to_string(),
    }
}
