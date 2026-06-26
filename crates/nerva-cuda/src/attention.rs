//! CUDA-backed exact tiered attention smoke.

use std::os::raw::c_int;

use crate::smoke::{SmokeStatus, escape_json};

const CUDA_ERROR_NO_DEVICE: i32 = 100;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct NervaCudaTieredAttentionResult {
    status: i32,
    cuda_error: i32,
    device_count: i32,
    hidden: u32,
    heads: u32,
    blocks: u32,
    tokens: u32,
    output: [f32; 2],
    output_hash: u64,
    cpu_block_events: u64,
    device_block_events: u64,
    resident_kv_bytes: u64,
    device_arena_bytes: u64,
    pinned_host_bytes: u64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    kernel_launches: u64,
    sync_calls: u64,
    hot_path_allocations: u64,
}

unsafe extern "C" {
    fn nerva_cuda_tiered_attention_smoke(out: *mut NervaCudaTieredAttentionResult) -> c_int;
}

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

    fn unavailable(error: impl Into<String>) -> Self {
        Self::empty(SmokeStatus::Unavailable, error)
    }

    fn failed(error: impl Into<String>) -> Self {
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

pub fn tiered_attention_smoke() -> CudaTieredAttentionSummary {
    let mut out = NervaCudaTieredAttentionResult::default();
    let return_code = unsafe { nerva_cuda_tiered_attention_smoke(&mut out) };

    if return_code == 0
        && out.status == 0
        && out.hidden == 2
        && out.heads == 1
        && out.blocks == 2
        && out.tokens == 4
        && out.output_hash != 0
        && out.cpu_block_events == 1
        && out.device_block_events == 1
        && out.resident_kv_bytes > 0
        && out.h2d_bytes >= out.resident_kv_bytes
        && out.d2h_bytes > 0
        && out.kernel_launches == 1
        && out.hot_path_allocations == 0
        && out.output.iter().all(|value| value.is_finite())
    {
        return CudaTieredAttentionSummary {
            status: SmokeStatus::Ok,
            hidden: out.hidden,
            heads: out.heads,
            blocks: out.blocks,
            tokens: out.tokens,
            output: out.output,
            output_hash: out.output_hash,
            cpu_block_events: out.cpu_block_events,
            device_block_events: out.device_block_events,
            resident_kv_bytes: out.resident_kv_bytes,
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
        "CUDA tiered attention smoke failed: return_code={} status={} cuda_error={} device_count={} hidden={} heads={} blocks={} tokens={} output_hash={} cpu_events={} device_events={} kernel_launches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.hidden,
        out.heads,
        out.blocks,
        out.tokens,
        out.output_hash,
        out.cpu_block_events,
        out.device_block_events,
        out.kernel_launches,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaTieredAttentionSummary::unavailable(reason)
    } else {
        CudaTieredAttentionSummary::failed(reason)
    }
}

fn json_opt_str(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("\"{}\"", escape_json(value)),
        None => "null".to_string(),
    }
}
