use crate::decode::hf_sequence::session::ffi::NervaCudaHfDecodeSequenceSessionCreateResult;
use crate::json::{json_opt_bool, json_opt_str, json_opt_usize};
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaHfDecodeSequenceSessionCreateSummary {
    pub status: SmokeStatus,
    pub dtype: u32,
    pub hidden: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub vocab_size: u32,
    pub layer_count: u32,
    pub max_context_tokens: u32,
    pub resident_weight_bytes: u64,
    pub planned_weight_blocks: u32,
    pub planned_gpu_resident_blocks: u32,
    pub planned_gpu_staged_blocks: u32,
    pub planned_weight_bytes: u64,
    pub descriptor_gpu_resident_h2d_bytes: u64,
    pub descriptor_gpu_staged_h2d_bytes: u64,
    pub planned_weight_descriptor_count: u32,
    pub planned_weight_descriptor_hash: u64,
    pub resident_kv_bytes: u64,
    pub device_arena_bytes: u64,
    pub device_total_memory_bytes: Option<usize>,
    pub device_free_memory_bytes: Option<usize>,
    pub fits_device_free_memory: Option<bool>,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaHfDecodeSequenceSessionCreateSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"status\":\"{}\",\"dtype\":{},\"hidden\":{},\"heads\":{},\"kv_heads\":{},\"head_dim\":{},\"intermediate\":{},\"vocab_size\":{},\"layer_count\":{},\"max_context_tokens\":{},\"resident_weight_bytes\":{},\"planned_weight_blocks\":{},\"planned_gpu_resident_blocks\":{},\"planned_gpu_staged_blocks\":{},\"planned_weight_bytes\":{},\"descriptor_gpu_resident_H2D_bytes\":{},\"descriptor_gpu_staged_H2D_bytes\":{},\"planned_weight_descriptor_count\":{},\"planned_weight_descriptor_hash\":{},\"resident_kv_bytes\":{},\"device_arena_bytes\":{},\"device_total_memory_bytes\":{},\"device_free_memory_bytes\":{},\"fits_device_free_memory\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status_str(&self.status),
            self.dtype,
            self.hidden,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.intermediate,
            self.vocab_size,
            self.layer_count,
            self.max_context_tokens,
            self.resident_weight_bytes,
            self.planned_weight_blocks,
            self.planned_gpu_resident_blocks,
            self.planned_gpu_staged_blocks,
            self.planned_weight_bytes,
            self.descriptor_gpu_resident_h2d_bytes,
            self.descriptor_gpu_staged_h2d_bytes,
            self.planned_weight_descriptor_count,
            self.planned_weight_descriptor_hash,
            self.resident_kv_bytes,
            self.device_arena_bytes,
            json_opt_usize(self.device_total_memory_bytes),
            json_opt_usize(self.device_free_memory_bytes),
            json_opt_bool(self.fits_device_free_memory),
            self.pinned_host_bytes,
            self.h2d_bytes,
            self.sync_calls,
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
    }
}

pub(crate) fn create_summary_from_result(
    return_code: i32,
    out: &NervaCudaHfDecodeSequenceSessionCreateResult,
    device_total_memory_bytes: Option<usize>,
    device_free_memory_bytes: Option<usize>,
) -> CudaHfDecodeSequenceSessionCreateSummary {
    let fits_device_free_memory =
        device_free_memory_bytes.map(|free| out.device_arena_bytes <= free as u64);
    CudaHfDecodeSequenceSessionCreateSummary {
        status: if return_code == 0 && out.status == 0 {
            SmokeStatus::Ok
        } else {
            SmokeStatus::Failed
        },
        dtype: out.dtype,
        hidden: out.hidden,
        heads: out.heads,
        kv_heads: out.kv_heads,
        head_dim: out.head_dim,
        intermediate: out.intermediate,
        vocab_size: out.vocab_size,
        layer_count: out.layer_count,
        max_context_tokens: out.max_context_tokens,
        resident_weight_bytes: out.resident_weight_bytes,
        planned_weight_blocks: out.planned_weight_blocks,
        planned_gpu_resident_blocks: out.planned_gpu_resident_blocks,
        planned_gpu_staged_blocks: out.planned_gpu_staged_blocks,
        planned_weight_bytes: out.planned_weight_bytes,
        descriptor_gpu_resident_h2d_bytes: out.descriptor_gpu_resident_h2d_bytes,
        descriptor_gpu_staged_h2d_bytes: out.descriptor_gpu_staged_h2d_bytes,
        planned_weight_descriptor_count: out.planned_weight_descriptor_count,
        planned_weight_descriptor_hash: out.planned_weight_descriptor_hash,
        resident_kv_bytes: out.resident_kv_bytes,
        device_arena_bytes: out.device_arena_bytes,
        device_total_memory_bytes,
        device_free_memory_bytes,
        fits_device_free_memory,
        pinned_host_bytes: out.pinned_host_bytes,
        h2d_bytes: out.h2d_bytes,
        sync_calls: out.sync_calls,
        hot_path_allocations: out.hot_path_allocations,
        error: (return_code != 0 || out.status != 0)
            .then(|| format!("CUDA HF decode sequence session failed: {}", out.cuda_error)),
    }
}

fn status_str(status: &SmokeStatus) -> &'static str {
    match status {
        SmokeStatus::Ok => "ok",
        SmokeStatus::Unavailable => "unavailable",
        SmokeStatus::Failed => "failed",
    }
}
