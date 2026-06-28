use crate::decode::hf_sequence::session::ffi::NervaCudaHfDecodeSequenceSessionCreateResult;
use crate::json::{json_opt_bool, json_opt_str, json_opt_usize};
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaHfDecodeSequenceSessionCreateSummary {
    pub status: SmokeStatus,
    pub failure_stage: i32,
    pub dtype: u32,
    pub hidden: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub vocab_size: u32,
    pub layer_count: u32,
    pub max_context_tokens: u32,
    pub prefill_chunk_tokens: u32,
    pub head_threads: u32,
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
            "{{\"status\":\"{}\",\"failure_stage\":{},\"failure_stage_label\":\"{}\",\"dtype\":{},\"hidden\":{},\"heads\":{},\"kv_heads\":{},\"head_dim\":{},\"intermediate\":{},\"vocab_size\":{},\"layer_count\":{},\"max_context_tokens\":{},\"prefill_chunk_tokens\":{},\"head_threads\":{},\"resident_weight_bytes\":{},\"planned_weight_blocks\":{},\"planned_gpu_resident_blocks\":{},\"planned_gpu_staged_blocks\":{},\"planned_weight_bytes\":{},\"descriptor_gpu_resident_H2D_bytes\":{},\"descriptor_gpu_staged_H2D_bytes\":{},\"planned_weight_descriptor_count\":{},\"planned_weight_descriptor_hash\":{},\"resident_kv_bytes\":{},\"device_arena_bytes\":{},\"device_total_memory_bytes\":{},\"device_free_memory_bytes\":{},\"fits_device_free_memory\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status_str(&self.status),
            self.failure_stage,
            create_stage_label(self.failure_stage),
            self.dtype,
            self.hidden,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.intermediate,
            self.vocab_size,
            self.layer_count,
            self.max_context_tokens,
            self.prefill_chunk_tokens,
            self.head_threads,
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
        failure_stage: out.failure_stage,
        dtype: out.dtype,
        hidden: out.hidden,
        heads: out.heads,
        kv_heads: out.kv_heads,
        head_dim: out.head_dim,
        intermediate: out.intermediate,
        vocab_size: out.vocab_size,
        layer_count: out.layer_count,
        max_context_tokens: out.max_context_tokens,
        prefill_chunk_tokens: out.prefill_chunk_tokens,
        head_threads: out.head_threads,
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
        error: (return_code != 0 || out.status != 0).then(|| {
            format!(
                "CUDA HF decode sequence session failed: cuda_error={} failure_stage={} ({})",
                out.cuda_error,
                out.failure_stage,
                create_stage_label(out.failure_stage),
            )
        }),
    }
}

pub(crate) fn create_stage_label(stage: i32) -> &'static str {
    match stage {
        0 => "none",
        1 => "invalid_request",
        2 => "cuda_get_device_count",
        3 => "cuda_set_device",
        4 => "session_alloc",
        5 => "host_weight_alloc",
        6 => "host_slots_alloc",
        7 => "device_arena_alloc",
        8 => "device_layouts_alloc",
        9 => "device_scratch_alloc",
        10 => "projection_input_alloc",
        11 => "packed_qkv_alloc",
        12 => "packed_gate_up_alloc",
        13 => "kv_keys_alloc",
        14 => "kv_values_alloc",
        15 => "prompt_tokens_alloc",
        16 => "device_slots_alloc",
        17 => "device_step_alloc",
        18 => "cublas_workspace_alloc",
        19 => "stream_create",
        20 => "cublas_create",
        21 => "cublaslt_create",
        22 => "cublas_configure",
        23 => "start_event_create",
        24 => "stop_event_create",
        25 => "descriptor_copy",
        26 => "layout_copy",
        27 => "pack_replicas",
        28 => "warm_cublas",
        29 => "setup_synchronize",
        30 => "prefill_hidden_alloc",
        31 => "prefill_chunk_alloc",
        32 => "decode_attention_alloc",
        33 => "verify_logits_alloc",
        _ => "unknown",
    }
}

fn status_str(status: &SmokeStatus) -> &'static str {
    match status {
        SmokeStatus::Ok => "ok",
        SmokeStatus::Unavailable => "unavailable",
        SmokeStatus::Failed => "failed",
    }
}
