use std::os::raw::c_int;

use crate::decode::hf_chain::ffi::NervaCudaHfDecodeChainLayer;
use crate::decode::hf_sequence::weight_plan::CudaHfDecodeSequenceWeightBlock;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaHfDecodeSequenceRequest {
    pub(crate) dtype: u32,
    pub(crate) hidden: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) intermediate: u32,
    pub(crate) vocab_size: u32,
    pub(crate) layer_count: u32,
    pub(crate) steps: u32,
    pub(crate) seed_token: u32,
    pub(crate) prompt_tokens: *const u32,
    pub(crate) prompt_token_count: u32,
    pub(crate) has_eos_token: u32,
    pub(crate) eos_token: u32,
    pub(crate) rms_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) embeddings: *const u16,
    pub(crate) layers: *const NervaCudaHfDecodeChainLayer,
    pub(crate) final_norm_weight: *const u16,
    pub(crate) lm_head: *const u16,
    pub(crate) planned_weight_blocks: u32,
    pub(crate) planned_gpu_resident_blocks: u32,
    pub(crate) planned_gpu_staged_blocks: u32,
    pub(crate) planned_weight_bytes: u64,
    pub(crate) planned_gpu_resident_weight_bytes: u64,
    pub(crate) planned_gpu_staged_weight_bytes: u64,
    pub(crate) planned_weight_descriptors: *const CudaHfDecodeSequenceWeightBlock,
    pub(crate) planned_weight_descriptor_count: u32,
    pub(crate) planned_weight_descriptor_hash: u64,
    pub(crate) output_tokens: *mut u32,
    pub(crate) output_token_capacity: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaHfDecodeSequenceResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) dtype: u32,
    pub(crate) hidden: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) intermediate: u32,
    pub(crate) vocab_size: u32,
    pub(crate) layer_count: u32,
    pub(crate) steps: u32,
    pub(crate) seed_token: u32,
    pub(crate) observed_tokens: u32,
    pub(crate) last_token: u32,
    pub(crate) observed_token_hash: u64,
    pub(crate) resident_weight_bytes: u64,
    pub(crate) planned_weight_blocks: u32,
    pub(crate) planned_gpu_resident_blocks: u32,
    pub(crate) planned_gpu_staged_blocks: u32,
    pub(crate) planned_weight_bytes: u64,
    pub(crate) planned_gpu_resident_weight_bytes: u64,
    pub(crate) planned_gpu_staged_weight_bytes: u64,
    pub(crate) descriptor_gpu_resident_h2d_bytes: u64,
    pub(crate) descriptor_gpu_staged_h2d_bytes: u64,
    pub(crate) planned_weight_descriptor_count: u32,
    pub(crate) planned_weight_descriptor_hash: u64,
    pub(crate) resident_kv_bytes: u64,
    pub(crate) kv_tokens: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) h2d_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) graph_replays: u64,
    pub(crate) graph_nodes: u64,
    pub(crate) graph_launches: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) device_elapsed_ns: u64,
    pub(crate) sync_calls: u64,
    pub(crate) host_causality_edges: u64,
    pub(crate) hot_path_allocations: u64,
}

unsafe extern "C" {
    fn nerva_cuda_hf_decode_sequence_u16(
        request: *const NervaCudaHfDecodeSequenceRequest,
        out: *mut NervaCudaHfDecodeSequenceResult,
    ) -> c_int;
}

pub(crate) fn run_hf_decode_sequence_u16(
    request: &NervaCudaHfDecodeSequenceRequest,
    out: &mut NervaCudaHfDecodeSequenceResult,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_sequence_u16(request, out) }
}
