use std::os::raw::{c_int, c_void};

use crate::decode::hf_chain::ffi::NervaCudaHfDecodeChainLayer;
use crate::decode::hf_sequence::ffi::{
    NervaCudaHfDecodeSamplerConfig, NervaCudaHfDecodeSequenceResult,
};
use crate::decode::hf_sequence::weight_plan::CudaHfDecodeSequenceWeightBlock;

pub(crate) type NervaCudaHfDecodeSequenceSession = c_void;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaHfDecodeSequenceSessionCreateRequest {
    pub(crate) dtype: u32,
    pub(crate) hidden: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) intermediate: u32,
    pub(crate) vocab_size: u32,
    pub(crate) layer_count: u32,
    pub(crate) max_context_tokens: u32,
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
    pub(crate) detailed_profile: u32,
    pub(crate) experimental_rt_decode: u32,
    pub(crate) experimental_rt_mode: u32,
    pub(crate) experimental_rt_page_tokens: u32,
    pub(crate) experimental_rt_pages: u32,
    pub(crate) experimental_rt_local_window_tokens: u32,
    pub(crate) experimental_rt_sink_tokens: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaHfDecodeSequenceSessionCreateResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) failure_stage: i32,
    pub(crate) device_count: i32,
    pub(crate) dtype: u32,
    pub(crate) hidden: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) intermediate: u32,
    pub(crate) vocab_size: u32,
    pub(crate) layer_count: u32,
    pub(crate) max_context_tokens: u32,
    pub(crate) prefill_chunk_tokens: u32,
    pub(crate) head_threads: u32,
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
    pub(crate) experimental_rt_decode_requested: u32,
    pub(crate) experimental_rt_decode_enabled: u32,
    pub(crate) experimental_rt_mode: u32,
    pub(crate) experimental_rt_page_tokens: u32,
    pub(crate) experimental_rt_pages: u32,
    pub(crate) experimental_rt_local_window_tokens: u32,
    pub(crate) experimental_rt_sink_tokens: u32,
    pub(crate) resident_kv_bytes: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) h2d_bytes: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaHfDecodeSequenceSessionForkSharedWeightsRequest {
    pub(crate) parent: *mut NervaCudaHfDecodeSequenceSession,
    pub(crate) detailed_profile: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaHfDecodeSequenceSessionRunRequest {
    pub(crate) session: *mut NervaCudaHfDecodeSequenceSession,
    pub(crate) steps: u32,
    pub(crate) seed_token: u32,
    pub(crate) prompt_tokens: *const u32,
    pub(crate) prompt_token_count: u32,
    pub(crate) has_eos_token: u32,
    pub(crate) eos_token: u32,
    pub(crate) output_tokens: *mut u32,
    pub(crate) output_token_capacity: u32,
    pub(crate) sampler: NervaCudaHfDecodeSamplerConfig,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaHfDecodeSequenceSessionStartRequest {
    pub(crate) session: *mut NervaCudaHfDecodeSequenceSession,
    pub(crate) prompt_tokens: *const u32,
    pub(crate) prompt_token_count: u32,
    pub(crate) has_eos_token: u32,
    pub(crate) eos_token: u32,
    pub(crate) sampler: NervaCudaHfDecodeSamplerConfig,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaHfDecodeSequenceSessionAdvanceRequest {
    pub(crate) session: *mut NervaCudaHfDecodeSequenceSession,
    pub(crate) steps: u32,
    pub(crate) output_tokens: *mut u32,
    pub(crate) output_token_capacity: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaHfDecodeSequenceProjectionBatchPlanRequest {
    pub(crate) sessions: *mut *mut NervaCudaHfDecodeSequenceSession,
    pub(crate) session_count: u32,
    pub(crate) target_block_tokens: u32,
    pub(crate) min_block_tokens: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaHfDecodeSequenceProjectionBatchPlanResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) reason: u32,
    pub(crate) exact: u32,
    pub(crate) requested_session_count: u32,
    pub(crate) eligible_session_count: u32,
    pub(crate) block_tokens: u32,
    pub(crate) target_block_tokens: u32,
    pub(crate) min_block_tokens: u32,
    pub(crate) dtype: u32,
    pub(crate) hidden: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) intermediate: u32,
    pub(crate) vocab_size: u32,
    pub(crate) layer_count: u32,
    pub(crate) max_context_tokens: u32,
    pub(crate) planned_weight_descriptor_hash: u64,
    pub(crate) resident_weight_bytes: u64,
    pub(crate) qkv_rows: u64,
    pub(crate) gate_up_rows: u64,
    pub(crate) qkv_input_bytes: u64,
    pub(crate) qkv_output_bytes: u64,
    pub(crate) attention_output_input_bytes: u64,
    pub(crate) attention_output_output_bytes: u64,
    pub(crate) gate_up_input_bytes: u64,
    pub(crate) gate_up_output_bytes: u64,
    pub(crate) down_input_bytes: u64,
    pub(crate) down_output_bytes: u64,
    pub(crate) lm_head_input_bytes: u64,
    pub(crate) lm_head_output_bytes: u64,
    pub(crate) pack_input_bytes: u64,
    pub(crate) max_projection_output_bytes: u64,
    pub(crate) hot_path_allocations: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest {
    pub(crate) sessions: *mut *mut NervaCudaHfDecodeSequenceSession,
    pub(crate) session_count: u32,
    pub(crate) target_block_tokens: u32,
    pub(crate) min_block_tokens: u32,
    pub(crate) projection_kind: u32,
    pub(crate) layer_index: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaHfDecodeSequenceProjectionBatchExecuteResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) reason: u32,
    pub(crate) exact: u32,
    pub(crate) projection_kind: u32,
    pub(crate) layer_index: u32,
    pub(crate) requested_session_count: u32,
    pub(crate) eligible_session_count: u32,
    pub(crate) block_tokens: u32,
    pub(crate) target_block_tokens: u32,
    pub(crate) min_block_tokens: u32,
    pub(crate) dtype: u32,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) input_bytes: u64,
    pub(crate) output_bytes: u64,
    pub(crate) elapsed_ns: u64,
    pub(crate) pack_kernel_launches: u64,
    pub(crate) projection_kernel_launches: u64,
    pub(crate) scatter_kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteRequest {
    pub(crate) sessions: *mut *mut NervaCudaHfDecodeSequenceSession,
    pub(crate) session_count: u32,
    pub(crate) target_block_tokens: u32,
    pub(crate) min_block_tokens: u32,
    pub(crate) layer_index: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) reason: u32,
    pub(crate) exact: u32,
    pub(crate) layer_index: u32,
    pub(crate) requested_session_count: u32,
    pub(crate) eligible_session_count: u32,
    pub(crate) block_tokens: u32,
    pub(crate) target_block_tokens: u32,
    pub(crate) min_block_tokens: u32,
    pub(crate) dtype: u32,
    pub(crate) qkv_rows: u32,
    pub(crate) attention_output_rows: u32,
    pub(crate) gate_up_rows: u32,
    pub(crate) down_rows: u32,
    pub(crate) hidden_cols: u32,
    pub(crate) attention_output_cols: u32,
    pub(crate) down_cols: u32,
    pub(crate) input_bytes: u64,
    pub(crate) output_bytes: u64,
    pub(crate) elapsed_ns: u64,
    pub(crate) qkv_elapsed_ns: u64,
    pub(crate) attention_output_elapsed_ns: u64,
    pub(crate) gate_up_elapsed_ns: u64,
    pub(crate) down_elapsed_ns: u64,
    pub(crate) pack_kernel_launches: u64,
    pub(crate) projection_kernel_launches: u64,
    pub(crate) scatter_kernel_launches: u64,
    pub(crate) dependency_kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaHfDecodeSequenceBatchAdvanceRequest {
    pub(crate) sessions: *mut *mut NervaCudaHfDecodeSequenceSession,
    pub(crate) session_count: u32,
    pub(crate) target_block_tokens: u32,
    pub(crate) min_block_tokens: u32,
    pub(crate) output_tokens: *mut u32,
    pub(crate) output_token_capacity: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaHfDecodeSequenceBatchAdvanceResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) reason: u32,
    pub(crate) exact: u32,
    pub(crate) requested_session_count: u32,
    pub(crate) eligible_session_count: u32,
    pub(crate) block_tokens: u32,
    pub(crate) target_block_tokens: u32,
    pub(crate) min_block_tokens: u32,
    pub(crate) dtype: u32,
    pub(crate) layer_count: u32,
    pub(crate) observed_tokens: u32,
    pub(crate) last_token: u32,
    pub(crate) observed_token_hash: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) projection_elapsed_ns: u64,
    pub(crate) qkv_elapsed_ns: u64,
    pub(crate) attention_output_elapsed_ns: u64,
    pub(crate) gate_up_elapsed_ns: u64,
    pub(crate) down_elapsed_ns: u64,
    pub(crate) lm_head_elapsed_ns: u64,
    pub(crate) pack_kernel_launches: u64,
    pub(crate) projection_kernel_launches: u64,
    pub(crate) scatter_kernel_launches: u64,
    pub(crate) dependency_kernel_launches: u64,
    pub(crate) sampling_kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

pub(crate) const PROJECTION_BATCH_PLAN_READY: u32 = 0;
pub(crate) const PROJECTION_BATCH_PLAN_INVALID_REQUEST: u32 = 1;
pub(crate) const PROJECTION_BATCH_PLAN_NO_SESSIONS: u32 = 2;
pub(crate) const PROJECTION_BATCH_PLAN_NO_READY_SESSIONS: u32 = 3;
pub(crate) const PROJECTION_BATCH_PLAN_SHARED_WEIGHTS_UNPROVEN: u32 = 4;
pub(crate) const PROJECTION_BATCH_PLAN_INSUFFICIENT_COMPATIBLE_READY: u32 = 5;
pub(crate) const PROJECTION_BATCH_PLAN_UNSUPPORTED_PROJECTION: u32 = 6;
pub(crate) const PROJECTION_BATCH_PLAN_INVALID_LAYER: u32 = 7;
pub(crate) const PROJECTION_BATCH_PLAN_INSUFFICIENT_SCRATCH: u32 = 8;
pub(crate) const PROJECTION_BATCH_KIND_QKV: u32 = 1;
pub(crate) const PROJECTION_BATCH_KIND_ATTENTION_OUTPUT: u32 = 2;
pub(crate) const PROJECTION_BATCH_KIND_GATE_UP: u32 = 3;
pub(crate) const PROJECTION_BATCH_KIND_DOWN: u32 = 4;
pub(crate) const PROJECTION_BATCH_KIND_LM_HEAD: u32 = 5;

unsafe extern "C" {
    fn nerva_cuda_hf_decode_sequence_session_create(
        request: *const NervaCudaHfDecodeSequenceSessionCreateRequest,
        out: *mut NervaCudaHfDecodeSequenceSessionCreateResult,
        session: *mut *mut NervaCudaHfDecodeSequenceSession,
    ) -> c_int;
    fn nerva_cuda_hf_decode_sequence_session_run(
        request: *const NervaCudaHfDecodeSequenceSessionRunRequest,
        out: *mut NervaCudaHfDecodeSequenceResult,
    ) -> c_int;
    fn nerva_cuda_hf_decode_sequence_session_start(
        request: *const NervaCudaHfDecodeSequenceSessionStartRequest,
        out: *mut NervaCudaHfDecodeSequenceResult,
    ) -> c_int;
    fn nerva_cuda_hf_decode_sequence_session_advance(
        request: *const NervaCudaHfDecodeSequenceSessionAdvanceRequest,
        out: *mut NervaCudaHfDecodeSequenceResult,
    ) -> c_int;
    fn nerva_cuda_hf_decode_sequence_projection_batch_plan(
        request: *const NervaCudaHfDecodeSequenceProjectionBatchPlanRequest,
        out: *mut NervaCudaHfDecodeSequenceProjectionBatchPlanResult,
    ) -> c_int;
    fn nerva_cuda_hf_decode_sequence_projection_batch_execute(
        request: *const NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest,
        out: *mut NervaCudaHfDecodeSequenceProjectionBatchExecuteResult,
    ) -> c_int;
    fn nerva_cuda_hf_decode_sequence_layer_projection_batch_execute(
        request: *const NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteRequest,
        out: *mut NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult,
    ) -> c_int;
    fn nerva_cuda_hf_decode_sequence_batch_advance_one(
        request: *const NervaCudaHfDecodeSequenceBatchAdvanceRequest,
        out: *mut NervaCudaHfDecodeSequenceBatchAdvanceResult,
    ) -> c_int;
    fn nerva_cuda_hf_decode_sequence_session_fork_shared_weights(
        request: *const NervaCudaHfDecodeSequenceSessionForkSharedWeightsRequest,
        out: *mut NervaCudaHfDecodeSequenceSessionCreateResult,
        session: *mut *mut NervaCudaHfDecodeSequenceSession,
    ) -> c_int;
    fn nerva_cuda_hf_decode_sequence_session_destroy(
        session: *mut NervaCudaHfDecodeSequenceSession,
        out: *mut NervaCudaHfDecodeSequenceSessionCreateResult,
    ) -> c_int;
}

pub(crate) fn create_hf_decode_sequence_session(
    request: &NervaCudaHfDecodeSequenceSessionCreateRequest,
    out: &mut NervaCudaHfDecodeSequenceSessionCreateResult,
    session: &mut *mut NervaCudaHfDecodeSequenceSession,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_sequence_session_create(request, out, session) }
}

pub(crate) fn run_hf_decode_sequence_session(
    request: &NervaCudaHfDecodeSequenceSessionRunRequest,
    out: &mut NervaCudaHfDecodeSequenceResult,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_sequence_session_run(request, out) }
}

pub(crate) fn start_hf_decode_sequence_session(
    request: &NervaCudaHfDecodeSequenceSessionStartRequest,
    out: &mut NervaCudaHfDecodeSequenceResult,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_sequence_session_start(request, out) }
}

pub(crate) fn advance_hf_decode_sequence_session(
    request: &NervaCudaHfDecodeSequenceSessionAdvanceRequest,
    out: &mut NervaCudaHfDecodeSequenceResult,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_sequence_session_advance(request, out) }
}

pub(crate) fn plan_hf_decode_sequence_projection_batch(
    request: &NervaCudaHfDecodeSequenceProjectionBatchPlanRequest,
    out: &mut NervaCudaHfDecodeSequenceProjectionBatchPlanResult,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_sequence_projection_batch_plan(request, out) }
}

pub(crate) fn execute_hf_decode_sequence_projection_batch(
    request: &NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest,
    out: &mut NervaCudaHfDecodeSequenceProjectionBatchExecuteResult,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_sequence_projection_batch_execute(request, out) }
}

pub(crate) fn execute_hf_decode_sequence_layer_projection_batch(
    request: &NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteRequest,
    out: &mut NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_sequence_layer_projection_batch_execute(request, out) }
}

pub(crate) fn batch_advance_one_hf_decode_sequence(
    request: &NervaCudaHfDecodeSequenceBatchAdvanceRequest,
    out: &mut NervaCudaHfDecodeSequenceBatchAdvanceResult,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_sequence_batch_advance_one(request, out) }
}

pub(crate) fn fork_shared_weights_hf_decode_sequence_session(
    request: &NervaCudaHfDecodeSequenceSessionForkSharedWeightsRequest,
    out: &mut NervaCudaHfDecodeSequenceSessionCreateResult,
    session: &mut *mut NervaCudaHfDecodeSequenceSession,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_sequence_session_fork_shared_weights(request, out, session) }
}

pub(crate) fn destroy_hf_decode_sequence_session(
    session: *mut NervaCudaHfDecodeSequenceSession,
    out: &mut NervaCudaHfDecodeSequenceSessionCreateResult,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_sequence_session_destroy(session, out) }
}
