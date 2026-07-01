use std::os::raw::c_int;

use crate::decode::hf_chain::ffi::NervaCudaHfDecodeChainLayer;
use crate::decode::hf_sequence::weight_plan::CudaHfDecodeSequenceWeightBlock;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NervaCudaHfDecodeSamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub reserved: u32,
    pub seed: u64,
}

impl Default for NervaCudaHfDecodeSamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            reserved: 0,
            seed: 0,
        }
    }
}

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
    pub(crate) sampler: NervaCudaHfDecodeSamplerConfig,
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
    pub(crate) deepseek_mhc_residual_bytes: u64,
    pub(crate) deepseek_mhc_post_mix_bytes: u64,
    pub(crate) deepseek_mhc_comb_mix_bytes: u64,
    pub(crate) kv_tokens: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) h2d_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) graph_replays: u64,
    pub(crate) graph_nodes: u64,
    pub(crate) graph_launches: u64,
    pub(crate) graph_captures: u64,
    pub(crate) graph_cache_hits: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) experimental_rt_selector_launches: u64,
    pub(crate) experimental_rt_sparse_attention_active: u32,
    pub(crate) experimental_rt_dense_attention_chunks: u32,
    pub(crate) experimental_rt_attention_chunks: u32,
    pub(crate) experimental_rt_reserved: u32,
    pub(crate) device_elapsed_ns: u64,
    pub(crate) projection_ns: u64,
    pub(crate) qkv_projection_ns: u64,
    pub(crate) attention_output_projection_ns: u64,
    pub(crate) gate_up_projection_ns: u64,
    pub(crate) down_projection_ns: u64,
    pub(crate) lm_head_projection_ns: u64,
    pub(crate) attention_ns: u64,
    pub(crate) mlp_ns: u64,
    pub(crate) norm_ns: u64,
    pub(crate) sampling_ns: u64,
    pub(crate) sync_calls: u64,
    pub(crate) host_causality_edges: u64,
    pub(crate) hot_path_allocations: u64,
    pub(crate) deepseek_compressor_state_writes: u64,
    pub(crate) deepseek_compressed_kv_writes: u64,
    pub(crate) deepseek_indexer_state_writes: u64,
    pub(crate) deepseek_indexer_kv_writes: u64,
    pub(crate) deepseek_compressed_kv_attention_reads: u64,
    pub(crate) deepseek_compressed_kv_attention_slots_scanned: u64,
    pub(crate) deepseek_sparse_topk_selections: u64,
    pub(crate) deepseek_sparse_topk_slots_selected: u64,
    pub(crate) deepseek_sparse_topk_candidates_scored: u64,
    pub(crate) deepseek_sparse_topk_selection_hash: u64,
    pub(crate) deepseek_v3_grouped_router_selections: u64,
    pub(crate) deepseek_v4_bias_router_selections: u64,
    pub(crate) deepseek_v4_hash_router_selections: u64,
    pub(crate) deepseek_raw_attention_tokens_scanned: u64,
    pub(crate) deepseek_sparse_attention_output_hash: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaHfDecodeSequenceLayoutPlanRequest {
    pub(crate) hidden: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) intermediate: u32,
    pub(crate) vocab_size: u32,
    pub(crate) layer_count: u32,
    pub(crate) layer_index: u32,
    pub(crate) layers: *const NervaCudaHfDecodeChainLayer,
}

#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct NervaCudaHfDecodeSequenceLayoutPlanResult {
    pub(crate) status: i32,
    pub(crate) hidden: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) intermediate: u32,
    pub(crate) vocab_size: u32,
    pub(crate) layer_count: u32,
    pub(crate) layer_index: u32,
    pub(crate) attention_kind: u32,
    pub(crate) deepseek_mode: u32,
    pub(crate) deepseek_flags: u32,
    pub(crate) deepseek_hc_mult: u32,
    pub(crate) deepseek_hc_sinkhorn_iters: u32,
    pub(crate) deepseek_qk_head_dim: u32,
    pub(crate) deepseek_q_rows: u32,
    pub(crate) deepseek_kv_cache_width: u32,
    pub(crate) deepseek_kv_b_rows: u32,
    pub(crate) deepseek_value_rows: u32,
    pub(crate) resident_weight_bytes: u64,
    pub(crate) layout_bytes: u64,
    pub(crate) final_norm: u64,
    pub(crate) lm_head: u64,
    pub(crate) rms_attn: u64,
    pub(crate) rms_mlp: u64,
    pub(crate) w_q: u64,
    pub(crate) q_norm: u64,
    pub(crate) w_k: u64,
    pub(crate) k_norm: u64,
    pub(crate) w_v: u64,
    pub(crate) w_o: u64,
    pub(crate) w_router: u64,
    pub(crate) w_expert_gate_up: u64,
    pub(crate) w_expert_down: u64,
    pub(crate) deepseek_q_a_scale: u64,
    pub(crate) deepseek_q_b: u64,
    pub(crate) deepseek_q_b_scale: u64,
    pub(crate) deepseek_kv_a_scale: u64,
    pub(crate) deepseek_kv_b_scale: u64,
    pub(crate) deepseek_o_a_scale: u64,
    pub(crate) deepseek_o_b: u64,
    pub(crate) deepseek_o_b_scale: u64,
    pub(crate) deepseek_attention_sink: u64,
    pub(crate) deepseek_indexer_q: u64,
    pub(crate) deepseek_indexer_q_scale: u64,
    pub(crate) deepseek_indexer_k: u64,
    pub(crate) deepseek_indexer_k_scale: u64,
    pub(crate) deepseek_indexer_k_norm: u64,
    pub(crate) deepseek_indexer_k_norm_bias: u64,
    pub(crate) deepseek_indexer_weights: u64,
    pub(crate) deepseek_compressor_ape: u64,
    pub(crate) deepseek_compressor_wkv: u64,
    pub(crate) deepseek_compressor_wgate: u64,
    pub(crate) deepseek_compressor_norm: u64,
    pub(crate) deepseek_indexer_compressor_ape: u64,
    pub(crate) deepseek_indexer_compressor_wkv: u64,
    pub(crate) deepseek_indexer_compressor_wgate: u64,
    pub(crate) deepseek_indexer_compressor_norm: u64,
    pub(crate) deepseek_hc_head_base: u64,
    pub(crate) deepseek_hc_head_fn: u64,
    pub(crate) deepseek_hc_head_scale: u64,
    pub(crate) deepseek_hc_attn_base: u64,
    pub(crate) deepseek_hc_attn_fn: u64,
    pub(crate) deepseek_hc_attn_scale: u64,
    pub(crate) deepseek_hc_ffn_base: u64,
    pub(crate) deepseek_hc_ffn_fn: u64,
    pub(crate) deepseek_hc_ffn_scale: u64,
    pub(crate) deepseek_hc_eps: f32,
    pub(crate) deepseek_hc_post_alpha: f32,
    pub(crate) deepseek_compress_rope_theta: f32,
    pub(crate) deepseek_swiglu_limit: f32,
}

unsafe extern "C" {
    fn nerva_cuda_hf_decode_sequence_u16(
        request: *const NervaCudaHfDecodeSequenceRequest,
        out: *mut NervaCudaHfDecodeSequenceResult,
    ) -> c_int;
    fn nerva_cuda_hf_decode_sequence_plan_layout(
        request: *const NervaCudaHfDecodeSequenceLayoutPlanRequest,
        out: *mut NervaCudaHfDecodeSequenceLayoutPlanResult,
    ) -> c_int;
}

pub(crate) fn run_hf_decode_sequence_u16(
    request: &NervaCudaHfDecodeSequenceRequest,
    out: &mut NervaCudaHfDecodeSequenceResult,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_sequence_u16(request, out) }
}

pub(crate) fn plan_hf_decode_sequence_layout(
    request: &NervaCudaHfDecodeSequenceLayoutPlanRequest,
    out: &mut NervaCudaHfDecodeSequenceLayoutPlanResult,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_sequence_plan_layout(request, out) }
}
