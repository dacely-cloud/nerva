use core::ptr;

use crate::decode::ffi::CUDA_ERROR_NO_DEVICE;
use crate::decode::hf_sequence::ffi::NervaCudaHfDecodeSequenceResult;
use crate::decode::hf_sequence::footprint::CudaHfDecodeSequenceFootprint;
use crate::decode::hf_sequence::session::summary::CudaHfDecodeSequenceSessionCreateSummary;
use crate::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use crate::decode::hf_sequence::weight_plan::{
    CudaHfDecodeSequenceWeightBlock, CudaHfDecodeSequenceWeightPlan,
};
use crate::smoke::status::SmokeStatus;

pub(super) fn planned_ptr(slice: &[u16], plan: CudaHfDecodeSequenceWeightPlan) -> *const u16 {
    if plan.is_declared() {
        ptr::null()
    } else {
        slice.as_ptr()
    }
}

pub(super) fn descriptor_ptr(
    blocks: &[CudaHfDecodeSequenceWeightBlock],
) -> *const CudaHfDecodeSequenceWeightBlock {
    if blocks.is_empty() {
        ptr::null()
    } else {
        blocks.as_ptr()
    }
}

pub(super) fn validate_run(prompt_tokens: &[u32], steps: usize, vocab_size: u32) -> Option<String> {
    if steps == 0 || prompt_tokens.is_empty() {
        return Some("CUDA HF decode sequence session run requires prompt and steps".to_string());
    }
    if prompt_tokens.iter().any(|token| *token >= vocab_size) {
        return Some(
            "CUDA HF decode sequence session prompt token is outside vocabulary".to_string(),
        );
    }
    None
}

pub(super) fn summary_from_run(
    return_code: i32,
    out: &NervaCudaHfDecodeSequenceResult,
    tokens: Vec<u32>,
    create: &CudaHfDecodeSequenceSessionCreateSummary,
) -> CudaHfDecodeSequenceSummary {
    let footprint = CudaHfDecodeSequenceFootprint {
        context_tokens: out.graph_replays,
        resident_weight_bytes: out.resident_weight_bytes,
        device_arena_bytes: out.device_arena_bytes,
        pinned_host_bytes: out.pinned_host_bytes,
        layout_bytes: 0,
        scratch_bytes: 0,
        resident_kv_bytes: out.resident_kv_bytes,
        token_slot_bytes: out.d2h_bytes,
        prompt_bytes: out.h2d_bytes,
    };
    CudaHfDecodeSequenceSummary {
        status: run_status(return_code, out),
        dtype: out.dtype,
        hidden: out.hidden,
        heads: out.heads,
        kv_heads: out.kv_heads,
        head_dim: out.head_dim,
        intermediate: out.intermediate,
        vocab_size: out.vocab_size,
        layer_count: out.layer_count,
        steps: out.steps,
        seed_token: out.seed_token,
        tokens,
        observed_token_hash: out.observed_token_hash,
        planned_footprint: footprint,
        device_total_memory_bytes: create.device_total_memory_bytes,
        device_free_memory_bytes: create.device_free_memory_bytes,
        fits_device_free_memory: create.fits_device_free_memory,
        resident_weight_bytes: out.resident_weight_bytes,
        planned_weight_blocks: out.planned_weight_blocks,
        planned_gpu_resident_blocks: out.planned_gpu_resident_blocks,
        planned_gpu_staged_blocks: out.planned_gpu_staged_blocks,
        planned_weight_bytes: out.planned_weight_bytes,
        planned_gpu_resident_weight_bytes: out.planned_gpu_resident_weight_bytes,
        planned_gpu_staged_weight_bytes: out.planned_gpu_staged_weight_bytes,
        descriptor_gpu_resident_h2d_bytes: out.descriptor_gpu_resident_h2d_bytes,
        descriptor_gpu_staged_h2d_bytes: out.descriptor_gpu_staged_h2d_bytes,
        planned_weight_descriptor_count: out.planned_weight_descriptor_count,
        planned_weight_descriptor_hash: out.planned_weight_descriptor_hash,
        resident_kv_bytes: out.resident_kv_bytes,
        deepseek_mhc_residual_bytes: out.deepseek_mhc_residual_bytes,
        deepseek_mhc_post_mix_bytes: out.deepseek_mhc_post_mix_bytes,
        deepseek_mhc_comb_mix_bytes: out.deepseek_mhc_comb_mix_bytes,
        kv_tokens: out.kv_tokens,
        device_arena_bytes: out.device_arena_bytes,
        pinned_host_bytes: out.pinned_host_bytes,
        h2d_bytes: out.h2d_bytes,
        d2h_bytes: out.d2h_bytes,
        graph_replays: out.graph_replays,
        graph_nodes: out.graph_nodes,
        graph_launches: out.graph_launches,
        graph_captures: out.graph_captures,
        graph_cache_hits: out.graph_cache_hits,
        kernel_launches: out.kernel_launches,
        experimental_rt_selector_launches: out.experimental_rt_selector_launches,
        experimental_rt_sparse_attention_active: out.experimental_rt_sparse_attention_active != 0,
        experimental_rt_dense_attention_chunks: out.experimental_rt_dense_attention_chunks,
        experimental_rt_attention_chunks: out.experimental_rt_attention_chunks,
        device_elapsed_ns: out.device_elapsed_ns,
        projection_ns: out.projection_ns,
        qkv_projection_ns: out.qkv_projection_ns,
        attention_output_projection_ns: out.attention_output_projection_ns,
        gate_up_projection_ns: out.gate_up_projection_ns,
        down_projection_ns: out.down_projection_ns,
        lm_head_projection_ns: out.lm_head_projection_ns,
        attention_ns: out.attention_ns,
        mlp_ns: out.mlp_ns,
        norm_ns: out.norm_ns,
        sampling_ns: out.sampling_ns,
        sync_calls: out.sync_calls,
        host_causality_edges: out.host_causality_edges,
        hot_path_allocations: out.hot_path_allocations,
        deepseek_compressor_state_writes: out.deepseek_compressor_state_writes,
        deepseek_compressed_kv_writes: out.deepseek_compressed_kv_writes,
        deepseek_indexer_state_writes: out.deepseek_indexer_state_writes,
        deepseek_indexer_kv_writes: out.deepseek_indexer_kv_writes,
        deepseek_compressed_kv_attention_reads: out.deepseek_compressed_kv_attention_reads,
        deepseek_compressed_kv_attention_slots_scanned: out
            .deepseek_compressed_kv_attention_slots_scanned,
        deepseek_sparse_topk_selections: out.deepseek_sparse_topk_selections,
        deepseek_sparse_topk_slots_selected: out.deepseek_sparse_topk_slots_selected,
        deepseek_sparse_topk_candidates_scored: out.deepseek_sparse_topk_candidates_scored,
        deepseek_sparse_topk_selection_hash: out.deepseek_sparse_topk_selection_hash,
        deepseek_v3_grouped_router_selections: out.deepseek_v3_grouped_router_selections,
        deepseek_v4_bias_router_selections: out.deepseek_v4_bias_router_selections,
        deepseek_v4_hash_router_selections: out.deepseek_v4_hash_router_selections,
        deepseek_raw_attention_tokens_scanned: out.deepseek_raw_attention_tokens_scanned,
        deepseek_sparse_attention_output_hash: out.deepseek_sparse_attention_output_hash,
        error: (return_code != 0 || out.status != 0).then(|| run_error(return_code, out)),
    }
}

fn run_status(return_code: i32, out: &NervaCudaHfDecodeSequenceResult) -> SmokeStatus {
    if return_code == 0 && out.status == 0 {
        SmokeStatus::Ok
    } else if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        SmokeStatus::Unavailable
    } else {
        SmokeStatus::Failed
    }
}

fn run_error(return_code: i32, out: &NervaCudaHfDecodeSequenceResult) -> String {
    format!(
        "CUDA HF decode sequence session run failed: return_code={return_code} status={} cuda_error={}",
        out.status, out.cuda_error,
    )
}
