#pragma once

#include "../types.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

// Unified MLA flash-attention family (shared by decode and batched
// prefill). One block covers one query position and a fixed tile of
// kDeepSeekMlaFaHeadTile heads; KV is staged in fixed ascending tiles of
// kDeepSeekMlaFaTokenTile tokens so the per-row arithmetic is identical for
// any launch shape.
constexpr uint32_t kDeepSeekMlaFaHeadTile = 16;
constexpr uint32_t kDeepSeekMlaFaTokenTile = 64;
constexpr uint32_t kDeepSeekMlaFaWarps = 8;
// KV latent dims served by the bf16 MMA kernel (all DeepSeek V3/V3.2
// checkpoints); other dims take the generic fallback kernel.
constexpr uint32_t kDeepSeekMlaFaLora = 512;
constexpr uint32_t kDeepSeekMlaFaRope = 64;
constexpr uint32_t kDeepSeekMlaFaWidth = kDeepSeekMlaFaLora + kDeepSeekMlaFaRope;
// Shared-memory row strides (halves) padded to keep ldmatrix bank-conflict
// free: 584 * 2B = 73 * 16B (odd multiple of 16B).
constexpr uint32_t kDeepSeekMlaFaSmemStride = kDeepSeekMlaFaWidth + 8;
constexpr uint32_t kDeepSeekMlaFaPStride = kDeepSeekMlaFaTokenTile + 8;
// Tokens absorbed per block by the query-latent and V-projection kernels;
// each output element stays a private serial dot product so the group size
// cannot change any value.
constexpr uint32_t kDeepSeekMlaQLatentTokensPerBlock = 8;
constexpr uint32_t kDeepSeekMlaQLatentMaxHeadDim = 256;
constexpr uint32_t kDeepSeekMlaVProjMaxLora = kDeepSeekMlaFaLora;
// Query positions processed per attention sub-launch in the batched prefill
// path (bounds the (token, head) latent staging buffers).
constexpr uint32_t kDeepSeekMlaAttentionSubChunkTokens = 128;

__host__ __device__ __forceinline__ uint32_t deepseek_mla_fa_smem_bytes() {
  return (kDeepSeekMlaFaHeadTile * kDeepSeekMlaFaSmemStride +
          kDeepSeekMlaFaTokenTile * kDeepSeekMlaFaSmemStride +
          kDeepSeekMlaFaHeadTile * kDeepSeekMlaFaPStride) *
             2u +
         kDeepSeekMlaFaHeadTile * kDeepSeekMlaFaWarps * 4u +
         kDeepSeekMlaFaTokenTile * 4u;
}
// Maximum number of indexer query heads processed by one block of
// hf_deepseek_v32_indexer_query_state_tokens_kernel.
constexpr uint32_t kDeepSeekV32IndexerQueryHeadsPerBlock = 2;
// Maximum q_lora_rank staged in dynamic shared memory by
// hf_deepseek_v32_indexer_query_state_tokens_kernel.
constexpr uint32_t kDeepSeekV32IndexerQueryStageMaxCols = 4096;

__host__ __device__ __forceinline__ uint32_t
deepseek_v32_indexer_query_heads_per_block(uint32_t index_head_dim,
                                           uint32_t block_threads) {
  if (index_head_dim == 0 || index_head_dim > block_threads) {
    return 1u;
  }
  const uint32_t fit = block_threads / index_head_dim;
  return fit < kDeepSeekV32IndexerQueryHeadsPerBlock
             ? fit
             : kDeepSeekV32IndexerQueryHeadsPerBlock;
}

__global__ void hf_deepseek_v32_indexer_kv_project_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t *step_cursor, uint32_t max_steps,
    const uint16_t *projection_input, float *projected_values);
__global__ void hf_deepseek_v32_indexer_kv_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t *step_cursor,
    uint32_t max_steps, float rope_theta, const float *projected_values,
    uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v32_indexer_kv_encode_tokens_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t chunk_start, uint32_t chunk_tokens,
    uint32_t max_steps, float rope_theta, const uint16_t *projection_input,
    uint32_t projection_input_stride, uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v32_indexer_weight_state_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t *step_cursor, uint32_t max_steps,
    const uint16_t *projection_input, uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes);
__global__ void hf_deepseek_v32_indexer_weight_state_tokens_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t chunk_start, uint32_t chunk_tokens,
    uint32_t max_steps, const uint16_t *projection_input,
    uint32_t projection_input_stride, uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes);
__global__ void hf_deepseek_v32_indexer_query_state_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t q_lora_rank, uint32_t *step_cursor, uint32_t max_steps,
    float rope_theta, const uint16_t *qr_norm,
    uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v32_indexer_query_state_projected_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t *step_cursor,
    uint32_t max_steps, float rope_theta, const float *projected_query,
    uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v32_indexer_query_state_tokens_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t q_lora_rank, uint32_t chunk_start, uint32_t chunk_tokens,
    uint32_t max_steps, float rope_theta, const uint16_t *qr_norm,
    uint32_t qr_norm_stride, uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v32_sparse_score_kernel(
    SequenceLayerLayout layout, uint32_t *step_cursor, uint32_t max_steps,
    const uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    const uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    float *sparse_topk_score_workspace,
    uint32_t sparse_topk_score_capacity);
__global__ void hf_deepseek_v32_sparse_topk_select_kernel(
    SequenceLayerLayout layout, uint32_t *step_cursor, uint32_t max_steps,
    const uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    const uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    int32_t *sparse_topk_slots, uint32_t *sparse_topk_count,
    float *sparse_topk_score_workspace, uint32_t sparse_topk_score_capacity,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v32_sparse_topk_select_tokens_kernel(
    SequenceLayerLayout layout, uint32_t chunk_start, uint32_t chunk_tokens,
    uint32_t max_steps, const uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    const uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    int32_t *sparse_topk_slots, uint32_t sparse_topk_stride,
    uint32_t *sparse_topk_count, uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v32_sparse_topk_select_tokens_parallel_kernel(
    SequenceLayerLayout layout, uint32_t chunk_start, uint32_t chunk_tokens,
    uint32_t max_steps, const uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    const uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    int32_t *sparse_topk_slots, uint32_t sparse_topk_stride,
    uint32_t *sparse_topk_count, float *sparse_topk_score_workspace,
    uint32_t sparse_topk_score_stride, uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v3_mla_cache_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t heads, uint32_t *step_cursor,
    uint32_t max_steps, float rope_theta, const float *q,
    const float *kv_a, float *latent_output,
    const uint16_t *kv_latent_norm, uint16_t *kv_keys,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint16_t *projection_input, uint8_t *deepseek_v32_mla_kv,
    uint64_t deepseek_v32_mla_kv_offset_bytes,
    uint32_t deepseek_v32_mla_kv_block_count,
    const uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    const uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_rms_norm_encoded_tokens_kernel(
    uint16_t *arena, uint64_t weight_offset, const uint16_t *input,
    uint32_t weight_dtype, uint32_t input_dtype, uint32_t output_dtype,
    uint32_t rows, uint32_t input_stride, uint32_t output_stride,
    uint32_t tokens, float rms_eps, uint16_t *output);
__global__ void hf_deepseek_rms_norm_f32_tokens_kernel(
    uint16_t *arena, uint64_t weight_offset, const float *input,
    uint32_t weight_dtype, uint32_t output_dtype, uint32_t rows,
    uint32_t input_stride, uint32_t output_stride, uint32_t tokens,
    float rms_eps, uint16_t *output);
__global__ void hf_deepseek_v3_mla_cache_encode_tokens_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t chunk_start, uint32_t chunk_tokens,
    uint32_t max_steps, float rope_theta, const float *kv_a_tokens,
    uint32_t kv_a_stride, const uint16_t *kv_latent_norm_tokens,
    uint32_t kv_latent_norm_stride, uint16_t *kv_keys,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint8_t *deepseek_v32_mla_kv,
    uint64_t deepseek_v32_mla_kv_offset_bytes,
    uint32_t deepseek_v32_mla_kv_block_count);
__global__ void hf_deepseek_mla_q_latent_tokens_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t heads, uint32_t *step_cursor, uint32_t max_steps,
    uint32_t chunk_start, uint32_t token_count, float rope_theta,
    const float *q_tokens, uint32_t q_stride, uint16_t *q_latent);
__global__ void hf_deepseek_mla_fa_tile_kernel(
    SequenceLayerLayout layout, uint32_t layer_index, uint32_t heads,
    uint32_t *step_cursor, uint32_t max_steps, uint32_t chunk_start,
    uint32_t token_count, const uint16_t *q_latent, const uint16_t *kv_keys,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint16_t *attn_latent, const int32_t *sparse_topk_slots,
    uint32_t sparse_topk_stride, const uint32_t *sparse_topk_count,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_mla_fa_generic_kernel(
    SequenceLayerLayout layout, uint32_t layer_index, uint32_t dtype,
    uint32_t heads, uint32_t *step_cursor, uint32_t max_steps,
    uint32_t chunk_start, uint32_t token_count, const uint16_t *q_latent,
    const uint16_t *kv_keys, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint16_t *attn_latent,
    const int32_t *sparse_topk_slots, uint32_t sparse_topk_stride,
    const uint32_t *sparse_topk_count,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_mla_v_proj_tokens_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t heads, uint32_t *step_cursor, uint32_t max_steps,
    uint32_t chunk_start, uint32_t token_count, const uint16_t *attn_latent,
    uint16_t *attn_out, uint32_t attn_stride,
    uint64_t *deepseek_runtime_counters, uint32_t record_sparse_attention);
__global__ void hf_deepseek_residual_mlp_norm_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t norm_weight_dtype, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input);
__global__ void hf_deepseek_v3_sparse_moe_route_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch, uint16_t *projection_input,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v3_sparse_moe_router_logits_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch, const uint16_t *projection_input);
__global__ void hf_deepseek_v4_sparse_moe_route_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, uint32_t vocab_size,
    const uint32_t *prompt_tokens, uint32_t prompt_token_count,
    const NervaCudaSyntheticTokenSlot *slots, float *scratch,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v4_sparse_moe_router_logits_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float *scratch);
__global__ void hf_deepseek_v3_sparse_moe_expert_gate_up_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t rank, uint32_t *step_cursor,
    uint32_t max_steps, float *scratch);
__global__ void hf_deepseek_v3_sparse_moe_expert_down_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t rank, uint32_t *step_cursor,
    uint32_t max_steps, float *scratch);
__global__ void hf_deepseek_v3_sparse_moe_shared_gate_up_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch);
__global__ void hf_deepseek_v3_sparse_moe_shared_down_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch);
__global__ void hf_deepseek_v4_sparse_moe_expert_gate_up_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t rank, uint32_t *step_cursor, uint32_t max_steps, float *scratch);
__global__ void hf_deepseek_v4_sparse_moe_expert_down_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t rank, uint32_t *step_cursor, uint32_t max_steps, float *scratch);
__global__ void hf_deepseek_sparse_moe_reduce_down_kernel(
    SequenceLayerLayout layout, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float *scratch);
__global__ void hf_deepseek_prefill_sparse_moe_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t intermediate, uint32_t chunk_tokens,
    const uint16_t *norm_in, float *gate_up_tmp, float *down_out,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_ff_encode_kernel(
    SequenceLayerLayout layout, uint32_t dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t active_intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch, uint16_t *projection_input);
__global__ void hf_deepseek_prefill_ff_split_kernel(
    SequenceLayerLayout layout, uint32_t dtype, uint32_t intermediate,
    uint32_t chunk_tokens, const float *gate, const float *up,
    uint16_t *ff_out);
__global__ void hf_deepseek_accumulate_residual_down_kernel(
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch);
__global__ void hf_deepseek_v4_swa_dense_layer_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rms_eps, float rope_theta, float *scratch, uint16_t *kv_keys,
    uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint16_t *projection_input,
    uint8_t *deepseek_swa_kv, uint64_t deepseek_swa_kv_offset_bytes,
    uint32_t deepseek_swa_kv_block_count,
    float *deepseek_compressor_state,
    uint64_t deepseek_compressor_state_offset_bytes,
    uint8_t *deepseek_compressed_kv,
    uint64_t deepseek_compressed_kv_offset_bytes,
    uint32_t deepseek_compressed_kv_block_count,
    float *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    float *deepseek_mhc_residual, float *deepseek_mhc_post_mix,
    float *deepseek_mhc_comb_mix,
    uint64_t *deepseek_runtime_counters, uint32_t local_window_tokens,
    uint32_t preprojected_qk, uint32_t precomputed_compressor_state,
    uint32_t precomputed_indexer_state, uint32_t skip_attention);
__global__ void hf_deepseek_v4_swa_attention_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rope_theta, float *scratch, uint16_t *kv_keys,
    uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint8_t *deepseek_swa_kv,
    uint64_t deepseek_swa_kv_offset_bytes,
    uint32_t deepseek_swa_kv_block_count,
    uint64_t *deepseek_runtime_counters, uint32_t local_window_tokens);
__global__ void hf_deepseek_v4_compressed_attention_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rope_theta, float *scratch, uint16_t *kv_keys,
    uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint8_t *deepseek_swa_kv,
    uint64_t deepseek_swa_kv_offset_bytes,
    uint32_t deepseek_swa_kv_block_count,
    uint8_t *deepseek_compressed_kv,
    uint64_t deepseek_compressed_kv_offset_bytes,
    uint32_t deepseek_compressed_kv_block_count,
    const int32_t *sparse_topk_slots, const uint32_t *sparse_topk_count,
    uint64_t *deepseek_runtime_counters, uint32_t local_window_tokens);
__global__ void hf_deepseek_v4_compressed_indexer_sparse_topk_select_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rope_theta, float *scratch, const uint16_t *projection_input,
    uint8_t *deepseek_compressed_kv,
    uint64_t deepseek_compressed_kv_offset_bytes,
    uint32_t deepseek_compressed_kv_block_count,
    uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    int32_t *sparse_topk_slots, uint32_t *sparse_topk_count,
    float *sparse_topk_score_workspace, uint32_t sparse_topk_score_capacity,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v4_q_a_norm_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t hidden,
    uint32_t heads, uint32_t head_dim, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps,
    float *scratch);
__global__ void hf_deepseek_v4_finalize_preprojected_qk_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rms_eps, float rope_theta, float *scratch);
__global__ void hf_deepseek_v4_compressor_state_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t head_dim, uint32_t *step_cursor,
    uint32_t max_steps, const uint16_t *projection_input,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    float *deepseek_compressor_state,
    uint64_t deepseek_compressor_state_offset_bytes,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v4_compressed_kv_write_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t head_dim,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps,
    float rope_theta, float *deepseek_compressor_state,
    uint64_t deepseek_compressor_state_offset_bytes,
    uint8_t *deepseek_compressed_kv,
    uint64_t deepseek_compressed_kv_offset_bytes,
    uint32_t deepseek_compressed_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v4_indexer_kv_write_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float rope_theta,
    float *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v4_indexer_state_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t *step_cursor, uint32_t max_steps,
    const uint16_t *projection_input, uint32_t kv_block_count,
    const uint32_t *kv_block_table, float *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    uint64_t *deepseek_runtime_counters);
__global__ void hf_deepseek_v4_attn_mhc_pre_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t layer_index, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input, float *deepseek_mhc_residual,
    float *deepseek_mhc_post_mix, float *deepseek_mhc_comb_mix);
__global__ void hf_deepseek_v4_ffn_mhc_pre_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rms_eps, float *scratch, uint16_t *projection_input,
    float *deepseek_mhc_residual, float *deepseek_mhc_post_mix,
    float *deepseek_mhc_comb_mix);
__global__ void hf_deepseek_v4_finish_final_norm_encode_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, SequenceLayerLayout layout,
    uint32_t dtype, uint32_t final_norm_weight_dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input, float *deepseek_mhc_residual,
    float *deepseek_mhc_post_mix, float *deepseek_mhc_comb_mix);
