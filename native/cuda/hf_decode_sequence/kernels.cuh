#pragma once

#include "types.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

__global__ void hf_deinterleave_q_gate_projection_kernel(
    const uint16_t *packed, uint16_t *q, uint16_t *q_gate,
    uint32_t heads, uint32_t head_dim, uint32_t hidden);

void launch_hf_layer_attention_chunk_kernel(
    cudaStream_t stream, dim3 grid, uint32_t dtype,
    bool use_shared_warp_gqa, bool use_grouped_gqa, uint32_t dense_threads,
    uint32_t layer_index, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, uint32_t attention_chunks, float *scratch,
    const uint16_t *kv_keys, const uint16_t *kv_values, float *partial_values,
    float *partial_m, float *partial_l, uint32_t kv_block_count,
    const uint32_t *kv_block_table, const uint32_t *selected_chunks,
    uint32_t qk_fused_selector, uint32_t qk_local_window_tokens,
    uint32_t qk_sink_tokens);
void launch_hf_experimental_qk_page_selector_kernel(
    cudaStream_t stream, dim3 grid, uint32_t dtype, uint32_t layer_index,
    uint32_t hidden, uint32_t heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    uint32_t selected_pages, uint32_t local_window_tokens,
    uint32_t sink_tokens, float *scratch, const uint16_t *kv_keys,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint32_t *candidate_pages);
void launch_hf_prefill_grouped_gqa_attention_direct_kernel(
    cudaStream_t stream, dim3 grid, uint32_t dtype, uint32_t layer_index,
    uint32_t heads, uint32_t kv_heads, uint32_t head_dim, uint32_t max_steps,
    uint32_t chunk_start, uint32_t chunk_tokens, const float *qkv,
    const uint16_t *kv_keys, const uint16_t *kv_values,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint16_t *attn_out, uint32_t local_window_tokens);

__global__ void hf_decode_final_head_rows_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t dtype,
    uint32_t hidden, uint32_t vocab_size, const uint32_t *step_cursor,
    uint32_t max_steps, const float *scratch, float *scores);
__global__ void hf_decode_sequence_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, SequenceLayerLayout *layers,
    uint32_t layer_count, uint32_t dtype, uint32_t hidden, uint32_t heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t intermediate, uint32_t position,
    uint32_t *step_cursor, uint32_t max_steps, const uint32_t *prompt_tokens,
    uint32_t prompt_token_count, float rms_eps, float rope_theta, float *scratch,
    uint16_t *kv_keys, uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, const NervaCudaSyntheticTokenSlot *slots,
    float *linear_gdn_conv_state, float *linear_gdn_recurrent_state);
__global__ void hf_decode_prepare_input_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t hidden,
    uint32_t *step_cursor, uint32_t max_steps, const uint32_t *prompt_tokens,
    uint32_t prompt_token_count, const NervaCudaSyntheticTokenSlot *slots);
__global__ void hf_decode_prepare_first_attn_norm_encode_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout,
    SequenceLayerLayout first_layout, uint32_t dtype, uint32_t hidden,
    uint32_t norm_weight_dtype, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, const uint32_t *prompt_tokens,
    uint32_t prompt_token_count, const NervaCudaSyntheticTokenSlot *slots,
    float rms_eps, float *scratch, uint16_t *projection_input);

__global__ void hf_layer_attn_norm_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint64_t input_offset,
    uint32_t dtype, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input);
__global__ void hf_decode_rms_norm_f32_to_encoded_kernel(
    uint16_t *arena, uint64_t weight_offset, const float *input,
    uint32_t weight_dtype, uint32_t output_dtype, uint32_t hidden,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps,
    uint16_t *projection_input);
__global__ void hf_layer_attention_encode_kernel(
    uint32_t layer_index, uint32_t dtype, uint32_t hidden, uint32_t heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float *scratch,
    const uint16_t *kv_keys, const uint16_t *kv_values,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint16_t *projection_input);
__global__ void hf_deepseek_v3_mla_attention_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t heads, uint32_t *step_cursor,
    uint32_t max_steps, float rope_theta, const float *q,
    const float *kv_a, float *latent_output,
    const uint16_t *kv_latent_norm, uint16_t *kv_keys,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint16_t *projection_input);
__global__ void hf_layer_qkv_attention_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float rope_theta, float *scratch,
    uint16_t *kv_keys, uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint16_t *projection_input);
__global__ void hf_layer_qkv_prepare_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float rope_theta, float *scratch,
    uint16_t *kv_keys, uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint16_t *decode_q,
    int32_t *decode_seq_len_q, int32_t *decode_seq_len_kv);
__global__ void hf_layer_attention_reduce_kernel(
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, uint32_t attention_chunks, float *scratch,
    const float *partial_values, const float *partial_m, const float *partial_l,
    uint16_t *projection_input);
__global__ void hf_layer_query_gate_attention_encode_kernel(
    uint32_t dtype, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float *scratch, uint16_t *projection_input);
__global__ void hf_layer_mlp_norm_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input);
__global__ void hf_deepseek_residual_mlp_norm_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t norm_weight_dtype, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input);
__global__ void hf_layer_ff_encode_kernel(
    uint32_t dtype, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float *scratch, uint16_t *projection_input);
__global__ void hf_layer_sparse_moe_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch, uint16_t *projection_input);
__global__ void hf_deepseek_v3_sparse_moe_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch, uint16_t *projection_input);
__global__ void hf_deepseek_v4_swa_dense_layer_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rms_eps, float rope_theta, float *scratch, uint16_t *kv_keys,
    uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint16_t *projection_input);
__global__ void hf_layer_finish_kernel(
    uint16_t *arena, uint64_t output_offset, uint32_t dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float *scratch);
__global__ void hf_layer_finish_next_attn_norm_encode_kernel(
    uint16_t *arena, uint64_t output_offset, SequenceLayerLayout next_layout,
    uint32_t dtype, uint32_t next_norm_weight_dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input);
__global__ void hf_layer_finish_final_norm_encode_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t dtype,
    uint32_t final_norm_weight_dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input);
__global__ void hf_final_norm_encode_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint64_t input_offset,
    uint32_t dtype, uint32_t hidden, uint32_t *step_cursor, uint32_t max_steps,
    float rms_eps, float *scratch, uint16_t *projection_input);

__global__ void hf_prefill_embed_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t hidden,
    const uint32_t *prompt_tokens, uint32_t prompt_token_count,
    uint16_t *hidden_out);
__global__ void hf_prefill_embed_range_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t hidden,
    const uint32_t *tokens, uint32_t token_count, uint32_t output_start,
    uint16_t *hidden_out);
__global__ void hf_prefill_attn_norm_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t chunk_start, uint32_t chunk_tokens,
    float rms_eps, const uint16_t *hidden_in, uint16_t *norm_out);
__global__ void hf_prefill_qkv_publish_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t max_steps, uint32_t chunk_start, uint32_t chunk_tokens,
    float rms_eps, float rope_theta, float *qkv, uint16_t *kv_keys,
    uint16_t *kv_values, uint16_t *qkv_encoded, uint32_t kv_block_count,
    const uint32_t *kv_block_table);
__global__ void hf_prefill_attention_kernel(
    uint32_t layer_index, uint32_t dtype, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t max_steps, uint32_t chunk_start,
    uint32_t chunk_tokens, const float *qkv, const uint16_t *kv_keys,
    const uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint16_t *attn_out,
    uint32_t local_window_tokens);
__global__ void hf_prefill_mlp_norm_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t chunk_start, uint32_t chunk_tokens,
    float rms_eps, const uint16_t *hidden_in, float *attn_projection,
    uint16_t *norm_out);
__global__ void hf_prefill_ff_kernel(uint32_t dtype, uint32_t intermediate,
                                     uint32_t chunk_tokens,
                                     const float *gate_up,
                                     uint16_t *ff_out);
__global__ void hf_prefill_query_gate_attention_kernel(
    uint32_t dtype, uint32_t attention_hidden, uint32_t chunk_tokens,
    const float *q_gate, uint16_t *attn_out);
__global__ void hf_prefill_sparse_moe_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t intermediate, uint32_t chunk_tokens,
    const uint16_t *norm_in, float *gate_up_tmp, float *down_out);
__global__ void hf_prefill_finish_kernel(uint32_t dtype, uint32_t hidden,
                                         uint32_t chunk_start,
                                         uint32_t chunk_tokens,
                                         const float *residual,
                                         const float *down,
                                         uint16_t *hidden_out);
__global__ void hf_prefill_final_norm_last_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t dtype,
    uint32_t hidden, uint32_t prompt_token_count, float rms_eps,
    const uint16_t *hidden_in, uint16_t *projection_input);
__global__ void hf_prefill_final_norm_range_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t dtype,
    uint32_t hidden, uint32_t chunk_start, uint32_t chunk_tokens,
    float rms_eps, const uint16_t *hidden_in, uint16_t *projection_input);

__global__ void hf_decode_set_step_kernel(uint32_t *step_cursor,
                                          uint32_t value);
__global__ void hf_init_identity_kv_block_table_kernel(uint32_t *block_table,
                                                       uint32_t block_count);

__global__ void hf_pack_qkv_weights_kernel(
    uint16_t *packed, const uint16_t *arena,
    const SequenceLayerLayout *layouts, uint32_t layer_count, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden);
__global__ void hf_pack_gate_up_weights_kernel(
    uint16_t *packed, const uint16_t *arena,
    const SequenceLayerLayout *layouts, uint32_t layer_count, uint32_t hidden,
    uint32_t intermediate);

__global__ void hf_projection_batch_pack_u16_kernel(
    const uint16_t *src, uint16_t *dst, uint32_t cols, uint32_t token_index);
__global__ void hf_projection_batch_pack_small_u16_kernel(
    const uint16_t *src0, const uint16_t *src1, const uint16_t *src2,
    const uint16_t *src3, const uint16_t *src4, const uint16_t *src5,
    const uint16_t *src6, const uint16_t *src7, const uint16_t *src8,
    const uint16_t *src9, const uint16_t *src10, const uint16_t *src11,
    const uint16_t *src12, const uint16_t *src13, const uint16_t *src14,
    const uint16_t *src15, const uint16_t *src16, const uint16_t *src17,
    const uint16_t *src18, const uint16_t *src19, const uint16_t *src20,
    const uint16_t *src21, const uint16_t *src22, const uint16_t *src23,
    const uint16_t *src24, const uint16_t *src25, const uint16_t *src26,
    const uint16_t *src27, const uint16_t *src28, const uint16_t *src29,
    const uint16_t *src30, const uint16_t *src31, uint16_t *dst,
    uint32_t cols, uint32_t tokens);
__global__ void hf_projection_batch_scatter_f32_kernel(
    const float *src, float *dst, uint32_t rows, uint32_t token_index);
__global__ void hf_projection_batch_scatter_small_f32_kernel(
    const float *src, float *dst0, float *dst1, float *dst2, float *dst3,
    float *dst4, float *dst5, float *dst6, float *dst7, float *dst8,
    float *dst9, float *dst10, float *dst11, float *dst12, float *dst13,
    float *dst14, float *dst15, float *dst16, float *dst17, float *dst18,
    float *dst19, float *dst20, float *dst21, float *dst22, float *dst23,
    float *dst24, float *dst25, float *dst26, float *dst27, float *dst28,
    float *dst29, float *dst30, float *dst31, uint32_t rows,
    uint32_t tokens);
