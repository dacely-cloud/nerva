#include "kernels.cuh"

#include "../deepseek_quant.cuh"
#include "../deepseek_router.cuh"
#include "device_ops.cuh"

#include <stdint.h>



__global__ void hf_deinterleave_q_gate_projection_kernel(
    const uint16_t *packed, uint16_t *q, uint16_t *q_gate,
    uint32_t heads, uint32_t head_dim, uint32_t hidden) {
  const uint64_t head_elements =
      static_cast<uint64_t>(head_dim) * static_cast<uint64_t>(hidden);
  const uint64_t total = static_cast<uint64_t>(heads) * head_elements;
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t stride =
      static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);
  for (uint64_t index = start; index < total; index += stride) {
    const uint64_t head = index / head_elements;
    const uint64_t within = index - head * head_elements;
    const uint64_t packed_base = head * head_elements * 2u;
    q[index] = packed[packed_base + within];
    q_gate[index] = packed[packed_base + head_elements + within];
  }
}









__global__ void hf_decode_final_head_rows_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t dtype,
    uint32_t hidden, uint32_t vocab_size, const uint32_t *step_cursor,
    uint32_t max_steps, const float *scratch, float *scores) {
  const uint32_t row = blockIdx.x;
  if (row >= vocab_size || (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  const uint16_t *lm_head = arena + arena_layout.lm_head;
  const float *final_norm = scratch + hidden;
  float sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    sum += encoded_to_f32(lm_head[static_cast<uint64_t>(row) * hidden + col], dtype) *
           final_norm[col];
  }
  sum = block_sum(sum);
  if (threadIdx.x == 0) {
    scores[row] = sum;
  }
}

__global__ void hf_decode_sequence_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, SequenceLayerLayout *layers,
    uint32_t layer_count, uint32_t dtype, uint32_t hidden, uint32_t heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t intermediate, uint32_t position,
    uint32_t *step_cursor, uint32_t max_steps, const uint32_t *prompt_tokens,
    uint32_t prompt_token_count, float rms_eps, float rope_theta, float *scratch,
    uint16_t *kv_keys, uint16_t *kv_values,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    const NervaCudaSyntheticTokenSlot *slots,
    float *linear_gdn_conv_state, float *linear_gdn_recurrent_state) {
  if (blockIdx.x != 0) {
    return;
  }
  __shared__ uint32_t current_position_shared;
  __shared__ uint32_t current_token_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? position : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  (void)max_steps;
  if (threadIdx.x == 0) {
    current_token_shared = current_position < prompt_token_count
                               ? prompt_tokens[current_position]
                               : slots[current_position - 1].token;
  }
  __syncthreads();
  const uint32_t current_token = current_token_shared;
  const uint64_t embedding_offset = arena_layout.embeddings +
                                    static_cast<uint64_t>(current_token) * hidden;
  copy_encoded_slice(arena + arena_layout.input, arena + embedding_offset, hidden);

  uint64_t input_offset = arena_layout.input;
  uint64_t output_offset = arena_layout.scratch;
  for (uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
    run_layer(arena, layers[layer_index], layer_index, input_offset, output_offset,
              dtype, hidden, heads, kv_heads, head_dim, intermediate,
              current_position, max_steps, rms_eps, rope_theta, scratch, kv_keys,
              kv_values, kv_block_count, kv_block_table,
              linear_gdn_conv_state, linear_gdn_recurrent_state);
    const uint64_t next_input = output_offset;
    output_offset = input_offset;
    input_offset = next_input;
  }

  float *decoded = scratch;
  float *final_norm = decoded + hidden;
  encoded_slice_to_f32(arena + input_offset, hidden, dtype, decoded);
  rms_norm(decoded, arena + arena_layout.final_norm, hidden, dtype, rms_eps, final_norm);
  f32_slice_to_encoded(final_norm, arena + arena_layout.input, hidden, dtype);
}

__global__ void hf_decode_prepare_input_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t hidden,
    uint32_t *step_cursor, uint32_t max_steps, const uint32_t *prompt_tokens,
    uint32_t prompt_token_count, const NervaCudaSyntheticTokenSlot *slots) {
  __shared__ uint32_t current_position_shared;
  __shared__ uint32_t current_token_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  (void)max_steps;
  if (threadIdx.x == 0) {
    current_token_shared = current_position < prompt_token_count
                               ? prompt_tokens[current_position]
                               : slots[current_position - 1].token;
  }
  __syncthreads();
  const uint32_t current_token = current_token_shared;
  const uint64_t embedding_offset = arena_layout.embeddings +
                                    static_cast<uint64_t>(current_token) * hidden;
  copy_encoded_slice(arena + arena_layout.input, arena + embedding_offset, hidden);
}

__global__ void hf_layer_attn_norm_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint64_t input_offset,
    uint32_t dtype, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  encoded_slice_to_f32(arena + input_offset, hidden, dtype, s.input);
  rms_norm_to_encoded(s.input, arena + layout.rms_attn, hidden, dtype, rms_eps,
                      projection_input);
}

__global__ void hf_decode_rms_norm_f32_to_encoded_kernel(
    uint16_t *arena, uint64_t weight_offset, const float *input,
    uint32_t weight_dtype, uint32_t output_dtype, uint32_t hidden,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps,
    uint16_t *projection_input) {
  if ((step_cursor != nullptr && *step_cursor >= max_steps) ||
      arena == nullptr || input == nullptr || projection_input == nullptr ||
      hidden == 0 || weight_dtype > kDTypeF32 || output_dtype > kDTypeBF16 ||
      weight_offset == kMissingOffset) {
    return;
  }
  rms_norm_to_encoded_with_weight_dtype(input, arena + weight_offset, hidden,
                                        weight_dtype, output_dtype, rms_eps,
                                        projection_input);
}

__global__ void hf_decode_prepare_first_attn_norm_encode_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout,
    SequenceLayerLayout first_layout, uint32_t dtype, uint32_t hidden,
    uint32_t norm_weight_dtype, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, const uint32_t *prompt_tokens,
    uint32_t prompt_token_count, const NervaCudaSyntheticTokenSlot *slots,
    float rms_eps, float *scratch, uint16_t *projection_input) {
  __shared__ uint32_t current_position_shared;
  __shared__ uint32_t current_token_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  (void)max_steps;
  if (threadIdx.x == 0) {
    current_token_shared = current_position < prompt_token_count
                               ? prompt_tokens[current_position]
                               : slots[current_position - 1].token;
  }
  __syncthreads();
  const uint32_t current_token = current_token_shared;
  const uint64_t embedding_offset = arena_layout.embeddings +
                                    static_cast<uint64_t>(current_token) * hidden;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    const uint16_t encoded = arena[embedding_offset + index];
    s.input[index] = encoded_to_f32(encoded, dtype);
  }
  __syncthreads();
  rms_norm_to_encoded_with_weight_dtype(s.input, arena + first_layout.rms_attn,
                                        hidden, norm_weight_dtype, dtype,
                                        rms_eps, projection_input);
}

__global__ void hf_layer_attention_encode_kernel(
    uint32_t layer_index, uint32_t dtype, uint32_t hidden, uint32_t heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float *scratch,
    const uint16_t *kv_keys, const uint16_t *kv_values,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint16_t *projection_input) {
  __shared__ uint32_t current_position_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  (void)max_steps;
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint32_t head = blockIdx.x;
  if (head >= heads) {
    return;
  }
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const uint32_t kv_head = head / (heads / kv_heads);
  const uint32_t head_start = head * head_dim;
  for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
    s.attn[head_start + offset] = 0.0f;
  }
  __syncthreads();

  float local_m = -INFINITY;
  float local_l = 0.0f;
  for (uint32_t token = 0; token <= current_position; ++token) {
    const uint64_t token_base = kv_cache_token_base(
        layer_index, kv_block_count, kv_block_table, token, kv_hidden,
        kv_head * head_dim);
    float partial = 0.0f;
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      partial += s.q[head_start + offset] *
                 encoded_to_f32(kv_keys[token_base + offset], dtype);
    }
    const float score = block_sum(partial) * scale;
    const float next_m = fmaxf(local_m, score);
    const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
    const float new_scale = expf(score - next_m);
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      const uint32_t out = head_start + offset;
      s.attn[out] =
          s.attn[out] * old_scale +
          encoded_to_f32(kv_values[token_base + offset], dtype) * new_scale;
    }
    local_l = local_l * old_scale + new_scale;
    local_m = next_m;
  }
  const bool normalize = local_l > 0.0f && isfinite(local_l);
  for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
    const uint32_t out = head_start + offset;
    if (normalize) {
      s.attn[out] /= local_l;
    }
    projection_input[out] = f32_to_encoded(s.attn[out], dtype);
  }
}

__device__ float deepseek_fp8_scaled_weight(const uint16_t *arena,
                                            uint64_t weight_offset,
                                            uint64_t scale_offset,
                                            uint32_t rows, uint32_t cols,
                                            uint32_t row, uint32_t col) {
  const auto *weights = reinterpret_cast<const uint8_t *>(arena + weight_offset);
  const uint32_t scale_cols = (cols + 127u) / 128u;
  const uint32_t scale_idx = (row / 128u) * scale_cols + (col / 128u);
  return nerva::deepseek::f8_e4m3fn_bits_to_f32(
             weights[static_cast<uint64_t>(row) * cols + col]) *
         f32_from_u16_slots(arena + scale_offset, scale_idx);
}

__device__ float deepseek_fp8_e8m0_scaled_weight(const uint16_t *arena,
                                                 uint64_t weight_offset,
                                                 uint64_t scale_offset,
                                                 uint32_t rows, uint32_t cols,
                                                 uint32_t row, uint32_t col) {
  const auto *weights = reinterpret_cast<const uint8_t *>(arena + weight_offset);
  const auto *scales = reinterpret_cast<const uint8_t *>(arena + scale_offset);
  const uint32_t scale_cols = (cols + 127u) / 128u;
  const uint32_t scale_idx = (row / 128u) * scale_cols + (col / 128u);
  return nerva::deepseek::f8_e4m3fn_bits_to_f32(
             weights[static_cast<uint64_t>(row) * cols + col]) *
         nerva::deepseek::e8m0_exponent_bits_to_f32(scales[scale_idx]);
}

__device__ __forceinline__ uint64_t deepseek_device_fp8_slots(
    uint64_t rows, uint64_t cols) {
  return (rows * cols + 1u) / 2u;
}

__device__ __forceinline__ uint64_t deepseek_device_byte_slots(
    uint64_t values) {
  return (values + 1u) / 2u;
}

__device__ __forceinline__ uint64_t deepseek_device_rank3_slots(
    uint64_t depth, uint64_t rows, uint64_t cols) {
  return deepseek_device_byte_slots(depth * rows * cols);
}

__device__ __forceinline__ uint64_t deepseek_u64_from_u16_slots(
    const uint16_t *slots, uint64_t index) {
  const uint64_t base = index * 4u;
  return static_cast<uint64_t>(slots[base]) |
         (static_cast<uint64_t>(slots[base + 1u]) << 16u) |
         (static_cast<uint64_t>(slots[base + 2u]) << 32u) |
         (static_cast<uint64_t>(slots[base + 3u]) << 48u);
}

__device__ __forceinline__ uint32_t deepseek_device_scale_dim(
    uint32_t value) {
  return (value + 127u) / 128u;
}

__device__ __forceinline__ uint64_t deepseek_device_scale_f32_slots(
    uint32_t rows, uint32_t cols) {
  return static_cast<uint64_t>(deepseek_device_scale_dim(rows)) *
         deepseek_device_scale_dim(cols) * 2u;
}

__device__ float deepseek_fp8_rank3_scaled_weight(
    const uint16_t *arena, uint64_t weight_offset, uint64_t scale_offset,
    uint32_t rows, uint32_t cols, uint32_t expert, uint32_t row,
    uint32_t col) {
  const auto *weights = reinterpret_cast<const uint8_t *>(arena + weight_offset);
  const uint64_t weight_idx =
      (static_cast<uint64_t>(expert) * rows + row) * cols + col;
  const uint32_t scale_rows = deepseek_device_scale_dim(rows);
  const uint32_t scale_cols = deepseek_device_scale_dim(cols);
  const uint64_t scale_idx =
      (static_cast<uint64_t>(expert) * scale_rows + (row / 128u)) *
          scale_cols +
      (col / 128u);
  return nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[weight_idx]) *
         f32_from_u16_slots(arena + scale_offset, scale_idx);
}

__device__ float deepseek_mxfp4_rank3_scaled_weight(
    const uint16_t *arena, uint64_t weight_offset, uint64_t scale_offset,
    uint32_t rows, uint32_t packed_cols, uint32_t expert, uint32_t row,
    uint32_t col) {
  const auto *weights = reinterpret_cast<const uint8_t *>(arena + weight_offset);
  const auto *scales = reinterpret_cast<const uint8_t *>(arena + scale_offset);
  const uint32_t packed_col = col >> 1u;
  const uint8_t byte =
      weights[(static_cast<uint64_t>(expert) * rows + row) * packed_cols +
              packed_col];
  const uint8_t nibble =
      (col & 1u) == 0 ? (byte & 0x0fu) : (byte >> 4u);
  const uint32_t scale_cols = (packed_cols + 15u) / 16u;
  const uint64_t scale_idx =
      (static_cast<uint64_t>(expert) * rows + row) * scale_cols +
      (packed_col / 16u);
  return nerva::deepseek::mxfp4_e2m1_nibble_to_f32(nibble) *
         nerva::deepseek::e8m0_exponent_bits_to_f32(scales[scale_idx]);
}

__device__ float deepseek_rope_value_serial(float left, float right,
                                            uint32_t offset, uint32_t dim,
                                            uint32_t position, float theta,
                                            bool second) {
  if (theta <= 0.0f || dim < 2) {
    return second ? right : left;
  }
  const float exponent =
      static_cast<float>(2u * offset) / static_cast<float>(dim);
  const float angle = static_cast<float>(position) / powf(theta, exponent);
  float sin_value = 0.0f;
  float cos_value = 0.0f;
  sincosf(angle, &sin_value, &cos_value);
  return second ? right * cos_value + left * sin_value
                : left * cos_value - right * sin_value;
}

__global__ void hf_deepseek_v3_mla_attention_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t heads, uint32_t *step_cursor,
    uint32_t max_steps, float rope_theta, const float *q,
    const float *kv_a, float *latent_output,
    const uint16_t *kv_latent_norm, uint16_t *kv_keys,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint16_t *projection_input) {
  if (blockIdx.x != 0 || threadIdx.x != 0 ||
      (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t q_lora_rank = layout.deepseek_q_lora_rank;
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t v_head = layout.deepseek_v_head_dim;
  const uint32_t qk_head_dim = qk_nope + qk_rope;
  const uint32_t kv_cache_width = kv_lora_rank + qk_rope;
  (void)q_lora_rank;
  if (heads == 0 || kv_lora_rank == 0 || qk_nope == 0 || qk_rope == 0 ||
      v_head == 0 || qk_head_dim == 0 ||
      layout.w_v == kMissingOffset ||
      layout.deepseek_kv_b_scale == kMissingOffset) {
    return;
  }

  const uint64_t write_base =
      kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                          position, kv_cache_width, 0);
  for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
    kv_keys[write_base + latent] = kv_latent_norm[latent];
  }
  const uint32_t rope_half = qk_rope / 2u;
  for (uint32_t dim = 0; dim < qk_rope; ++dim) {
    float value = kv_a[kv_lora_rank + dim];
    if (rope_half != 0) {
      const uint32_t offset = dim % rope_half;
      const uint32_t pair = dim < rope_half ? dim + rope_half : dim - rope_half;
      value = deepseek_rope_value_serial(
          kv_a[kv_lora_rank + offset], kv_a[kv_lora_rank + offset + rope_half],
          offset, qk_rope, position, rope_theta, dim >= rope_half);
      (void)pair;
    }
    kv_keys[write_base + kv_lora_rank + dim] = f32_to_encoded(value, dtype);
  }

  const float softmax_scale = rsqrtf(static_cast<float>(qk_head_dim));
  const uint32_t kv_b_cols = kv_lora_rank;
  const uint32_t kv_b_rows = heads * (qk_nope + v_head);
  for (uint32_t head = 0; head < heads; ++head) {
    for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
      latent_output[latent] = 0.0f;
    }

    float local_m = -INFINITY;
    float local_l = 0.0f;
    for (uint32_t token = 0; token <= position; ++token) {
      const uint64_t token_base =
          kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                              token, kv_cache_width, 0);
      float score = 0.0f;
      for (uint32_t nope = 0; nope < qk_nope; ++nope) {
        const uint32_t row = head * (qk_nope + v_head) + nope;
        const float q_value = q[head * qk_head_dim + nope];
        for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
          score += q_value *
                   deepseek_fp8_scaled_weight(arena, layout.w_v,
                                              layout.deepseek_kv_b_scale,
                                              kv_b_rows, kv_b_cols, row,
                                              latent) *
                   encoded_to_f32(kv_keys[token_base + latent], dtype);
        }
      }
      const uint32_t q_pe_base = head * qk_head_dim + qk_nope;
      for (uint32_t dim = 0; dim < qk_rope; ++dim) {
        float q_pe = q[q_pe_base + dim];
        if (rope_half != 0) {
          const uint32_t offset = dim % rope_half;
          q_pe = deepseek_rope_value_serial(
              q[q_pe_base + offset], q[q_pe_base + offset + rope_half],
              offset, qk_rope, position, rope_theta, dim >= rope_half);
        }
        score += q_pe *
                 encoded_to_f32(kv_keys[token_base + kv_lora_rank + dim],
                                dtype);
      }
      score *= softmax_scale;
      const float next_m = fmaxf(local_m, score);
      const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
      const float new_scale = expf(score - next_m);
      for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
        latent_output[latent] =
            latent_output[latent] * old_scale +
            encoded_to_f32(kv_keys[token_base + latent], dtype) * new_scale;
      }
      local_l = local_l * old_scale + new_scale;
      local_m = next_m;
    }

    if (local_l > 0.0f && isfinite(local_l)) {
      for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
        latent_output[latent] /= local_l;
      }
    }
    for (uint32_t value = 0; value < v_head; ++value) {
      const uint32_t row = head * (qk_nope + v_head) + qk_nope + value;
      float sum = 0.0f;
      for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
        sum += latent_output[latent] *
               deepseek_fp8_scaled_weight(arena, layout.w_v,
                                          layout.deepseek_kv_b_scale,
                                          kv_b_rows, kv_b_cols, row, latent);
      }
      projection_input[head * v_head + value] = f32_to_encoded(sum, dtype);
    }
  }
}

__global__ void hf_layer_qkv_attention_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float rope_theta, float *scratch,
    uint16_t *kv_keys, uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint16_t *projection_input) {
  __shared__ uint32_t current_position_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  (void)max_steps;
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint32_t head = blockIdx.x;
  if (head >= heads) {
    return;
  }
  const uint32_t group = heads / kv_heads;
  const uint32_t kv_head = head / group;
  const uint32_t head_start = head * head_dim;
  const uint32_t kv_start = kv_head * head_dim;

  float q_mean_square = 0.0f;
  for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
    const uint32_t index = head_start + offset;
    float value = s.q[index];
    if (layout.q_bias != kMissingOffset) {
      value += encoded_to_f32(arena[layout.q_bias + index], dtype);
    }
    s.q[index] = value;
    q_mean_square += value * value;
  }
  q_mean_square = block_sum(q_mean_square);
  if (layout.q_norm != kMissingOffset) {
    const float scale = rsqrtf(q_mean_square / static_cast<float>(head_dim) + rms_eps);
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      const uint32_t index = head_start + offset;
      s.q[index] *= scale * encoded_to_f32(arena[layout.q_norm + offset], dtype);
    }
  }
  __syncthreads();
  apply_rope_head(s.q + head_start, head_dim, current_position, rope_theta);

  float k_mean_square = 0.0f;
  for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
    float value = s.k[kv_start + offset];
    if (layout.k_bias != kMissingOffset) {
      value += encoded_to_f32(arena[layout.k_bias + kv_start + offset], dtype);
    }
    k_mean_square += value * value;
  }
  k_mean_square = block_sum(k_mean_square);
  const bool has_k_norm = layout.k_norm != kMissingOffset;
  const float k_scale =
      has_k_norm ? rsqrtf(k_mean_square / static_cast<float>(head_dim) + rms_eps) : 1.0f;
  const uint32_t half = head_dim / 2;
  const bool publish_kv = head % group == 0;
  const uint64_t publish_base =
      publish_kv
          ? kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                                current_position, kv_hidden, kv_start)
          : 0;
  for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
    const uint32_t rope_offset = offset < half ? offset : offset - half;
    const uint32_t left_index = kv_start + rope_offset;
    const uint32_t right_index = left_index + half;
    float left = s.k[left_index];
    float right = s.k[right_index];
    if (layout.k_bias != kMissingOffset) {
      left += encoded_to_f32(arena[layout.k_bias + left_index], dtype);
      right += encoded_to_f32(arena[layout.k_bias + right_index], dtype);
    }
    if (has_k_norm) {
      left *= k_scale * encoded_to_f32(arena[layout.k_norm + rope_offset], dtype);
      right *= k_scale *
               encoded_to_f32(arena[layout.k_norm + rope_offset + half], dtype);
    }
    float current_k = left;
    if (rope_theta > 0.0f) {
      const float exponent =
          static_cast<float>(2 * rope_offset) / static_cast<float>(head_dim);
      const float angle = static_cast<float>(current_position) / powf(rope_theta, exponent);
      float sin_value = 0.0f;
      float cos_value = 0.0f;
      sincosf(angle, &sin_value, &cos_value);
      const float rotated_left = left * cos_value - right * sin_value;
      const float rotated_right = right * cos_value + left * sin_value;
      current_k = offset < half ? rotated_left : rotated_right;
    }
    float current_v = s.v[kv_start + offset];
    if (layout.v_bias != kMissingOffset) {
      current_v += encoded_to_f32(arena[layout.v_bias + kv_start + offset], dtype);
    }
    s.residual[head_start + offset] = current_k;
    s.mlp_norm[head_start + offset] = current_v;
    if (publish_kv) {
      kv_keys[publish_base + offset] = f32_to_encoded(current_k, dtype);
      kv_values[publish_base + offset] = f32_to_encoded(current_v, dtype);
    }
  }
  __syncthreads();

  const float scale = rsqrtf(static_cast<float>(head_dim));
  for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
    s.attn[head_start + offset] = 0.0f;
  }
  __syncthreads();
  float local_m = -INFINITY;
  float local_l = 0.0f;
  for (uint32_t token = 0; token <= current_position; ++token) {
    const uint64_t token_base = kv_cache_token_base(
        layer_index, kv_block_count, kv_block_table, token, kv_hidden,
        kv_start);
    float partial = 0.0f;
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      const float key_value =
          encoded_to_f32(kv_keys[token_base + offset], dtype);
      partial += s.q[head_start + offset] * key_value;
    }
    const float score = block_sum(partial) * scale;
    const float next_m = fmaxf(local_m, score);
    const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
    const float new_scale = expf(score - next_m);
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      const uint32_t out = head_start + offset;
      const float value_value =
          encoded_to_f32(kv_values[token_base + offset], dtype);
      s.attn[out] = s.attn[out] * old_scale + value_value * new_scale;
    }
    local_l = local_l * old_scale + new_scale;
    local_m = next_m;
  }
  const bool normalize = local_l > 0.0f && isfinite(local_l);
  for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
    const uint32_t out = head_start + offset;
    if (normalize) {
      s.attn[out] /= local_l;
    }
    projection_input[out] = f32_to_encoded(s.attn[out], dtype);
  }
}

__global__ void hf_layer_qkv_prepare_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float rope_theta, float *scratch,
    uint16_t *kv_keys, uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint16_t *decode_q,
    int32_t *decode_seq_len_q, int32_t *decode_seq_len_kv) {
  __shared__ uint32_t current_position_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (decode_seq_len_q != nullptr) {
      decode_seq_len_q[0] = 1;
    }
    if (decode_seq_len_kv != nullptr) {
      const uint32_t kv_tokens =
          max_steps == 0 ? current_position + 1u
                         : min(current_position + 1u, max_steps);
      decode_seq_len_kv[0] = static_cast<int32_t>(kv_tokens);
    }
  }
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint32_t head = blockIdx.x;
  if (head < heads) {
    float *q = s.q + head * head_dim;
    add_optional_head_bias(arena, layout.q_bias, head, head_dim, dtype, q);
    per_head_rms_norm_block(arena, layout.q_norm, q, head_dim, dtype, rms_eps);
    apply_rope_head(q, head_dim, current_position, rope_theta);
    if (decode_q != nullptr) {
      const uint32_t head_start = head * head_dim;
      for (uint32_t index = threadIdx.x; index < head_dim;
           index += blockDim.x) {
        decode_q[head_start + index] = f32_to_encoded(q[index], dtype);
      }
    }
  }
  if (head < kv_heads) {
    float *k = s.k + head * head_dim;
    float *v = s.v + head * head_dim;
    add_optional_head_bias(arena, layout.k_bias, head, head_dim, dtype, k);
    add_optional_head_bias(arena, layout.v_bias, head, head_dim, dtype, v);
    per_head_rms_norm_block(arena, layout.k_norm, k, head_dim, dtype, rms_eps);
    apply_rope_head(k, head_dim, current_position, rope_theta);
    const uint64_t token_base = kv_cache_token_base(
        layer_index, kv_block_count, kv_block_table, current_position,
        kv_hidden, static_cast<uint32_t>(head) * head_dim);
    for (uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
      kv_keys[token_base + index] = f32_to_encoded(k[index], dtype);
      kv_values[token_base + index] = f32_to_encoded(v[index], dtype);
    }
  }
}

__global__ void hf_layer_attention_reduce_kernel(
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, uint32_t attention_chunks, float *scratch,
    const float *partial_values, const float *partial_m, const float *partial_l,
    uint16_t *projection_input) {
  extern __shared__ float chunk_weights[];
  (void)step_cursor;
  (void)max_steps;
  const uint32_t head = blockIdx.x;
  if (head >= heads || head_dim > blockDim.x * kHeadThreadElements) {
    return;
  }
  if (threadIdx.x == 0) {
    float global_m = -INFINITY;
    float global_l = 0.0f;
    for (uint32_t chunk = 0; chunk < attention_chunks; ++chunk) {
      const uint64_t slot =
          (static_cast<uint64_t>(head) * attention_chunks + chunk);
      const float chunk_l = partial_l[slot];
      if (chunk_l <= 0.0f || !isfinite(chunk_l)) {
        continue;
      }
      const float chunk_m = partial_m[slot];
      const float next_m = fmaxf(global_m, chunk_m);
      const float old_scale =
          global_l == 0.0f ? 0.0f : expf(global_m - next_m);
      const float new_scale = expf(chunk_m - next_m);
      global_l = global_l * old_scale + chunk_l * new_scale;
      global_m = next_m;
    }
    for (uint32_t chunk = 0; chunk < attention_chunks; ++chunk) {
      const uint64_t slot =
          (static_cast<uint64_t>(head) * attention_chunks + chunk);
      const float chunk_l = partial_l[slot];
      if (global_l > 0.0f && isfinite(global_l) && chunk_l > 0.0f &&
          isfinite(chunk_l)) {
        chunk_weights[chunk] = expf(partial_m[slot] - global_m) / global_l;
      } else {
        chunk_weights[chunk] = 0.0f;
      }
    }
  }
  __syncthreads();
  const uint32_t head_start = head * head_dim;
  for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
    float value = 0.0f;
    for (uint32_t chunk = 0; chunk < attention_chunks; ++chunk) {
      const float weight = chunk_weights[chunk];
      if (weight != 0.0f) {
        const uint64_t slot =
            (static_cast<uint64_t>(head) * attention_chunks + chunk);
        value += partial_values[slot * head_dim + offset] * weight;
      }
    }
    LayerScratch s =
        layer_scratch_ptrs(scratch, hidden, heads * head_dim,
                           kv_heads * head_dim, intermediate);
    s.attn[head_start + offset] = value;
    projection_input[head_start + offset] = f32_to_encoded(value, dtype);
  }
}

__global__ void hf_layer_query_gate_attention_encode_kernel(
    uint32_t dtype, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float *scratch, uint16_t *projection_input) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint32_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t index = start; index < attention_hidden; index += stride) {
    const float value =
        encoded_to_f32(projection_input[index], dtype) *
        sigmoid(s.q_gate[index]);
    s.attn[index] = value;
    projection_input[index] = f32_to_encoded(value, dtype);
  }
}

__global__ void hf_layer_mlp_norm_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input) {
  (void)step_cursor;
  (void)max_steps;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  add_bias(arena, layout.o_bias, hidden, dtype, s.residual);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    s.residual[index] += s.input[index];
  }
  __syncthreads();
  rms_norm_to_encoded(s.residual, arena + layout.rms_mlp, hidden, dtype, rms_eps,
                      projection_input);
}

__global__ void hf_deepseek_residual_mlp_norm_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t norm_weight_dtype, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    s.residual[index] += s.input[index];
  }
  __syncthreads();
  rms_norm_to_encoded_with_weight_dtype(s.residual, arena + layout.rms_mlp,
                                        hidden, norm_weight_dtype, dtype,
                                        rms_eps, projection_input);
}

__global__ void hf_layer_ff_encode_kernel(
    uint32_t dtype, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float *scratch, uint16_t *projection_input) {
  (void)step_cursor;
  (void)max_steps;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint32_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t index = start; index < intermediate; index += stride) {
    const float value = silu(s.gate[index]) * s.up[index];
    s.ff[index] = value;
    projection_input[index] = f32_to_encoded(value, dtype);
  }
}

__global__ void hf_layer_sparse_moe_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch, uint16_t *projection_input) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  encoded_slice_to_f32(projection_input, hidden, dtype, s.mlp_norm);
  run_sparse_moe_mlp(arena, layout, dtype, hidden, intermediate, s.mlp_norm,
                     s.gate, s.up, s.ff, s.down, s.input);
}

__global__ void hf_deepseek_v3_sparse_moe_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch, uint16_t *projection_input) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  __shared__ float router_logits[kSparseMoeExpertsMax];
  __shared__ float correction_bias[kSparseMoeExpertsMax];
  __shared__ uint32_t selected_experts[kSparseMoeTopKMax];
  __shared__ float selected_weights[kSparseMoeTopKMax];
  __shared__ int route_status;

  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  encoded_slice_to_f32(projection_input, hidden, dtype, s.mlp_norm);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    s.down[index] = 0.0f;
  }
  __syncthreads();

  const uint32_t num_experts = layout.num_experts;
  const uint32_t top_k = layout.experts_per_token;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  if (layout.w_router == kMissingOffset ||
      layout.w_expert_gate_up == kMissingOffset ||
      layout.w_expert_down == kMissingOffset ||
      num_experts == 0 || num_experts > kSparseMoeExpertsMax ||
      top_k == 0 || top_k > kSparseMoeTopKMax || top_k > num_experts ||
      moe_intermediate == 0 || moe_intermediate > intermediate) {
    return;
  }

  for (uint32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    float sum = 0.0f;
    const uint64_t row = layout.w_router +
                         static_cast<uint64_t>(expert) * hidden;
    for (uint32_t col = 0; col < hidden; ++col) {
      sum += encoded_to_f32(arena[row + col], kDTypeBF16) * s.mlp_norm[col];
    }
    router_logits[expert] = sum;
  }
  const bool has_router_bias =
      (layout.deepseek_flags & kDeepSeekFlagRouterBias) != 0;
  const uint64_t router_bias_offset =
      layout.w_router + static_cast<uint64_t>(num_experts) * hidden;
  for (uint32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    correction_bias[expert] =
        has_router_bias ? f32_from_u16_slots(arena + router_bias_offset, expert)
                        : 0.0f;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const uint32_t router_groups =
        layout.deepseek_router_num_groups == 0 ? 1u
                                               : layout.deepseek_router_num_groups;
    const uint32_t router_topk_groups =
        layout.deepseek_router_topk_groups == 0
            ? 1u
            : layout.deepseek_router_topk_groups;
    const float routed_scale =
        isfinite(layout.deepseek_routed_scaling_factor) &&
                layout.deepseek_routed_scaling_factor != 0.0f
            ? layout.deepseek_routed_scaling_factor
            : 1.0f;
    route_status = nerva::deepseek::router::route_v3_grouped_sigmoid(
        router_logits, has_router_bias ? correction_bias : nullptr,
        num_experts, router_groups, router_topk_groups, top_k,
        layout.norm_topk_prob, routed_scale, selected_experts,
        selected_weights);
  }
  __syncthreads();
  if (route_status != 0) {
    return;
  }

  const uint64_t expert_gate = layout.w_expert_gate_up;
  const uint64_t expert_gate_scale =
      expert_gate +
      deepseek_device_fp8_slots(
          static_cast<uint64_t>(num_experts) * moe_intermediate, hidden);
  const uint64_t expert_up =
      expert_gate_scale +
      static_cast<uint64_t>(num_experts) *
          deepseek_device_scale_f32_slots(moe_intermediate, hidden);
  const uint64_t expert_up_scale =
      expert_up +
      deepseek_device_fp8_slots(
          static_cast<uint64_t>(num_experts) * moe_intermediate, hidden);
  const uint64_t expert_down = layout.w_expert_down;
  const uint64_t expert_down_scale =
      expert_down +
      deepseek_device_fp8_slots(
          static_cast<uint64_t>(num_experts) * hidden, moe_intermediate);

  for (uint32_t rank = 0; rank < top_k; ++rank) {
    const uint32_t expert = selected_experts[rank];
    const float expert_weight = selected_weights[rank];
    for (uint32_t row = threadIdx.x; row < moe_intermediate;
         row += blockDim.x) {
      float gate_sum = 0.0f;
      float up_sum = 0.0f;
      for (uint32_t col = 0; col < hidden; ++col) {
        gate_sum += deepseek_fp8_rank3_scaled_weight(
                        arena, expert_gate, expert_gate_scale,
                        moe_intermediate, hidden, expert, row, col) *
                    s.mlp_norm[col];
        up_sum += deepseek_fp8_rank3_scaled_weight(
                      arena, expert_up, expert_up_scale, moe_intermediate,
                      hidden, expert, row, col) *
                  s.mlp_norm[col];
      }
      s.gate[row] = gate_sum;
      s.up[row] = up_sum;
      s.ff[row] = silu(gate_sum) * up_sum;
    }
    __syncthreads();
    for (uint32_t row = threadIdx.x; row < hidden; row += blockDim.x) {
      float down_sum = 0.0f;
      for (uint32_t col = 0; col < moe_intermediate; ++col) {
        down_sum += deepseek_fp8_rank3_scaled_weight(
                        arena, expert_down, expert_down_scale, hidden,
                        moe_intermediate, expert, row, col) *
                    s.ff[col];
      }
      s.down[row] += expert_weight * down_sum;
    }
    __syncthreads();
  }

  const uint32_t shared_intermediate = layout.shared_expert_intermediate;
  if (shared_intermediate != 0) {
    if (layout.w_shared_expert_gate == kMissingOffset ||
        layout.w_shared_expert_up == kMissingOffset ||
        layout.w_shared_expert_down == kMissingOffset ||
        shared_intermediate > intermediate) {
      return;
    }
    const uint64_t shared_gate_scale =
        layout.w_shared_expert_gate +
        deepseek_device_fp8_slots(shared_intermediate, hidden);
    const uint64_t shared_up_scale =
        layout.w_shared_expert_up +
        deepseek_device_fp8_slots(shared_intermediate, hidden);
    const uint64_t shared_down_scale =
        layout.w_shared_expert_down +
        deepseek_device_fp8_slots(hidden, shared_intermediate);
    for (uint32_t row = threadIdx.x; row < shared_intermediate;
         row += blockDim.x) {
      float gate_sum = 0.0f;
      float up_sum = 0.0f;
      for (uint32_t col = 0; col < hidden; ++col) {
        gate_sum += deepseek_fp8_scaled_weight(
                        arena, layout.w_shared_expert_gate,
                        shared_gate_scale, shared_intermediate, hidden, row,
                        col) *
                    s.mlp_norm[col];
        up_sum += deepseek_fp8_scaled_weight(
                      arena, layout.w_shared_expert_up, shared_up_scale,
                      shared_intermediate, hidden, row, col) *
                  s.mlp_norm[col];
      }
      s.ff[row] = silu(gate_sum) * up_sum;
    }
    __syncthreads();
    for (uint32_t row = threadIdx.x; row < hidden; row += blockDim.x) {
      float down_sum = 0.0f;
      for (uint32_t col = 0; col < shared_intermediate; ++col) {
        down_sum += deepseek_fp8_scaled_weight(
                        arena, layout.w_shared_expert_down,
                        shared_down_scale, hidden, shared_intermediate, row,
                        col) *
                    s.ff[col];
      }
      s.down[row] += down_sum;
    }
    __syncthreads();
  }
}

__global__ void hf_deepseek_v4_swa_dense_layer_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rms_eps, float rope_theta, float *scratch, uint16_t *kv_keys,
    uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint32_t vocab_size,
    const uint32_t *prompt_tokens, uint32_t prompt_token_count,
    const NervaCudaSyntheticTokenSlot *slots, uint16_t *projection_input) {
  if (threadIdx.x != 0 ||
      (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t current_token =
      position < prompt_token_count ? prompt_tokens[position]
                                    : slots[position - 1u].token;
  const uint32_t q_lora_rank = layout.deepseek_q_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t o_lora_rank = layout.deepseek_o_lora_rank;
  const uint32_t o_groups = layout.deepseek_o_groups;
  if (q_lora_rank == 0 || qk_rope == 0 || head_dim == 0 ||
      qk_nope + qk_rope != head_dim || o_lora_rank == 0 ||
      o_groups == 0 || heads % o_groups != 0 ||
      layout.w_q == kMissingOffset || layout.deepseek_q_a_scale == kMissingOffset ||
      layout.q_norm == kMissingOffset || layout.deepseek_q_b == kMissingOffset ||
      layout.deepseek_q_b_scale == kMissingOffset ||
      layout.w_k == kMissingOffset || layout.deepseek_kv_a_scale == kMissingOffset ||
      layout.k_norm == kMissingOffset || layout.w_o == kMissingOffset ||
      layout.deepseek_o_a_scale == kMissingOffset ||
      layout.deepseek_o_b == kMissingOffset ||
      layout.deepseek_o_b_scale == kMissingOffset) {
    return;
  }

  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, heads * head_dim, head_dim, intermediate);
  const uint32_t attention_hidden = heads * head_dim;
  for (uint32_t row = 0; row < q_lora_rank; ++row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      sum += deepseek_fp8_e8m0_scaled_weight(
                 arena, layout.w_q, layout.deepseek_q_a_scale, q_lora_rank,
                 hidden, row, col) *
             encoded_to_f32(projection_input[col], dtype);
    }
    s.q[row] = sum;
  }
  for (uint32_t row = 0; row < head_dim; ++row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      sum += deepseek_fp8_e8m0_scaled_weight(
                 arena, layout.w_k, layout.deepseek_kv_a_scale, head_dim,
                 hidden, row, col) *
             encoded_to_f32(projection_input[col], dtype);
    }
    s.k[row] = sum;
  }

  float q_norm_sum = 0.0f;
  for (uint32_t index = 0; index < q_lora_rank; ++index) {
    q_norm_sum += s.q[index] * s.q[index];
  }
  const float q_norm_scale =
      rsqrtf(q_norm_sum / static_cast<float>(q_lora_rank) + rms_eps);
  for (uint32_t index = 0; index < q_lora_rank; ++index) {
    s.q[index] *= q_norm_scale *
                  encoded_to_f32(arena[layout.q_norm + index], kDTypeBF16);
  }
  for (uint32_t row = 0; row < attention_hidden; ++row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < q_lora_rank; ++col) {
      sum += deepseek_fp8_e8m0_scaled_weight(
                 arena, layout.deepseek_q_b, layout.deepseek_q_b_scale,
                 attention_hidden, q_lora_rank, row, col) *
             s.q[col];
    }
    s.q[row] = sum;
  }

  float kv_norm_sum = 0.0f;
  for (uint32_t index = 0; index < head_dim; ++index) {
    kv_norm_sum += s.k[index] * s.k[index];
  }
  const float kv_norm_scale =
      rsqrtf(kv_norm_sum / static_cast<float>(head_dim) + rms_eps);
  for (uint32_t index = 0; index < head_dim; ++index) {
    s.k[index] *= kv_norm_scale *
                  encoded_to_f32(arena[layout.k_norm + index], kDTypeBF16);
  }

  const uint32_t rope_half = qk_rope / 2u;
  for (uint32_t head = 0; head < heads; ++head) {
    const uint32_t head_start = head * head_dim;
    float q_head_norm = 0.0f;
    for (uint32_t dim = 0; dim < head_dim; ++dim) {
      const float value = s.q[head_start + dim];
      q_head_norm += value * value;
    }
    const float q_head_scale =
        rsqrtf(q_head_norm / static_cast<float>(head_dim) + rms_eps);
    for (uint32_t dim = 0; dim < head_dim; ++dim) {
      s.q[head_start + dim] *= q_head_scale;
    }
    if (rope_half != 0) {
      for (uint32_t offset = 0; offset < rope_half; ++offset) {
        const uint32_t left = head_start + qk_nope + offset;
        const uint32_t right = left + rope_half;
        const float left_value = s.q[left];
        const float right_value = s.q[right];
        s.q[left] = deepseek_rope_value_serial(
            left_value, right_value, offset, qk_rope, position, rope_theta,
            false);
        s.q[right] = deepseek_rope_value_serial(
            left_value, right_value, offset, qk_rope, position, rope_theta,
            true);
      }
    }
  }
  if (rope_half != 0) {
    for (uint32_t offset = 0; offset < rope_half; ++offset) {
      const uint32_t left = qk_nope + offset;
      const uint32_t right = left + rope_half;
      const float left_value = s.k[left];
      const float right_value = s.k[right];
      s.k[left] = deepseek_rope_value_serial(
          left_value, right_value, offset, qk_rope, position, rope_theta,
          false);
      s.k[right] = deepseek_rope_value_serial(
          left_value, right_value, offset, qk_rope, position, rope_theta,
          true);
    }
  }

  const uint64_t write_base =
      kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                          position, head_dim, 0);
  for (uint32_t dim = 0; dim < head_dim; ++dim) {
    const uint16_t encoded = f32_to_encoded(s.k[dim], dtype);
    kv_keys[write_base + dim] = encoded;
    kv_values[write_base + dim] = encoded;
  }

  const float attn_scale = rsqrtf(static_cast<float>(head_dim));
  for (uint32_t head = 0; head < heads; ++head) {
    const uint32_t head_start = head * head_dim;
    float local_m = layout.deepseek_attention_sink == kMissingOffset
                        ? -INFINITY
                        : f32_from_u16_slots(arena + layout.deepseek_attention_sink,
                                             head);
    float local_l = isfinite(local_m) ? 1.0f : 0.0f;
    for (uint32_t dim = 0; dim < head_dim; ++dim) {
      s.attn[head_start + dim] = 0.0f;
    }
    for (uint32_t token = 0; token <= position; ++token) {
      const uint64_t token_base =
          kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                              token, head_dim, 0);
      float score = 0.0f;
      for (uint32_t dim = 0; dim < head_dim; ++dim) {
        score += s.q[head_start + dim] *
                 encoded_to_f32(kv_keys[token_base + dim], dtype);
      }
      score *= attn_scale;
      const float next_m = fmaxf(local_m, score);
      const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
      const float new_scale = expf(score - next_m);
      for (uint32_t dim = 0; dim < head_dim; ++dim) {
        const uint32_t out = head_start + dim;
        s.attn[out] =
            s.attn[out] * old_scale +
            encoded_to_f32(kv_values[token_base + dim], dtype) * new_scale;
      }
      local_l = local_l * old_scale + new_scale;
      local_m = next_m;
    }
    if (local_l > 0.0f && isfinite(local_l)) {
      for (uint32_t dim = 0; dim < head_dim; ++dim) {
        s.attn[head_start + dim] /= local_l;
      }
    }
    if (rope_half != 0) {
      for (uint32_t offset = 0; offset < rope_half; ++offset) {
        const uint32_t left = head_start + qk_nope + offset;
        const uint32_t right = left + rope_half;
        const float exponent =
            static_cast<float>(2u * offset) / static_cast<float>(qk_rope);
        const float angle =
            static_cast<float>(position) / powf(rope_theta, exponent);
        float sin_value = 0.0f;
        float cos_value = 0.0f;
        sincosf(angle, &sin_value, &cos_value);
        const float left_value = s.attn[left];
        const float right_value = s.attn[right];
        s.attn[left] = left_value * cos_value + right_value * sin_value;
        s.attn[right] = right_value * cos_value - left_value * sin_value;
      }
    }
  }

  const uint32_t heads_per_group = heads / o_groups;
  const uint32_t wo_a_cols = heads_per_group * head_dim;
  const uint32_t wo_a_rows = o_groups * o_lora_rank;
  for (uint32_t group = 0; group < o_groups; ++group) {
    for (uint32_t row = 0; row < o_lora_rank; ++row) {
      float sum = 0.0f;
      const uint32_t global_row = group * o_lora_rank + row;
      for (uint32_t col = 0; col < wo_a_cols; ++col) {
        sum += deepseek_fp8_e8m0_scaled_weight(
                   arena, layout.w_o, layout.deepseek_o_a_scale, wo_a_rows,
                   wo_a_cols, global_row, col) *
               s.attn[group * wo_a_cols + col];
      }
      s.q_gate[global_row] = sum;
    }
  }
  for (uint32_t row = 0; row < hidden; ++row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < wo_a_rows; ++col) {
      sum += deepseek_fp8_e8m0_scaled_weight(
                 arena, layout.deepseek_o_b, layout.deepseek_o_b_scale, hidden,
                 wo_a_rows, row, col) *
             s.q_gate[col];
    }
    s.residual[row] = sum + s.input[row];
  }

  float mlp_norm_sum = 0.0f;
  for (uint32_t index = 0; index < hidden; ++index) {
    mlp_norm_sum += s.residual[index] * s.residual[index];
  }
  const float mlp_norm_scale =
      rsqrtf(mlp_norm_sum / static_cast<float>(hidden) + rms_eps);
  for (uint32_t index = 0; index < hidden; ++index) {
    s.mlp_norm[index] =
        s.residual[index] * mlp_norm_scale *
        encoded_to_f32(arena[layout.rms_mlp + index], kDTypeBF16);
  }
  if (layout.mlp_kind == kMlpKindSparseMoe) {
    float router_logits[kSparseMoeExpertsMax];
    float correction_bias[kSparseMoeExpertsMax];
    uint32_t selected_experts[kSparseMoeTopKMax];
    float selected_weights[kSparseMoeTopKMax];
    const uint32_t num_experts = layout.num_experts;
    const uint32_t top_k = layout.experts_per_token;
    const uint32_t moe_intermediate = layout.moe_intermediate;
    if (layout.w_router == kMissingOffset ||
        layout.w_expert_gate_up == kMissingOffset ||
        layout.w_expert_down == kMissingOffset ||
        num_experts == 0 || num_experts > kSparseMoeExpertsMax ||
        top_k == 0 || top_k > kSparseMoeTopKMax ||
        top_k > num_experts || moe_intermediate == 0 ||
        moe_intermediate > intermediate || (hidden & 1u) != 0 ||
        (moe_intermediate & 1u) != 0) {
      return;
    }
    const uint64_t router_metadata_offset =
        layout.w_router + static_cast<uint64_t>(num_experts) * hidden;
    const bool hash_router =
        (layout.deepseek_flags & kDeepSeekFlagHashRouter) != 0;
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
      float sum = 0.0f;
      const uint64_t row = layout.w_router +
                           static_cast<uint64_t>(expert) * hidden;
      for (uint32_t col = 0; col < hidden; ++col) {
        sum += encoded_to_f32(arena[row + col], kDTypeBF16) * s.mlp_norm[col];
      }
      router_logits[expert] = sum;
      correction_bias[expert] =
          hash_router ? 0.0f
                      : f32_from_u16_slots(arena + router_metadata_offset,
                                           expert);
    }
    const float routed_scale =
        isfinite(layout.deepseek_routed_scaling_factor) &&
                layout.deepseek_routed_scaling_factor != 0.0f
            ? layout.deepseek_routed_scaling_factor
            : 1.0f;
    if (hash_router) {
      if (current_token >= vocab_size) {
        return;
      }
      float weight_sum = 0.0f;
      for (uint32_t rank = 0; rank < top_k; ++rank) {
        const uint64_t table_index =
            static_cast<uint64_t>(current_token) * top_k + rank;
        const uint64_t expert64 =
            deepseek_u64_from_u16_slots(arena + router_metadata_offset,
                                        table_index);
        if (expert64 >= num_experts) {
          return;
        }
        const uint32_t expert = static_cast<uint32_t>(expert64);
        selected_experts[rank] = expert;
        selected_weights[rank] =
            nerva::deepseek::router::sqrtsoftplus_score(router_logits[expert]);
        weight_sum += selected_weights[rank];
      }
      const float scale = nerva::deepseek::router::route_scale(
          weight_sum, layout.norm_topk_prob, routed_scale);
      for (uint32_t rank = 0; rank < top_k; ++rank) {
        selected_weights[rank] *= scale;
      }
    } else {
      if (nerva::deepseek::router::route_v4_sqrtsoftplus(
              router_logits, correction_bias, num_experts, top_k,
              layout.norm_topk_prob, routed_scale, selected_experts,
              selected_weights) != 0) {
        return;
      }
    }
    for (uint32_t row = 0; row < hidden; ++row) {
      s.down[row] = 0.0f;
    }

    const uint32_t half_hidden = hidden >> 1u;
    const uint32_t half_intermediate = moe_intermediate >> 1u;
    const uint64_t expert_gate = layout.w_expert_gate_up;
    const uint64_t expert_gate_scale =
        expert_gate +
        deepseek_device_rank3_slots(num_experts, moe_intermediate,
                                    half_hidden);
    const uint32_t gate_scale_cols = (half_hidden + 15u) / 16u;
    const uint64_t expert_up =
        expert_gate_scale +
        deepseek_device_rank3_slots(num_experts, moe_intermediate,
                                    gate_scale_cols);
    const uint64_t expert_up_scale =
        expert_up +
        deepseek_device_rank3_slots(num_experts, moe_intermediate,
                                    half_hidden);
    const uint64_t expert_down = layout.w_expert_down;
    const uint64_t expert_down_scale =
        expert_down +
        deepseek_device_rank3_slots(num_experts, hidden, half_intermediate);
    for (uint32_t rank = 0; rank < top_k; ++rank) {
      const uint32_t expert = selected_experts[rank];
      const float expert_weight = selected_weights[rank];
      for (uint32_t row = 0; row < moe_intermediate; ++row) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (uint32_t col = 0; col < hidden; ++col) {
          gate_sum += deepseek_mxfp4_rank3_scaled_weight(
                          arena, expert_gate, expert_gate_scale,
                          moe_intermediate, half_hidden, expert, row, col) *
                      s.mlp_norm[col];
          up_sum += deepseek_mxfp4_rank3_scaled_weight(
                        arena, expert_up, expert_up_scale, moe_intermediate,
                        half_hidden, expert, row, col) *
                    s.mlp_norm[col];
        }
        s.ff[row] = silu(gate_sum) * up_sum;
      }
      for (uint32_t row = 0; row < hidden; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < moe_intermediate; ++col) {
          sum += deepseek_mxfp4_rank3_scaled_weight(
                     arena, expert_down, expert_down_scale, hidden,
                     half_intermediate, expert, row, col) *
                 s.ff[col];
        }
        s.down[row] += expert_weight * sum;
      }
    }

    const uint32_t shared_intermediate = layout.shared_expert_intermediate;
    if (shared_intermediate != 0) {
      if (layout.w_shared_expert_gate == kMissingOffset ||
          layout.w_shared_expert_up == kMissingOffset ||
          layout.w_shared_expert_down == kMissingOffset ||
          shared_intermediate > intermediate) {
        return;
      }
      const uint64_t shared_gate_scale =
          layout.w_shared_expert_gate +
          deepseek_device_fp8_slots(shared_intermediate, hidden);
      const uint64_t shared_up_scale =
          layout.w_shared_expert_up +
          deepseek_device_fp8_slots(shared_intermediate, hidden);
      const uint64_t shared_down_scale =
          layout.w_shared_expert_down +
          deepseek_device_fp8_slots(hidden, shared_intermediate);
      for (uint32_t row = 0; row < shared_intermediate; ++row) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (uint32_t col = 0; col < hidden; ++col) {
          gate_sum += deepseek_fp8_e8m0_scaled_weight(
                          arena, layout.w_shared_expert_gate,
                          shared_gate_scale, shared_intermediate, hidden, row,
                          col) *
                      s.mlp_norm[col];
          up_sum += deepseek_fp8_e8m0_scaled_weight(
                        arena, layout.w_shared_expert_up, shared_up_scale,
                        shared_intermediate, hidden, row, col) *
                    s.mlp_norm[col];
        }
        s.ff[row] = silu(gate_sum) * up_sum;
      }
      for (uint32_t row = 0; row < hidden; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < shared_intermediate; ++col) {
          sum += deepseek_fp8_e8m0_scaled_weight(
                     arena, layout.w_shared_expert_down, shared_down_scale,
                     hidden, shared_intermediate, row, col) *
                 s.ff[col];
        }
        s.down[row] += sum;
      }
    }
  } else {
    const uint64_t gate_scale =
        layout.w_gate + deepseek_device_fp8_slots(intermediate, hidden);
    const uint64_t up_scale =
        layout.w_up + deepseek_device_fp8_slots(intermediate, hidden);
    const uint64_t down_scale =
        layout.w_down + deepseek_device_fp8_slots(hidden, intermediate);
    for (uint32_t row = 0; row < intermediate; ++row) {
      float gate_sum = 0.0f;
      float up_sum = 0.0f;
      for (uint32_t col = 0; col < hidden; ++col) {
        gate_sum += deepseek_fp8_scaled_weight(
                        arena, layout.w_gate, gate_scale, intermediate, hidden,
                        row, col) *
                    s.mlp_norm[col];
        up_sum += deepseek_fp8_scaled_weight(
                      arena, layout.w_up, up_scale, intermediate, hidden, row,
                      col) *
                  s.mlp_norm[col];
      }
      s.ff[row] = silu(gate_sum) * up_sum;
    }
    for (uint32_t row = 0; row < hidden; ++row) {
      float sum = 0.0f;
      for (uint32_t col = 0; col < intermediate; ++col) {
        sum += deepseek_fp8_scaled_weight(
                   arena, layout.w_down, down_scale, hidden, intermediate, row,
                   col) *
               s.ff[col];
      }
      s.down[row] = sum;
    }
  }
}

__global__ void hf_layer_finish_kernel(
    uint16_t *arena, uint64_t output_offset, uint32_t dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float *scratch) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    s.down[index] += s.residual[index];
  }
  __syncthreads();
  f32_slice_to_encoded(s.down, arena + output_offset, hidden, dtype);
}

__global__ void hf_layer_finish_next_attn_norm_encode_kernel(
    uint16_t *arena, uint64_t output_offset, SequenceLayerLayout next_layout,
    uint32_t dtype, uint32_t next_norm_weight_dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input) {
  (void)step_cursor;
  (void)max_steps;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    s.input[index] = s.residual[index] + s.down[index];
  }
  __syncthreads();
  rms_norm_to_encoded_with_weight_dtype(s.input, arena + next_layout.rms_attn,
                                        hidden, next_norm_weight_dtype, dtype,
                                        rms_eps, projection_input);
}

__global__ void hf_layer_finish_final_norm_encode_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t dtype,
    uint32_t final_norm_weight_dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input) {
  (void)step_cursor;
  (void)max_steps;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    s.input[index] = s.residual[index] + s.down[index];
  }
  __syncthreads();
  rms_norm_to_encoded_with_weight_dtype(s.input, arena + arena_layout.final_norm,
                                        hidden, final_norm_weight_dtype, dtype,
                                        rms_eps, projection_input);
}

__global__ void hf_final_norm_encode_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint64_t input_offset,
    uint32_t dtype, uint32_t hidden, uint32_t *step_cursor, uint32_t max_steps,
    float rms_eps, float *scratch, uint16_t *projection_input) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  float *decoded = scratch;
  float *final_norm = decoded + hidden;
  encoded_slice_to_f32(arena + input_offset, hidden, dtype, decoded);
  rms_norm(decoded, arena + arena_layout.final_norm, hidden, dtype, rms_eps, final_norm);
  f32_slice_to_encoded(final_norm, projection_input, hidden, dtype);
}












__global__ void hf_decode_set_step_kernel(uint32_t *step_cursor,
                                          uint32_t value) {
  if (threadIdx.x == 0) {
    *step_cursor = value;
  }
}

__global__ void hf_init_identity_kv_block_table_kernel(uint32_t *block_table,
                                                       uint32_t block_count) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < block_count; index += blockDim.x * gridDim.x) {
    block_table[index] = index;
  }
}

__global__ void hf_pack_qkv_weights_kernel(
    uint16_t *packed, const uint16_t *arena,
    const SequenceLayerLayout *layouts, uint32_t layer_count, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden) {
  const uint64_t rows_per_layer =
      static_cast<uint64_t>(attention_hidden) + static_cast<uint64_t>(kv_hidden) * 2;
  const uint64_t global_row = blockIdx.x;
  if (global_row >= rows_per_layer * layer_count) {
    return;
  }
  const uint32_t layer_index =
      static_cast<uint32_t>(global_row / rows_per_layer);
  const uint32_t row = static_cast<uint32_t>(global_row % rows_per_layer);
  const SequenceLayerLayout layout = layouts[layer_index];
  const uint16_t *src = nullptr;
  if (row < attention_hidden) {
    src = arena + layout.w_q + static_cast<uint64_t>(row) * hidden;
  } else if (row < attention_hidden + kv_hidden) {
    const uint32_t local_row = row - attention_hidden;
    src = arena + layout.w_k + static_cast<uint64_t>(local_row) * hidden;
  } else {
    const uint32_t local_row = row - attention_hidden - kv_hidden;
    src = arena + layout.w_v + static_cast<uint64_t>(local_row) * hidden;
  }
  uint16_t *dst = packed + global_row * hidden;
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    dst[col] = src[col];
  }
}

__global__ void hf_pack_gate_up_weights_kernel(
    uint16_t *packed, const uint16_t *arena,
    const SequenceLayerLayout *layouts, uint32_t layer_count, uint32_t hidden,
    uint32_t intermediate) {
  const uint64_t rows_per_layer = static_cast<uint64_t>(intermediate) * 2;
  const uint64_t global_row = blockIdx.x;
  if (global_row >= rows_per_layer * layer_count) {
    return;
  }
  const uint32_t layer_index =
      static_cast<uint32_t>(global_row / rows_per_layer);
  const uint32_t row = static_cast<uint32_t>(global_row % rows_per_layer);
  const SequenceLayerLayout layout = layouts[layer_index];
  if (layout.mlp_kind == kMlpKindSparseMoe) {
    uint16_t *dst = packed + global_row * hidden;
    for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
      dst[col] = 0;
    }
    return;
  }
  const uint16_t *src = nullptr;
  if (row < intermediate) {
    src = arena + layout.w_gate + static_cast<uint64_t>(row) * hidden;
  } else {
    const uint32_t local_row = row - intermediate;
    src = arena + layout.w_up + static_cast<uint64_t>(local_row) * hidden;
  }
  uint16_t *dst = packed + global_row * hidden;
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    dst[col] = src[col];
  }
}

__global__ void hf_projection_batch_pack_u16_kernel(
    const uint16_t *src, uint16_t *dst, uint32_t cols, uint32_t token_index) {
  const uint64_t offset = static_cast<uint64_t>(token_index) * cols;
  const uint32_t stride = static_cast<uint32_t>(gridDim.x) * blockDim.x;
  for (uint32_t col = blockIdx.x * blockDim.x + threadIdx.x; col < cols;
       col += stride) {
    dst[offset + col] = src[col];
  }
}

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
    uint32_t cols, uint32_t tokens) {
  const uint64_t total = static_cast<uint64_t>(cols) * tokens;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       index < total; index += stride) {
    const uint32_t token = static_cast<uint32_t>(index / cols);
    const uint32_t col = static_cast<uint32_t>(index % cols);
    const uint16_t *src = token == 0 ? src0
                          : token == 1 ? src1
                          : token == 2 ? src2
                          : token == 3 ? src3
                          : token == 4 ? src4
                          : token == 5 ? src5
                          : token == 6 ? src6
                          : token == 7 ? src7
                          : token == 8 ? src8
                          : token == 9 ? src9
                          : token == 10 ? src10
                          : token == 11 ? src11
                          : token == 12 ? src12
                          : token == 13 ? src13
                          : token == 14 ? src14
                          : token == 15 ? src15
                          : token == 16 ? src16
                          : token == 17 ? src17
                          : token == 18 ? src18
                          : token == 19 ? src19
                          : token == 20 ? src20
                          : token == 21 ? src21
                          : token == 22 ? src22
                          : token == 23 ? src23
                          : token == 24 ? src24
                          : token == 25 ? src25
                          : token == 26 ? src26
                          : token == 27 ? src27
                          : token == 28 ? src28
                          : token == 29 ? src29
                          : token == 30 ? src30
                                        : src31;
    dst[index] = src[col];
  }
}

__global__ void hf_projection_batch_scatter_f32_kernel(
    const float *src, float *dst, uint32_t rows, uint32_t token_index) {
  const uint64_t offset = static_cast<uint64_t>(token_index) * rows;
  const uint32_t stride = static_cast<uint32_t>(gridDim.x) * blockDim.x;
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += stride) {
    dst[row] = src[offset + row];
  }
}

__global__ void hf_projection_batch_scatter_small_f32_kernel(
    const float *src, float *dst0, float *dst1, float *dst2, float *dst3,
    float *dst4, float *dst5, float *dst6, float *dst7, float *dst8,
    float *dst9, float *dst10, float *dst11, float *dst12, float *dst13,
    float *dst14, float *dst15, float *dst16, float *dst17, float *dst18,
    float *dst19, float *dst20, float *dst21, float *dst22, float *dst23,
    float *dst24, float *dst25, float *dst26, float *dst27, float *dst28,
    float *dst29, float *dst30, float *dst31, uint32_t rows,
    uint32_t tokens) {
  const uint64_t total = static_cast<uint64_t>(rows) * tokens;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       index < total; index += stride) {
    const uint32_t token = static_cast<uint32_t>(index / rows);
    const uint32_t row = static_cast<uint32_t>(index % rows);
    float *dst = token == 0 ? dst0
                 : token == 1 ? dst1
                 : token == 2 ? dst2
                 : token == 3 ? dst3
                 : token == 4 ? dst4
                 : token == 5 ? dst5
                 : token == 6 ? dst6
                 : token == 7 ? dst7
                 : token == 8 ? dst8
                 : token == 9 ? dst9
                 : token == 10 ? dst10
                 : token == 11 ? dst11
                 : token == 12 ? dst12
                 : token == 13 ? dst13
                 : token == 14 ? dst14
                 : token == 15 ? dst15
                 : token == 16 ? dst16
                 : token == 17 ? dst17
                 : token == 18 ? dst18
                 : token == 19 ? dst19
                 : token == 20 ? dst20
                 : token == 21 ? dst21
                 : token == 22 ? dst22
                 : token == 23 ? dst23
                 : token == 24 ? dst24
                 : token == 25 ? dst25
                 : token == 26 ? dst26
                 : token == 27 ? dst27
                 : token == 28 ? dst28
                 : token == 29 ? dst29
                 : token == 30 ? dst30
                               : dst31;
    dst[row] = src[index];
  }
}
