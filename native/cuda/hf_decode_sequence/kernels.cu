#include "kernels.cuh"

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

__global__ void hf_decode_prepare_first_attn_norm_encode_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout,
    SequenceLayerLayout first_layout, uint32_t dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, const uint32_t *prompt_tokens,
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
  rms_norm_to_encoded(s.input, arena + first_layout.rms_attn, hidden, dtype,
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
    uint32_t dtype, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input) {
  (void)step_cursor;
  (void)max_steps;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    s.input[index] = s.residual[index] + s.down[index];
  }
  __syncthreads();
  rms_norm_to_encoded(s.input, arena + next_layout.rms_attn, hidden, dtype,
                      rms_eps, projection_input);
}

__global__ void hf_layer_finish_final_norm_encode_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rms_eps, float *scratch, uint16_t *projection_input) {
  (void)step_cursor;
  (void)max_steps;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    s.input[index] = s.residual[index] + s.down[index];
  }
  __syncthreads();
  rms_norm_to_encoded(s.input, arena + arena_layout.final_norm, hidden, dtype,
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
