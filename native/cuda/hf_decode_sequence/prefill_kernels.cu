#include "kernels.cuh"

#include "device_ops.cuh"

#include <stdint.h>

__global__ void hf_prefill_embed_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t hidden,
    const uint32_t *prompt_tokens, uint32_t prompt_token_count,
    uint16_t *hidden_out) {
  const uint32_t token = blockIdx.x;
  if (token >= prompt_token_count) {
    return;
  }
  const uint32_t token_id = prompt_tokens[token];
  const uint64_t embedding_offset =
      arena_layout.embeddings + static_cast<uint64_t>(token_id) * hidden;
  uint16_t *out = hidden_out + static_cast<uint64_t>(token) * hidden;
  copy_encoded_slice(out, arena + embedding_offset, hidden);
}

__global__ void hf_prefill_embed_range_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t hidden,
    const uint32_t *tokens, uint32_t token_count, uint32_t output_start,
    uint16_t *hidden_out) {
  const uint32_t token = blockIdx.x;
  if (token >= token_count) {
    return;
  }
  const uint32_t token_id = tokens[token];
  const uint64_t embedding_offset =
      arena_layout.embeddings + static_cast<uint64_t>(token_id) * hidden;
  uint16_t *out =
      hidden_out + static_cast<uint64_t>(output_start + token) * hidden;
  copy_encoded_slice(out, arena + embedding_offset, hidden);
}

__global__ void hf_prefill_attn_norm_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t chunk_start, uint32_t chunk_tokens,
    float rms_eps, const uint16_t *hidden_in, uint16_t *norm_out) {
  const uint32_t local_token = blockIdx.x;
  if (local_token >= chunk_tokens) {
    return;
  }
  const uint64_t global_token = chunk_start + local_token;
  const uint16_t *input = hidden_in + global_token * hidden;
  uint16_t *out = norm_out + static_cast<uint64_t>(local_token) * hidden;
  float mean_square = 0.0f;
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    const float value = encoded_to_f32(input[index], dtype);
    mean_square += value * value;
  }
  mean_square = block_sum(mean_square);
  const float scale =
      rsqrtf(mean_square / static_cast<float>(hidden) + rms_eps);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    const float value = encoded_to_f32(input[index], dtype) * scale *
                        encoded_to_f32(arena[layout.rms_attn + index], dtype);
    out[index] = f32_to_encoded(value, dtype);
  }
}

__global__ void hf_prefill_qkv_publish_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t max_steps, uint32_t chunk_start, uint32_t chunk_tokens,
    float rms_eps, float rope_theta, float *qkv, uint16_t *kv_keys,
    uint16_t *kv_values, uint16_t *qkv_encoded, uint32_t kv_block_count,
    const uint32_t *kv_block_table) {
  const uint32_t local_token = blockIdx.x;
  const uint32_t lane = blockIdx.y;
  if (local_token >= chunk_tokens) {
    return;
  }
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  const uint64_t rows = static_cast<uint64_t>(attention_hidden) + kv_hidden * 2;
  float *token_qkv = qkv + static_cast<uint64_t>(local_token) * rows;
  const uint32_t global_pos = chunk_start + local_token;
  if (lane < heads) {
    const uint32_t head_start = lane * head_dim;
    float mean_square = 0.0f;
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      const uint32_t index = head_start + offset;
      float value = token_qkv[index];
      if (layout.q_bias != kMissingOffset) {
        value += encoded_to_f32(arena[layout.q_bias + index], dtype);
      }
      token_qkv[index] = value;
      mean_square += value * value;
    }
    mean_square = block_sum(mean_square);
    if (layout.q_norm != kMissingOffset) {
      const float scale =
          rsqrtf(mean_square / static_cast<float>(head_dim) + rms_eps);
      for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
        const uint32_t index = head_start + offset;
        token_qkv[index] *=
            scale * encoded_to_f32(arena[layout.q_norm + offset], dtype);
      }
    }
    __syncthreads();
    apply_rope_head(token_qkv + head_start, head_dim, global_pos, rope_theta);
    if (qkv_encoded != nullptr) {
      uint16_t *encoded =
          qkv_encoded + static_cast<uint64_t>(local_token) * rows + head_start;
      for (uint32_t offset = threadIdx.x; offset < head_dim;
           offset += blockDim.x) {
        encoded[offset] = f32_to_encoded(token_qkv[head_start + offset], dtype);
      }
    }
  }
  if (lane < kv_heads) {
    const uint32_t kv_start = lane * head_dim;
    float *k = token_qkv + attention_hidden;
    float *v = k + kv_hidden;
    float mean_square = 0.0f;
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      const uint32_t index = kv_start + offset;
      float value = k[index];
      if (layout.k_bias != kMissingOffset) {
        value += encoded_to_f32(arena[layout.k_bias + index], dtype);
      }
      k[index] = value;
      mean_square += value * value;
    }
    mean_square = block_sum(mean_square);
    if (layout.k_norm != kMissingOffset) {
      const float scale =
          rsqrtf(mean_square / static_cast<float>(head_dim) + rms_eps);
      for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
        const uint32_t index = kv_start + offset;
        k[index] *= scale * encoded_to_f32(arena[layout.k_norm + offset], dtype);
      }
    }
    __syncthreads();
    apply_rope_head(k + kv_start, head_dim, global_pos, rope_theta);
    uint16_t *encoded_k = nullptr;
    uint16_t *encoded_v = nullptr;
    if (qkv_encoded != nullptr) {
      uint16_t *encoded =
          qkv_encoded + static_cast<uint64_t>(local_token) * rows;
      encoded_k = encoded + attention_hidden + kv_start;
      encoded_v = encoded + attention_hidden + kv_hidden + kv_start;
    }
    const uint64_t token_base = kv_cache_token_base(
        layer_index, kv_block_count, kv_block_table, global_pos, kv_hidden,
        kv_start);
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      float value = v[kv_start + offset];
      if (layout.v_bias != kMissingOffset) {
        value += encoded_to_f32(arena[layout.v_bias + kv_start + offset], dtype);
      }
      v[kv_start + offset] = value;
      const uint16_t encoded_key =
          f32_to_encoded(k[kv_start + offset], dtype);
      const uint16_t encoded_value = f32_to_encoded(value, dtype);
      kv_keys[token_base + offset] = encoded_key;
      kv_values[token_base + offset] = encoded_value;
      if (encoded_k != nullptr) {
        encoded_k[offset] = encoded_key;
        encoded_v[offset] = encoded_value;
      }
    }
  }
}

__global__ void hf_prefill_attention_kernel(
    uint32_t layer_index, uint32_t dtype, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t max_steps, uint32_t chunk_start,
    uint32_t chunk_tokens, const float *qkv, const uint16_t *kv_keys,
    const uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint16_t *attn_out) {
  const uint32_t local_token = blockIdx.x;
  const uint32_t head = blockIdx.y;
  if (local_token >= chunk_tokens || head >= heads) {
    return;
  }
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  const uint64_t rows = static_cast<uint64_t>(attention_hidden) + kv_hidden * 2;
  const float *q = qkv + static_cast<uint64_t>(local_token) * rows;
  uint16_t *out = attn_out + static_cast<uint64_t>(local_token) * attention_hidden;
  const uint32_t global_pos = chunk_start + local_token;
  const uint32_t group = heads / kv_heads;
  const uint32_t kv_head = head / group;
  const uint32_t head_start = head * head_dim;
  const uint32_t kv_start = kv_head * head_dim;
  extern __shared__ float shared_attn[];
  for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
    shared_attn[offset] = 0.0f;
  }
  __syncthreads();
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float local_m = -INFINITY;
  float local_l = 0.0f;
  for (uint32_t token = 0; token <= global_pos; ++token) {
    const uint64_t token_base = kv_cache_token_base(
        layer_index, kv_block_count, kv_block_table, token, kv_hidden,
        kv_start);
    float partial = 0.0f;
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      partial += q[head_start + offset] *
                 encoded_to_f32(kv_keys[token_base + offset], dtype);
    }
    const float score = block_sum(partial) * scale;
    const float next_m = fmaxf(local_m, score);
    const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
    const float new_scale = expf(score - next_m);
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      shared_attn[offset] =
          shared_attn[offset] * old_scale +
          encoded_to_f32(kv_values[token_base + offset], dtype) * new_scale;
    }
    local_l = local_l * old_scale + new_scale;
    local_m = next_m;
  }
  const bool normalize = local_l > 0.0f && isfinite(local_l);
  for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
    float value = shared_attn[offset];
    if (normalize) {
      value /= local_l;
    }
    out[head_start + offset] = f32_to_encoded(value, dtype);
  }
}

__global__ void hf_prefill_mlp_norm_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t chunk_start, uint32_t chunk_tokens,
    float rms_eps, const uint16_t *hidden_in, float *attn_projection,
    uint16_t *norm_out) {
  const uint32_t local_token = blockIdx.x;
  if (local_token >= chunk_tokens) {
    return;
  }
  const uint64_t global_token = chunk_start + local_token;
  const uint16_t *input = hidden_in + global_token * hidden;
  float *residual = attn_projection + static_cast<uint64_t>(local_token) * hidden;
  uint16_t *out = norm_out + static_cast<uint64_t>(local_token) * hidden;
  float mean_square = 0.0f;
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    float value = residual[index];
    if (layout.o_bias != kMissingOffset) {
      value += encoded_to_f32(arena[layout.o_bias + index], dtype);
    }
    value += encoded_to_f32(input[index], dtype);
    residual[index] = value;
    mean_square += value * value;
  }
  mean_square = block_sum(mean_square);
  const float scale =
      rsqrtf(mean_square / static_cast<float>(hidden) + rms_eps);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    out[index] = f32_to_encoded(
        residual[index] * scale * encoded_to_f32(arena[layout.rms_mlp + index], dtype),
        dtype);
  }
}

__global__ void hf_prefill_ff_kernel(uint32_t dtype, uint32_t intermediate,
                                     uint32_t chunk_tokens,
                                     const float *gate_up,
                                     uint16_t *ff_out) {
  const uint64_t total = static_cast<uint64_t>(chunk_tokens) * intermediate;
  const uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t stride = blockDim.x * gridDim.x;
  for (uint64_t cursor = index; cursor < total; cursor += stride) {
    const uint64_t token = cursor / intermediate;
    const uint64_t offset = cursor - token * intermediate;
    const float *base = gate_up + token * intermediate * 2;
    const float value = silu(base[offset]) * base[intermediate + offset];
    ff_out[cursor] = f32_to_encoded(value, dtype);
  }
}

__global__ void hf_prefill_finish_kernel(uint32_t dtype, uint32_t hidden,
                                         uint32_t chunk_start,
                                         uint32_t chunk_tokens,
                                         const float *residual,
                                         const float *down,
                                         uint16_t *hidden_out) {
  const uint64_t total = static_cast<uint64_t>(chunk_tokens) * hidden;
  const uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t stride = blockDim.x * gridDim.x;
  for (uint64_t cursor = index; cursor < total; cursor += stride) {
    const uint64_t token = cursor / hidden;
    const uint64_t offset = cursor - token * hidden;
    const uint64_t out_index = (static_cast<uint64_t>(chunk_start) + token) * hidden + offset;
    hidden_out[out_index] = f32_to_encoded(residual[cursor] + down[cursor], dtype);
  }
}

__global__ void hf_prefill_final_norm_last_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t dtype,
    uint32_t hidden, uint32_t prompt_token_count, float rms_eps,
    const uint16_t *hidden_in, uint16_t *projection_input) {
  if (prompt_token_count == 0) {
    return;
  }
  const uint64_t base = static_cast<uint64_t>(prompt_token_count - 1u) * hidden;
  const uint16_t *input = hidden_in + base;
  float mean_square = 0.0f;
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    const float value = encoded_to_f32(input[index], dtype);
    mean_square += value * value;
  }
  mean_square = block_sum(mean_square);
  const float scale =
      rsqrtf(mean_square / static_cast<float>(hidden) + rms_eps);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    projection_input[index] = f32_to_encoded(
        encoded_to_f32(input[index], dtype) * scale *
            encoded_to_f32(arena[arena_layout.final_norm + index], dtype),
        dtype);
  }
}

__global__ void hf_prefill_final_norm_range_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, uint32_t dtype,
    uint32_t hidden, uint32_t chunk_start, uint32_t chunk_tokens,
    float rms_eps, const uint16_t *hidden_in, uint16_t *projection_input) {
  const uint32_t local_token = blockIdx.x;
  if (local_token >= chunk_tokens) {
    return;
  }
  const uint64_t base =
      static_cast<uint64_t>(chunk_start + local_token) * hidden;
  const uint16_t *input = hidden_in + base;
  uint16_t *out = projection_input + static_cast<uint64_t>(local_token) * hidden;
  float mean_square = 0.0f;
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    const float value = encoded_to_f32(input[index], dtype);
    mean_square += value * value;
  }
  mean_square = block_sum(mean_square);
  const float scale =
      rsqrtf(mean_square / static_cast<float>(hidden) + rms_eps);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    out[index] = f32_to_encoded(
        encoded_to_f32(input[index], dtype) * scale *
            encoded_to_f32(arena[arena_layout.final_norm + index], dtype),
        dtype);
  }
}
