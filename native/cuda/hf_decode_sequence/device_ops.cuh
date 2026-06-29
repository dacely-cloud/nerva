#pragma once

#include "types.cuh"

#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

static __device__ __forceinline__ float encoded_to_f32(uint16_t value, uint32_t dtype) {
  if (dtype == kDTypeBF16) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
  }
  return __half2float(__ushort_as_half(value));
}

template <uint32_t DType>
static __device__ __forceinline__ float encoded_to_f32_typed(uint16_t value) {
  if (DType == kDTypeBF16) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
  }
  return __half2float(__ushort_as_half(value));
}

static __device__ __forceinline__ uint16_t f32_to_encoded(float value, uint32_t dtype) {
  if (dtype == kDTypeBF16) {
    uint32_t bits = __float_as_uint(value);
    uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
  }
  return __half_as_ushort(__float2half_rn(value));
}

static __device__ __forceinline__ uint64_t kv_cache_token_offset(
    uint32_t layer_index, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint32_t token, uint32_t kv_hidden,
    uint32_t kv_offset) {
  const uint32_t logical_block = token / kKvCacheBlockTokens;
  const uint32_t block_offset = token - logical_block * kKvCacheBlockTokens;
  const uint32_t physical_block = kv_block_table[logical_block];
  return kv_cache_page_offset(layer_index, kv_block_count, physical_block,
                              block_offset, kv_hidden, kv_offset);
}

static __device__ __forceinline__ uint64_t kv_cache_token_base(
    uint32_t layer_index, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint32_t token, uint32_t kv_hidden,
    uint32_t kv_offset) {
  const uint32_t logical_block = token / kKvCacheBlockTokens;
  const uint32_t block_offset = token - logical_block * kKvCacheBlockTokens;
  const uint32_t physical_block = kv_block_table[logical_block];
  return kv_cache_page_offset(layer_index, kv_block_count, physical_block,
                              block_offset, kv_hidden, kv_offset);
}

static __device__ float silu(float value) {
  return value / (1.0f + expf(-value));
}

static __device__ __forceinline__ float sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

static __device__ float block_sum(float value) {
  __shared__ float warp_sums[32];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t warp = tid >> 5u;
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  float total = 0.0f;
  const uint32_t warp_count = (blockDim.x + 31u) >> 5u;
  if (warp == 0) {
    total = lane < warp_count ? warp_sums[lane] : 0.0f;
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xffffffffu, total, offset);
    }
    if (lane == 0) {
      warp_sums[0] = total;
    }
  }
  __syncthreads();
  return warp_sums[0];
}

static __device__ __forceinline__ float warp_sum(float value) {
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

static __device__ void encoded_slice_to_f32(const uint16_t *input, uint32_t len,
                                     uint32_t dtype, float *output) {
  for (uint32_t index = threadIdx.x; index < len; index += blockDim.x) {
    output[index] = encoded_to_f32(input[index], dtype);
  }
  __syncthreads();
}

static __device__ void f32_slice_to_encoded(const float *input, uint16_t *output,
                                     uint32_t len, uint32_t dtype) {
  for (uint32_t index = threadIdx.x; index < len; index += blockDim.x) {
    output[index] = f32_to_encoded(input[index], dtype);
  }
  __syncthreads();
}

static __device__ void copy_encoded_slice(uint16_t *dst, const uint16_t *src, uint32_t len) {
  for (uint32_t index = threadIdx.x; index < len; index += blockDim.x) {
    dst[index] = src[index];
  }
  __syncthreads();
}

static __device__ void mat_vec(const uint16_t *matrix, const float *input, uint32_t rows,
                        uint32_t cols, uint32_t dtype, float *output) {
  for (uint32_t row = threadIdx.x; row < rows; row += blockDim.x) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < cols; ++col) {
      sum += encoded_to_f32(matrix[row * cols + col], dtype) * input[col];
    }
    output[row] = sum;
  }
  __syncthreads();
}

static __device__ void rms_norm(const float *input, const uint16_t *weight, uint32_t hidden,
                         uint32_t dtype, float eps, float *output) {
  float mean_square = 0.0f;
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    mean_square += input[index] * input[index];
  }
  mean_square = block_sum(mean_square);
  const float scale = rsqrtf(mean_square / static_cast<float>(hidden) + eps);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    output[index] = input[index] * scale * encoded_to_f32(weight[index], dtype);
  }
  __syncthreads();
}

static __device__ void run_sparse_moe_mlp(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t intermediate, const float *mlp_norm,
    float *gate, float *up, float *ff, float *down, float *expert_down_tmp) {
  __shared__ float router_logits[kSparseMoeExpertsMax];
  __shared__ uint32_t selected_experts[kSparseMoeTopKMax];
  __shared__ float selected_weights[kSparseMoeTopKMax];

  const uint32_t num_experts = layout.num_experts;
  const uint32_t top_k = layout.experts_per_token;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    down[index] = 0.0f;
  }
  __syncthreads();

  if (num_experts == 0 || num_experts > kSparseMoeExpertsMax ||
      top_k == 0 || top_k > kSparseMoeTopKMax ||
      top_k > num_experts || moe_intermediate == 0 ||
      moe_intermediate > intermediate) {
    __syncthreads();
    return;
  }

  mat_vec(arena + layout.w_router, mlp_norm, num_experts, hidden, dtype,
          router_logits);

  if (threadIdx.x == 0) {
    float max_logit = -INFINITY;
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
      max_logit = fmaxf(max_logit, router_logits[expert]);
    }
    float total = 0.0f;
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
      total += expf(router_logits[expert] - max_logit);
    }
    for (uint32_t rank = 0; rank < top_k; ++rank) {
      uint32_t best_expert = UINT32_MAX;
      float best_weight = -INFINITY;
      for (uint32_t expert = 0; expert < num_experts; ++expert) {
        bool already_selected = false;
        for (uint32_t prev = 0; prev < rank; ++prev) {
          already_selected |= selected_experts[prev] == expert;
        }
        if (already_selected) {
          continue;
        }
        float weight = expf(router_logits[expert] - max_logit);
        if (total > 0.0f && isfinite(total)) {
          weight /= total;
        }
        if (weight > best_weight ||
            (weight == best_weight && expert < best_expert)) {
          best_weight = weight;
          best_expert = expert;
        }
      }
      selected_experts[rank] = best_expert;
      selected_weights[rank] = best_weight;
    }
    if (layout.norm_topk_prob != 0) {
      float selected_sum = 0.0f;
      for (uint32_t rank = 0; rank < top_k; ++rank) {
        selected_sum += selected_weights[rank];
      }
      if (selected_sum > 0.0f) {
        for (uint32_t rank = 0; rank < top_k; ++rank) {
          selected_weights[rank] /= selected_sum;
        }
      }
    }
  }
  __syncthreads();

  const uint64_t expert_gate_up_stride =
      static_cast<uint64_t>(moe_intermediate) * 2u * hidden;
  const uint64_t expert_down_stride =
      static_cast<uint64_t>(hidden) * moe_intermediate;
  for (uint32_t rank = 0; rank < top_k; ++rank) {
    const uint32_t expert = selected_experts[rank];
    const float expert_weight = selected_weights[rank];
    const uint64_t gate_up_base =
        layout.w_expert_gate_up +
        static_cast<uint64_t>(expert) * expert_gate_up_stride;
    const uint16_t *expert_gate = arena + gate_up_base;
    const uint16_t *expert_up =
        arena + gate_up_base +
        static_cast<uint64_t>(moe_intermediate) * hidden;
    const uint16_t *expert_down =
        arena + layout.w_expert_down +
        static_cast<uint64_t>(expert) * expert_down_stride;

    mat_vec(expert_gate, mlp_norm, moe_intermediate, hidden, dtype, gate);
    mat_vec(expert_up, mlp_norm, moe_intermediate, hidden, dtype, up);
    for (uint32_t index = threadIdx.x; index < moe_intermediate;
         index += blockDim.x) {
      ff[index] = silu(gate[index]) * up[index];
    }
    __syncthreads();
    mat_vec(expert_down, ff, hidden, moe_intermediate, dtype, expert_down_tmp);
    for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
      down[index] += expert_weight * expert_down_tmp[index];
    }
    __syncthreads();
  }

  const uint32_t shared_intermediate = layout.shared_expert_intermediate;
  if (shared_intermediate != 0) {
    float gate_weight = 0.0f;
    for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
      gate_weight +=
          encoded_to_f32(arena[layout.w_shared_expert_router + col], dtype) *
          mlp_norm[col];
    }
    gate_weight = block_sum(gate_weight);
    mat_vec(arena + layout.w_shared_expert_gate, mlp_norm,
            shared_intermediate, hidden, dtype, gate);
    mat_vec(arena + layout.w_shared_expert_up, mlp_norm, shared_intermediate,
            hidden, dtype, up);
    for (uint32_t index = threadIdx.x; index < shared_intermediate;
         index += blockDim.x) {
      ff[index] = silu(gate[index]) * up[index];
    }
    __syncthreads();
    mat_vec(arena + layout.w_shared_expert_down, ff, hidden,
            shared_intermediate, dtype, expert_down_tmp);
    const float shared_scale = sigmoid(gate_weight);
    for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
      down[index] += shared_scale * expert_down_tmp[index];
    }
    __syncthreads();
  }
}

static __device__ void rms_norm_to_encoded(const float *input, const uint16_t *weight,
                                    uint32_t hidden, uint32_t dtype, float eps,
                                    uint16_t *output) {
  float mean_square = 0.0f;
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    mean_square += input[index] * input[index];
  }
  mean_square = block_sum(mean_square);
  const float scale = rsqrtf(mean_square / static_cast<float>(hidden) + eps);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    output[index] =
        f32_to_encoded(input[index] * scale * encoded_to_f32(weight[index], dtype),
                       dtype);
  }
  __syncthreads();
}

static __device__ void add_bias(const uint16_t *arena, uint64_t offset, uint32_t len,
                         uint32_t dtype, float *output) {
  if (offset == kMissingOffset) {
    __syncthreads();
    return;
  }
  const uint16_t *bias = arena + offset;
  for (uint32_t index = threadIdx.x; index < len; index += blockDim.x) {
    output[index] += encoded_to_f32(bias[index], dtype);
  }
  __syncthreads();
}

static __device__ void per_head_rms_norm(uint16_t *arena, uint64_t offset, float *values,
                                  uint32_t heads, uint32_t head_dim,
                                  uint32_t dtype, float eps) {
  if (offset == kMissingOffset) {
    __syncthreads();
    return;
  }
  const uint16_t *weight = arena + offset;
  for (uint32_t head = 0; head < heads; ++head) {
    float mean_square = 0.0f;
    float *base = values + head * head_dim;
    for (uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
      mean_square += base[index] * base[index];
    }
    mean_square = block_sum(mean_square);
    const float scale = rsqrtf(mean_square / static_cast<float>(head_dim) + eps);
    for (uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
      base[index] *= scale * encoded_to_f32(weight[index], dtype);
    }
    __syncthreads();
  }
}

static __device__ void add_optional_head_bias(const uint16_t *arena, uint64_t offset,
                                       uint32_t head, uint32_t head_dim,
                                       uint32_t dtype, float *values) {
  if (offset != kMissingOffset) {
    const uint16_t *bias = arena + offset + static_cast<uint64_t>(head) * head_dim;
    for (uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
      values[index] += encoded_to_f32(bias[index], dtype);
    }
  }
  __syncthreads();
}

static __device__ void per_head_rms_norm_block(uint16_t *arena, uint64_t offset,
                                        float *values, uint32_t head_dim,
                                        uint32_t dtype, float eps) {
  if (offset == kMissingOffset) {
    __syncthreads();
    return;
  }
  const uint16_t *weight = arena + offset;
  float mean_square = 0.0f;
  for (uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
    mean_square += values[index] * values[index];
  }
  mean_square = block_sum(mean_square);
  const float scale = rsqrtf(mean_square / static_cast<float>(head_dim) + eps);
  for (uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
    values[index] *= scale * encoded_to_f32(weight[index], dtype);
  }
  __syncthreads();
}

static __device__ void apply_rope_head(float *values, uint32_t head_dim,
                                uint32_t position, float theta) {
  if (theta <= 0.0f) {
    __syncthreads();
    return;
  }
  const uint32_t half = head_dim / 2;
  for (uint32_t offset = threadIdx.x; offset < half; offset += blockDim.x) {
    const uint32_t second = offset + half;
    const float exponent =
        static_cast<float>(2 * offset) / static_cast<float>(head_dim);
    float angle = static_cast<float>(position) / powf(theta, exponent);
    float sin_value = 0.0f;
    float cos_value = 0.0f;
    sincosf(angle, &sin_value, &cos_value);
    const float left = values[offset];
    const float right = values[second];
    values[offset] = left * cos_value - right * sin_value;
    values[second] = right * cos_value + left * sin_value;
  }
  __syncthreads();
}

static __device__ void apply_rope(float *values, uint32_t heads, uint32_t head_dim,
                           uint32_t position, float theta) {
  if (theta <= 0.0f) {
    return;
  }
  const uint32_t half = head_dim / 2;
  const uint32_t total = heads * half;
  for (uint32_t index = threadIdx.x; index < total; index += blockDim.x) {
    const uint32_t head = index / half;
    const uint32_t offset = index % half;
    const uint32_t start = head * head_dim;
    const uint32_t first = start + offset;
    const uint32_t second = first + half;
    const float exponent = static_cast<float>(2 * offset) / static_cast<float>(head_dim);
    float angle = static_cast<float>(position) / powf(theta, exponent);
    float sin_value = 0.0f;
    float cos_value = 0.0f;
    sincosf(angle, &sin_value, &cos_value);
    const float left = values[first];
    const float right = values[second];
    values[first] = left * cos_value - right * sin_value;
    values[second] = right * cos_value + left * sin_value;
  }
  __syncthreads();
}

static __device__ __forceinline__ float f32_from_u16_slots(
    const uint16_t *slots, uint32_t index) {
  const uint32_t lo = static_cast<uint32_t>(slots[index * 2u]);
  const uint32_t hi = static_cast<uint32_t>(slots[index * 2u + 1u]);
  return __uint_as_float(lo | (hi << 16));
}

static __device__ __forceinline__ float softplus_device(float value) {
  return value <= 20.0f ? log1pf(expf(value)) : value;
}

static __device__ void normalize_linear_gdn_qk(float *query, float *key,
                                               uint32_t key_heads,
                                               uint32_t key_head_dim) {
  for (uint32_t head = 0; head < key_heads; ++head) {
    float *q_head = query + static_cast<uint64_t>(head) * key_head_dim;
    float *k_head = key + static_cast<uint64_t>(head) * key_head_dim;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint32_t index = threadIdx.x; index < key_head_dim;
         index += blockDim.x) {
      q_sum += q_head[index] * q_head[index];
      k_sum += k_head[index] * k_head[index];
    }
    q_sum = block_sum(q_sum);
    k_sum = block_sum(k_sum);
    const float q_scale = rsqrtf(q_sum + 1e-6f);
    const float k_scale = rsqrtf(k_sum + 1e-6f);
    const float query_dim_scale =
        rsqrtf(static_cast<float>(key_head_dim));
    for (uint32_t index = threadIdx.x; index < key_head_dim;
         index += blockDim.x) {
      q_head[index] *= q_scale * query_dim_scale;
      k_head[index] *= k_scale;
    }
    __syncthreads();
  }
}

static __device__ void run_linear_gdn_layer(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t intermediate, uint64_t input_offset,
    uint64_t output_offset, float rms_eps, float *scratch,
    float *linear_conv_state_base, float *linear_recurrent_state_base) {
  const uint32_t key_heads = layout.linear_key_heads;
  const uint32_t value_heads = layout.linear_value_heads;
  const uint32_t key_head_dim = layout.linear_key_head_dim;
  const uint32_t value_head_dim = layout.linear_value_head_dim;
  const uint32_t conv_kernel = layout.linear_conv_kernel;
  const uint32_t key_dim = key_heads * key_head_dim;
  const uint32_t value_dim = value_heads * value_head_dim;
  const uint32_t conv_dim = key_dim * 2u + value_dim;
  const uint32_t conv_state_len = conv_kernel == 0 ? 0 : conv_kernel - 1u;
  if (key_heads == 0 || value_heads == 0 || key_head_dim == 0 ||
      value_head_dim == 0 || conv_kernel == 0 || key_dim == 0 ||
      value_dim == 0 || conv_dim == 0 ||
      value_heads % key_heads != 0 ||
      linear_conv_state_base == nullptr ||
      linear_recurrent_state_base == nullptr ||
      layout.linear_conv_state == kMissingOffset ||
      layout.linear_recurrent_state == kMissingOffset) {
    copy_encoded_slice(arena + output_offset, arena + input_offset, hidden);
    return;
  }

  float *input = scratch;
  float *attn_norm = input + hidden;
  float *mixed = attn_norm + hidden;
  float *convolved = mixed + conv_dim;
  float *z = convolved + conv_dim;
  float *b = z + value_dim;
  float *a = b + value_heads;
  float *core = a + value_heads;
  float *normed_core = core + value_dim;
  float *residual = normed_core + value_dim;
  float *mlp_norm = residual + hidden;
  float *gate = mlp_norm + hidden;
  float *up = gate + intermediate;
  float *ff = up + intermediate;
  float *down = ff + intermediate;

  float *conv_state = linear_conv_state_base + layout.linear_conv_state;
  float *recurrent_state =
      linear_recurrent_state_base + layout.linear_recurrent_state;

  encoded_slice_to_f32(arena + input_offset, hidden, dtype, input);
  rms_norm(input, arena + layout.rms_attn, hidden, dtype, rms_eps, attn_norm);
  mat_vec(arena + layout.w_linear_qkv, attn_norm, conv_dim, hidden, dtype,
          mixed);
  mat_vec(arena + layout.w_linear_z, attn_norm, value_dim, hidden, dtype, z);
  mat_vec(arena + layout.w_linear_b, attn_norm, value_heads, hidden, dtype, b);
  mat_vec(arena + layout.w_linear_a, attn_norm, value_heads, hidden, dtype, a);

  for (uint32_t dim = threadIdx.x; dim < conv_dim; dim += blockDim.x) {
    const uint64_t weight_start = static_cast<uint64_t>(dim) * conv_kernel;
    const uint64_t state_start = static_cast<uint64_t>(dim) * conv_state_len;
    float acc = 0.0f;
    for (uint32_t tap = 0; tap < conv_state_len; ++tap) {
      acc += encoded_to_f32(arena[layout.w_linear_conv + weight_start + tap],
                            dtype) *
             conv_state[state_start + tap];
    }
    acc += encoded_to_f32(
               arena[layout.w_linear_conv + weight_start + conv_state_len],
               dtype) *
           mixed[dim];
    convolved[dim] = silu(acc);
    for (uint32_t tap = 1; tap < conv_state_len; ++tap) {
      conv_state[state_start + tap - 1u] = conv_state[state_start + tap];
    }
    if (conv_state_len != 0) {
      conv_state[state_start + conv_state_len - 1u] = mixed[dim];
    }
  }
  __syncthreads();

  float *query = convolved;
  float *key = convolved + key_dim;
  float *value = convolved + key_dim * 2u;
  normalize_linear_gdn_qk(query, key, key_heads, key_head_dim);
  const uint32_t value_heads_per_key = value_heads / key_heads;
  for (uint32_t value_head = 0; value_head < value_heads; ++value_head) {
    const uint32_t key_head = value_head / value_heads_per_key;
    const float *q_slice =
        query + static_cast<uint64_t>(key_head) * key_head_dim;
    const float *k_slice =
        key + static_cast<uint64_t>(key_head) * key_head_dim;
    const uint32_t value_start = value_head * value_head_dim;
    const float *v_slice = value + value_start;
    float *out_slice = core + value_start;
    float *state = recurrent_state +
                   static_cast<uint64_t>(value_head) * value_head_dim *
                       key_head_dim;
    const float a_log =
        f32_from_u16_slots(arena + layout.w_linear_a_log, value_head);
    const float dt =
        a[value_head] +
        encoded_to_f32(arena[layout.w_linear_dt_bias + value_head], dtype);
    const float decay = expf(-expf(a_log) * softplus_device(dt));
    const float beta = sigmoid(b[value_head]);
    for (uint32_t value_offset = threadIdx.x; value_offset < value_head_dim;
         value_offset += blockDim.x) {
      float *row = state + static_cast<uint64_t>(value_offset) * key_head_dim;
      float previous = 0.0f;
      for (uint32_t col = 0; col < key_head_dim; ++col) {
        row[col] *= decay;
        previous += row[col] * k_slice[col];
      }
      const float delta = (v_slice[value_offset] - previous) * beta;
      float projected = 0.0f;
      for (uint32_t col = 0; col < key_head_dim; ++col) {
        row[col] += delta * k_slice[col];
        projected += row[col] * q_slice[col];
      }
      out_slice[value_offset] = projected;
    }
    __syncthreads();
  }

  for (uint32_t value_head = 0; value_head < value_heads; ++value_head) {
    const uint32_t value_start = value_head * value_head_dim;
    float mean_square = 0.0f;
    for (uint32_t index = threadIdx.x; index < value_head_dim;
         index += blockDim.x) {
      const float item = core[value_start + index];
      mean_square += item * item;
    }
    mean_square = block_sum(mean_square);
    const float scale =
        rsqrtf(mean_square / static_cast<float>(value_head_dim) + rms_eps);
    for (uint32_t index = threadIdx.x; index < value_head_dim;
         index += blockDim.x) {
      const uint32_t offset = value_start + index;
      normed_core[offset] =
          core[offset] * scale *
          f32_from_u16_slots(arena + layout.w_linear_norm, index) *
          silu(z[offset]);
    }
    __syncthreads();
  }

  mat_vec(arena + layout.w_linear_out, normed_core, hidden, value_dim, dtype,
          residual);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    residual[index] += input[index];
  }
  __syncthreads();
  rms_norm(residual, arena + layout.rms_mlp, hidden, dtype, rms_eps, mlp_norm);
  if (layout.mlp_kind == kMlpKindSparseMoe) {
    run_sparse_moe_mlp(arena, layout, dtype, hidden, intermediate, mlp_norm,
                       gate, up, ff, down, input);
  } else {
    mat_vec(arena + layout.w_gate, mlp_norm, intermediate, hidden, dtype, gate);
    mat_vec(arena + layout.w_up, mlp_norm, intermediate, hidden, dtype, up);
    for (uint32_t index = threadIdx.x; index < intermediate;
         index += blockDim.x) {
      ff[index] = silu(gate[index]) * up[index];
    }
    __syncthreads();
    mat_vec(arena + layout.w_down, ff, hidden, intermediate, dtype, down);
  }
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    down[index] += residual[index];
  }
  __syncthreads();
  f32_slice_to_encoded(down, arena + output_offset, hidden, dtype);
}

static __device__ void run_layer(uint16_t *arena, SequenceLayerLayout layout,
                          uint32_t layer_index, uint64_t input_offset,
                          uint64_t output_offset, uint32_t dtype, uint32_t hidden,
                          uint32_t heads, uint32_t kv_heads, uint32_t head_dim,
                          uint32_t intermediate, uint32_t position, uint32_t max_steps,
                          float rms_eps, float rope_theta, float *scratch,
                          uint16_t *kv_keys, uint16_t *kv_values,
                          uint32_t kv_block_count,
                          const uint32_t *kv_block_table,
                          float *linear_conv_state,
                          float *linear_recurrent_state) {
  if (layout.attention_kind == kAttentionKindLinearGdn) {
    run_linear_gdn_layer(arena, layout, dtype, hidden, intermediate,
                         input_offset, output_offset, rms_eps, scratch,
                         linear_conv_state, linear_recurrent_state);
    return;
  }
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  float *input = scratch;
  float *attn_norm = input + hidden;
  float *q = attn_norm + hidden;
  float *k = q + attention_hidden;
  float *v = k + kv_hidden;
  float *attn = v + kv_hidden;
  float *residual = attn + attention_hidden;
  float *mlp_norm = residual + hidden;
  float *q_gate = mlp_norm + hidden;
  float *gate = q_gate + attention_hidden;
  float *up = gate + intermediate;
  float *ff = up + intermediate;
  float *down = ff + intermediate;

  encoded_slice_to_f32(arena + input_offset, hidden, dtype, input);
  rms_norm(input, arena + layout.rms_attn, hidden, dtype, rms_eps, attn_norm);
  mat_vec(arena + layout.w_q, attn_norm, attention_hidden, hidden, dtype, q);
  if (layout.w_q_gate != kMissingOffset) {
    mat_vec(arena + layout.w_q_gate, attn_norm, attention_hidden, hidden, dtype,
            q_gate);
  }
  mat_vec(arena + layout.w_k, attn_norm, kv_hidden, hidden, dtype, k);
  mat_vec(arena + layout.w_v, attn_norm, kv_hidden, hidden, dtype, v);
  add_bias(arena, layout.q_bias, attention_hidden, dtype, q);
  add_bias(arena, layout.k_bias, kv_hidden, dtype, k);
  add_bias(arena, layout.v_bias, kv_hidden, dtype, v);
  per_head_rms_norm(arena, layout.q_norm, q, heads, head_dim, dtype, rms_eps);
  per_head_rms_norm(arena, layout.k_norm, k, kv_heads, head_dim, dtype, rms_eps);
  apply_rope(q, heads, head_dim, position, rope_theta);
  apply_rope(k, kv_heads, head_dim, position, rope_theta);

  const uint64_t write_base = kv_cache_token_base(
      layer_index, kv_block_count, kv_block_table, position, kv_hidden, 0);
  for (uint32_t index = threadIdx.x; index < kv_hidden; index += blockDim.x) {
    kv_keys[write_base + index] = f32_to_encoded(k[index], dtype);
    kv_values[write_base + index] = f32_to_encoded(v[index], dtype);
  }
  __syncthreads();

  const float scale = rsqrtf(static_cast<float>(head_dim));
  for (uint32_t head = threadIdx.x; head < heads; head += blockDim.x) {
    const uint32_t kv_head = head / (heads / kv_heads);
    const uint32_t head_start = head * head_dim;
    float local_m = -INFINITY;
    float local_l = 0.0f;
    for (uint32_t offset = 0; offset < head_dim; ++offset) {
      attn[head_start + offset] = 0.0f;
    }
    for (uint32_t token = 0; token <= position; ++token) {
      const uint64_t token_base = kv_cache_token_base(
          layer_index, kv_block_count, kv_block_table, token, kv_hidden,
          kv_head * head_dim);
      float score = 0.0f;
      for (uint32_t offset = 0; offset < head_dim; ++offset) {
        score += q[head_start + offset] *
                 encoded_to_f32(kv_keys[token_base + offset], dtype);
      }
      score *= scale;
      const float next_m = fmaxf(local_m, score);
      const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
      const float new_scale = expf(score - next_m);
      for (uint32_t offset = 0; offset < head_dim; ++offset) {
        const uint32_t out = head_start + offset;
        attn[out] =
            attn[out] * old_scale +
            encoded_to_f32(kv_values[token_base + offset], dtype) * new_scale;
      }
      local_l = local_l * old_scale + new_scale;
      local_m = next_m;
    }
    if (local_l > 0.0f && isfinite(local_l)) {
      for (uint32_t offset = 0; offset < head_dim; ++offset) {
        attn[head_start + offset] /= local_l;
      }
    }
  }
  __syncthreads();
  if (layout.w_q_gate != kMissingOffset) {
    for (uint32_t index = threadIdx.x; index < attention_hidden;
         index += blockDim.x) {
      attn[index] *= sigmoid(q_gate[index]);
    }
    __syncthreads();
  }
  mat_vec(arena + layout.w_o, attn, hidden, attention_hidden, dtype, residual);
  add_bias(arena, layout.o_bias, hidden, dtype, residual);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    residual[index] += input[index];
  }
  __syncthreads();

  rms_norm(residual, arena + layout.rms_mlp, hidden, dtype, rms_eps, mlp_norm);
  if (layout.mlp_kind == kMlpKindSparseMoe) {
    run_sparse_moe_mlp(arena, layout, dtype, hidden, intermediate, mlp_norm,
                       gate, up, ff, down, input);
  } else {
    mat_vec(arena + layout.w_gate, mlp_norm, intermediate, hidden, dtype, gate);
    mat_vec(arena + layout.w_up, mlp_norm, intermediate, hidden, dtype, up);
    for (uint32_t index = threadIdx.x; index < intermediate; index += blockDim.x) {
      ff[index] = silu(gate[index]) * up[index];
    }
    __syncthreads();
    mat_vec(arena + layout.w_down, ff, hidden, intermediate, dtype, down);
  }
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    down[index] += residual[index];
  }
  __syncthreads();
  f32_slice_to_encoded(down, arena + output_offset, hidden, dtype);
}
