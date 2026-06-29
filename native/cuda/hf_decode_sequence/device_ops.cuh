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

static __device__ void run_layer(uint16_t *arena, SequenceLayerLayout layout,
                          uint32_t layer_index, uint64_t input_offset,
                          uint64_t output_offset, uint32_t dtype, uint32_t hidden,
                          uint32_t heads, uint32_t kv_heads, uint32_t head_dim,
                          uint32_t intermediate, uint32_t position, uint32_t max_steps,
                          float rms_eps, float rope_theta, float *scratch,
                          uint16_t *kv_keys, uint16_t *kv_values,
                          uint32_t kv_block_count,
                          const uint32_t *kv_block_table) {
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
  float *gate = mlp_norm + hidden;
  float *up = gate + intermediate;
  float *ff = up + intermediate;
  float *down = ff + intermediate;

  encoded_slice_to_f32(arena + input_offset, hidden, dtype, input);
  rms_norm(input, arena + layout.rms_attn, hidden, dtype, rms_eps, attn_norm);
  mat_vec(arena + layout.w_q, attn_norm, attention_hidden, hidden, dtype, q);
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
  mat_vec(arena + layout.w_o, attn, hidden, attention_hidden, dtype, residual);
  add_bias(arena, layout.o_bias, hidden, dtype, residual);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    residual[index] += input[index];
  }
  __syncthreads();

  rms_norm(residual, arena + layout.rms_mlp, hidden, dtype, rms_eps, mlp_norm);
  mat_vec(arena + layout.w_gate, mlp_norm, intermediate, hidden, dtype, gate);
  mat_vec(arena + layout.w_up, mlp_norm, intermediate, hidden, dtype, up);
  for (uint32_t index = threadIdx.x; index < intermediate; index += blockDim.x) {
    ff[index] = silu(gate[index]) * up[index];
  }
  __syncthreads();
  mat_vec(arena + layout.w_down, ff, hidden, intermediate, dtype, down);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    down[index] += residual[index];
  }
  __syncthreads();
  f32_slice_to_encoded(down, arena + output_offset, hidden, dtype);
}
