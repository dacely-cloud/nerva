#include "kernels.cuh"

#include "device_ops.cuh"

#include <stdint.h>

template <uint32_t DType>
__global__ void hf_layer_attention_chunk_kernel(
    uint32_t layer_index, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, uint32_t attention_chunks, float *scratch,
    const uint16_t *kv_keys, const uint16_t *kv_values, float *partial_values,
    float *partial_m, float *partial_l, uint32_t kv_block_count,
    const uint32_t *kv_block_table) {
  __shared__ uint32_t current_position_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  (void)max_steps;
  if (head_dim > blockDim.x * kHeadThreadElements) {
    return;
  }
  const uint32_t head = blockIdx.x;
  const uint32_t chunk = blockIdx.y;
  if (head >= heads || chunk >= attention_chunks) {
    return;
  }
  const uint32_t chunk_start = chunk * kDecodeAttentionChunkTokens;
  const uint64_t slot =
      (static_cast<uint64_t>(head) * attention_chunks + chunk);
  if (chunk_start > current_position) {
    if (threadIdx.x == 0) {
      partial_m[slot] = -INFINITY;
      partial_l[slot] = 0.0f;
    }
    return;
  }
  const uint32_t chunk_limit = chunk_start + kDecodeAttentionChunkTokens;
  const uint32_t position_limit = current_position + 1u;
  const uint32_t chunk_end =
      chunk_limit < position_limit ? chunk_limit : position_limit;
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  const uint32_t group = heads / kv_heads;
  const uint32_t kv_head = head / group;
  const uint32_t head_start = head * head_dim;
  const uint32_t kv_start = kv_head * head_dim;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float local_m = -INFINITY;
  float local_l = 0.0f;
  float q_frag[kHeadThreadElements];
  float acc[kHeadThreadElements];
#pragma unroll
  for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
    const uint32_t offset = threadIdx.x + item * blockDim.x;
    q_frag[item] = offset < head_dim ? s.q[head_start + offset] : 0.0f;
    acc[item] = 0.0f;
  }
  for (uint32_t token = chunk_start; token < chunk_end;) {
    const uint32_t logical_block = token / kKvCacheBlockTokens;
    const uint32_t logical_block_end =
        (logical_block + 1u) * kKvCacheBlockTokens;
    const uint32_t block_limit =
        logical_block_end < chunk_end ? logical_block_end : chunk_end;
    uint64_t token_base = kv_cache_page_offset(
        layer_index, kv_block_count, kv_block_table[logical_block],
        token - logical_block * kKvCacheBlockTokens, kv_hidden, kv_start);
    for (; token < block_limit; ++token, token_base += kv_hidden) {
      float partial = 0.0f;
#pragma unroll
      for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
        const uint32_t offset = threadIdx.x + item * blockDim.x;
        if (offset < head_dim) {
          partial += q_frag[item] *
                     encoded_to_f32_typed<DType>(kv_keys[token_base + offset]);
        }
      }
      const float score = block_sum(partial) * scale;
      const float next_m = fmaxf(local_m, score);
      const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
      const float new_scale = expf(score - next_m);
#pragma unroll
      for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
        const uint32_t offset = threadIdx.x + item * blockDim.x;
        if (offset < head_dim) {
          acc[item] =
              acc[item] * old_scale +
              encoded_to_f32_typed<DType>(kv_values[token_base + offset]) *
                  new_scale;
        }
      }
      local_l = local_l * old_scale + new_scale;
      local_m = next_m;
    }
  }
  if (threadIdx.x == 0) {
    partial_m[slot] = local_m;
    partial_l[slot] = local_l;
  }
#pragma unroll
  for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
    const uint32_t offset = threadIdx.x + item * blockDim.x;
    if (offset < head_dim) {
      partial_values[slot * head_dim + offset] = acc[item];
    }
  }
}

template <uint32_t DType>
__global__ void hf_layer_grouped_gqa_attention_chunk_kernel(
    uint32_t layer_index, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, uint32_t attention_chunks, float *scratch,
    const uint16_t *kv_keys, const uint16_t *kv_values, float *partial_values,
    float *partial_m, float *partial_l, uint32_t kv_block_count,
    const uint32_t *kv_block_table) {
  __shared__ uint32_t current_position_shared;
  __shared__ float shared_k[kGroupedGqaHeadDimMax];
  __shared__ float shared_v[kGroupedGqaHeadDimMax];
  __shared__ float reduce[kGroupedGqaHeads][2];
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  (void)max_steps;
  if (kv_heads == 0 ||
      heads / kv_heads != kGroupedGqaHeads || heads % kv_heads != 0 ||
      head_dim > kGroupedGqaHeadDimMax ||
      blockDim.x != kGroupedGqaThreads) {
    return;
  }
  const uint32_t kv_head = blockIdx.x;
  const uint32_t chunk = blockIdx.y;
  if (kv_head >= kv_heads || chunk >= attention_chunks) {
    return;
  }
  const uint32_t chunk_start = chunk * kDecodeAttentionChunkTokens;
  if (chunk_start > current_position) {
    if (threadIdx.x < kGroupedGqaHeads) {
      const uint32_t head = kv_head * kGroupedGqaHeads + threadIdx.x;
      const uint64_t slot =
          (static_cast<uint64_t>(head) * attention_chunks + chunk);
      partial_m[slot] = -INFINITY;
      partial_l[slot] = 0.0f;
    }
    return;
  }
  const uint32_t chunk_limit = chunk_start + kDecodeAttentionChunkTokens;
  const uint32_t position_limit = current_position + 1u;
  const uint32_t chunk_end =
      chunk_limit < position_limit ? chunk_limit : position_limit;
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  const uint32_t kv_start = kv_head * head_dim;
  const uint32_t q_in_group = threadIdx.x / kGroupedGqaThreadsPerHead;
  const uint32_t lane = threadIdx.x - q_in_group * kGroupedGqaThreadsPerHead;
  const uint32_t lane_in_warp = lane & 31u;
  const uint32_t warp_in_group = lane >> 5u;
  const uint32_t head = kv_head * kGroupedGqaHeads + q_in_group;
  const uint32_t head_start = head * head_dim;
  const uint64_t slot =
      (static_cast<uint64_t>(head) * attention_chunks + chunk);
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float local_m = -INFINITY;
  float local_l = 0.0f;
  float q_frag[kHeadThreadElements];
  float acc[kHeadThreadElements];
#pragma unroll
  for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
    const uint32_t offset = lane + item * kGroupedGqaThreadsPerHead;
    q_frag[item] = offset < head_dim ? s.q[head_start + offset] : 0.0f;
    acc[item] = 0.0f;
  }
  for (uint32_t token = chunk_start; token < chunk_end;) {
    const uint32_t logical_block = token / kKvCacheBlockTokens;
    const uint32_t logical_block_end =
        (logical_block + 1u) * kKvCacheBlockTokens;
    const uint32_t block_limit =
        logical_block_end < chunk_end ? logical_block_end : chunk_end;
    uint64_t token_base = kv_cache_page_offset(
        layer_index, kv_block_count, kv_block_table[logical_block],
        token - logical_block * kKvCacheBlockTokens, kv_hidden, kv_start);
    for (; token < block_limit; ++token, token_base += kv_hidden) {
      for (uint32_t offset = threadIdx.x; offset < head_dim;
           offset += blockDim.x) {
        shared_k[offset] =
            encoded_to_f32_typed<DType>(kv_keys[token_base + offset]);
        shared_v[offset] =
            encoded_to_f32_typed<DType>(kv_values[token_base + offset]);
      }
      __syncthreads();
      float partial = 0.0f;
#pragma unroll
      for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
        const uint32_t offset = lane + item * kGroupedGqaThreadsPerHead;
        if (offset < head_dim) {
          partial += q_frag[item] * shared_k[offset];
        }
      }
      const float partial_sum = warp_sum(partial);
      if (lane_in_warp == 0) {
        reduce[q_in_group][warp_in_group] = partial_sum;
      }
      __syncthreads();
      float old_scale = 0.0f;
      float new_scale = 0.0f;
      float next_m_value = local_m;
      float next_l_value = local_l;
      if (lane_in_warp == 0) {
        const float score =
            (reduce[q_in_group][0] + reduce[q_in_group][1]) * scale;
        const float next_m = fmaxf(local_m, score);
        old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
        new_scale = expf(score - next_m);
        next_l_value = local_l * old_scale + new_scale;
        next_m_value = next_m;
      }
      old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
      new_scale = __shfl_sync(0xffffffffu, new_scale, 0);
      local_l = __shfl_sync(0xffffffffu, next_l_value, 0);
      local_m = __shfl_sync(0xffffffffu, next_m_value, 0);
#pragma unroll
      for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
        const uint32_t offset = lane + item * kGroupedGqaThreadsPerHead;
        if (offset < head_dim) {
          acc[item] = acc[item] * old_scale + shared_v[offset] * new_scale;
        }
      }
      __syncthreads();
    }
  }
  if (lane == 0) {
    partial_m[slot] = local_m;
    partial_l[slot] = local_l;
  }
#pragma unroll
  for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
    const uint32_t offset = lane + item * kGroupedGqaThreadsPerHead;
    if (offset < head_dim) {
      partial_values[slot * head_dim + offset] = acc[item];
    }
  }
}

template <uint32_t DType>
__global__ void hf_layer_shared_warp_gqa_attention_chunk_kernel(
    uint32_t layer_index, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, uint32_t attention_chunks, float *scratch,
    const uint16_t *kv_keys, const uint16_t *kv_values, float *partial_values,
    float *partial_m, float *partial_l, uint32_t kv_block_count,
    const uint32_t *kv_block_table) {
  __shared__ uint32_t current_position_shared;
  __shared__ float shared_k[kSharedWarpGqaTileElements];
  __shared__ float shared_v[kSharedWarpGqaTileElements];
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  (void)max_steps;
  if (kv_heads == 0 ||
      heads / kv_heads != kGroupedGqaHeads || heads % kv_heads != 0 ||
      head_dim > kSharedWarpGqaHeadDimMax ||
      blockDim.x != kSharedWarpGqaThreads) {
    return;
  }
  const uint32_t kv_head = blockIdx.x;
  const uint32_t chunk = blockIdx.y;
  if (kv_head >= kv_heads || chunk >= attention_chunks) {
    return;
  }
  const uint32_t q_in_group = threadIdx.x / kSharedWarpGqaThreadsPerHead;
  const uint32_t lane = threadIdx.x - q_in_group * kSharedWarpGqaThreadsPerHead;
  const uint32_t head = kv_head * kGroupedGqaHeads + q_in_group;
  const uint64_t slot =
      (static_cast<uint64_t>(head) * attention_chunks + chunk);
  const uint32_t chunk_start = chunk * kDecodeAttentionChunkTokens;
  if (chunk_start > current_position) {
    if (lane == 0) {
      partial_m[slot] = -INFINITY;
      partial_l[slot] = 0.0f;
    }
    return;
  }
  const uint32_t chunk_limit = chunk_start + kDecodeAttentionChunkTokens;
  const uint32_t position_limit = current_position + 1u;
  const uint32_t chunk_end =
      chunk_limit < position_limit ? chunk_limit : position_limit;
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  const uint32_t kv_start = kv_head * head_dim;
  const uint32_t head_start = head * head_dim;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float local_m = -INFINITY;
  float local_l = 0.0f;
  float q_frag[kHeadThreadElements];
  float acc[kHeadThreadElements];
#pragma unroll
  for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
    const uint32_t offset = lane + item * kSharedWarpGqaThreadsPerHead;
    q_frag[item] = offset < head_dim ? s.q[head_start + offset] : 0.0f;
    acc[item] = 0.0f;
  }
  for (uint32_t token = chunk_start; token < chunk_end;) {
    const uint32_t logical_block = token / kKvCacheBlockTokens;
    const uint32_t logical_block_end =
        (logical_block + 1u) * kKvCacheBlockTokens;
    const uint32_t block_limit =
        logical_block_end < chunk_end ? logical_block_end : chunk_end;
    uint64_t block_token_base = kv_cache_page_offset(
        layer_index, kv_block_count, kv_block_table[logical_block],
        token - logical_block * kKvCacheBlockTokens, kv_hidden, kv_start);
    while (token < block_limit) {
      const uint32_t remaining = block_limit - token;
      const uint32_t tile_count = remaining < kSharedWarpGqaTileTokens
                                      ? remaining
                                      : kSharedWarpGqaTileTokens;
      const uint32_t tile_elements = tile_count * head_dim;
      for (uint32_t index = threadIdx.x; index < tile_elements;
           index += blockDim.x) {
        const uint32_t tile_token = index / head_dim;
        const uint32_t offset = index - tile_token * head_dim;
        const uint64_t source = block_token_base +
                                static_cast<uint64_t>(tile_token) * kv_hidden +
                                offset;
        shared_k[index] = encoded_to_f32_typed<DType>(kv_keys[source]);
        shared_v[index] = encoded_to_f32_typed<DType>(kv_values[source]);
      }
      __syncthreads();
      for (uint32_t tile_token = 0; tile_token < tile_count; ++tile_token) {
        const uint32_t tile_base = tile_token * head_dim;
        float partial = 0.0f;
#pragma unroll
        for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
          const uint32_t offset = lane + item * kSharedWarpGqaThreadsPerHead;
          if (offset < head_dim) {
            partial += q_frag[item] * shared_k[tile_base + offset];
          }
        }
        const float score = warp_sum(partial) * scale;
        float old_scale = 0.0f;
        float new_scale = 0.0f;
        float next_m_value = local_m;
        float next_l_value = local_l;
        if (lane == 0) {
          const float next_m = fmaxf(local_m, score);
          old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
          new_scale = expf(score - next_m);
          next_l_value = local_l * old_scale + new_scale;
          next_m_value = next_m;
        }
        old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
        new_scale = __shfl_sync(0xffffffffu, new_scale, 0);
        local_l = __shfl_sync(0xffffffffu, next_l_value, 0);
        local_m = __shfl_sync(0xffffffffu, next_m_value, 0);
#pragma unroll
        for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
          const uint32_t offset = lane + item * kSharedWarpGqaThreadsPerHead;
          if (offset < head_dim) {
            acc[item] =
                acc[item] * old_scale + shared_v[tile_base + offset] * new_scale;
          }
        }
      }
      __syncthreads();
      token += tile_count;
      block_token_base += static_cast<uint64_t>(tile_count) * kv_hidden;
    }
  }
  if (lane == 0) {
    partial_m[slot] = local_m;
    partial_l[slot] = local_l;
  }
#pragma unroll
  for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
    const uint32_t offset = lane + item * kSharedWarpGqaThreadsPerHead;
    if (offset < head_dim) {
      partial_values[slot * head_dim + offset] = acc[item];
    }
  }
}

template <uint32_t DType>
__global__ void hf_prefill_grouped_gqa_attention_kernel(
    uint32_t layer_index, uint32_t heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t max_steps, uint32_t chunk_start, uint32_t chunk_tokens,
    const float *qkv, const uint16_t *kv_keys, const uint16_t *kv_values,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint16_t *attn_out) {
  __shared__ float shared_k[kGroupedGqaHeadDimMax];
  __shared__ float shared_v[kGroupedGqaHeadDimMax];
  const uint32_t local_token = blockIdx.x;
  const uint32_t kv_head = blockIdx.y;
  if (local_token >= chunk_tokens || kv_head >= kv_heads ||
      kv_heads == 0 || heads / kv_heads != kGroupedGqaHeads ||
      heads % kv_heads != 0 || head_dim > kSharedWarpGqaHeadDimMax ||
      blockDim.x != kSharedWarpGqaThreads) {
    return;
  }
  (void)max_steps;
  const uint32_t q_in_group = threadIdx.x / kSharedWarpGqaThreadsPerHead;
  const uint32_t lane = threadIdx.x - q_in_group * kSharedWarpGqaThreadsPerHead;
  const uint32_t head = kv_head * kGroupedGqaHeads + q_in_group;
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  const uint64_t rows = static_cast<uint64_t>(attention_hidden) + kv_hidden * 2;
  const float *q = qkv + static_cast<uint64_t>(local_token) * rows;
  uint16_t *out = attn_out + static_cast<uint64_t>(local_token) * attention_hidden;
  const uint32_t global_pos = chunk_start + local_token;
  const uint32_t head_start = head * head_dim;
  const uint32_t kv_start = kv_head * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float local_m = -INFINITY;
  float local_l = 0.0f;
  float q_frag[kHeadThreadElements];
  float acc[kHeadThreadElements];
#pragma unroll
  for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
    const uint32_t offset = lane + item * kSharedWarpGqaThreadsPerHead;
    q_frag[item] = offset < head_dim ? q[head_start + offset] : 0.0f;
    acc[item] = 0.0f;
  }
  for (uint32_t token = 0; token <= global_pos;) {
    const uint32_t logical_block = token / kKvCacheBlockTokens;
    const uint32_t logical_block_end =
        (logical_block + 1u) * kKvCacheBlockTokens;
    const uint32_t block_limit =
        logical_block_end <= global_pos ? logical_block_end : global_pos + 1u;
    uint64_t token_base = kv_cache_page_offset(
        layer_index, kv_block_count, kv_block_table[logical_block],
        token - logical_block * kKvCacheBlockTokens, kv_hidden, kv_start);
    for (; token < block_limit; ++token, token_base += kv_hidden) {
      for (uint32_t offset = threadIdx.x; offset < head_dim;
           offset += blockDim.x) {
        shared_k[offset] =
            encoded_to_f32_typed<DType>(kv_keys[token_base + offset]);
        shared_v[offset] =
            encoded_to_f32_typed<DType>(kv_values[token_base + offset]);
      }
      __syncthreads();
      float partial = 0.0f;
#pragma unroll
      for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
        const uint32_t offset = lane + item * kSharedWarpGqaThreadsPerHead;
        if (offset < head_dim) {
          partial += q_frag[item] * shared_k[offset];
        }
      }
      const float score = warp_sum(partial) * scale;
      float old_scale = 0.0f;
      float new_scale = 0.0f;
      float next_m_value = local_m;
      float next_l_value = local_l;
      if (lane == 0) {
        const float next_m = fmaxf(local_m, score);
        old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
        new_scale = expf(score - next_m);
        next_l_value = local_l * old_scale + new_scale;
        next_m_value = next_m;
      }
      old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
      new_scale = __shfl_sync(0xffffffffu, new_scale, 0);
      local_l = __shfl_sync(0xffffffffu, next_l_value, 0);
      local_m = __shfl_sync(0xffffffffu, next_m_value, 0);
#pragma unroll
      for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
        const uint32_t offset = lane + item * kSharedWarpGqaThreadsPerHead;
        if (offset < head_dim) {
          acc[item] = acc[item] * old_scale + shared_v[offset] * new_scale;
        }
      }
      __syncthreads();
    }
  }
  const bool normalize = local_l > 0.0f && isfinite(local_l);
#pragma unroll
  for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
    const uint32_t offset = lane + item * kSharedWarpGqaThreadsPerHead;
    if (offset < head_dim) {
      const float value = normalize ? acc[item] / local_l : acc[item];
      out[head_start + offset] = f32_to_encoded(value, DType);
    }
  }
}

template <uint32_t DType>
__global__ void hf_prefill_grouped_gqa_attention_direct_kernel(
    uint32_t layer_index, uint32_t heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t max_steps, uint32_t chunk_start, uint32_t chunk_tokens,
    const float *qkv, const uint16_t *kv_keys, const uint16_t *kv_values,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint16_t *attn_out) {
  const uint32_t local_token = blockIdx.x;
  const uint32_t kv_head = blockIdx.y;
  if (local_token >= chunk_tokens || kv_head >= kv_heads ||
      kv_heads == 0 || heads / kv_heads != kGroupedGqaHeads ||
      heads % kv_heads != 0 || head_dim > kSharedWarpGqaHeadDimMax ||
      blockDim.x != kSharedWarpGqaThreads) {
    return;
  }
  (void)max_steps;
  const uint32_t q_in_group = threadIdx.x / kSharedWarpGqaThreadsPerHead;
  const uint32_t lane = threadIdx.x - q_in_group * kSharedWarpGqaThreadsPerHead;
  const uint32_t head = kv_head * kGroupedGqaHeads + q_in_group;
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  const uint64_t rows = static_cast<uint64_t>(attention_hidden) + kv_hidden * 2;
  const float *q = qkv + static_cast<uint64_t>(local_token) * rows;
  uint16_t *out = attn_out + static_cast<uint64_t>(local_token) * attention_hidden;
  const uint32_t global_pos = chunk_start + local_token;
  const uint32_t head_start = head * head_dim;
  const uint32_t kv_start = kv_head * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float local_m = -INFINITY;
  float local_l = 0.0f;
  float q_frag[kHeadThreadElements];
  float acc[kHeadThreadElements];
#pragma unroll
  for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
    const uint32_t offset = lane + item * kSharedWarpGqaThreadsPerHead;
    q_frag[item] = offset < head_dim ? q[head_start + offset] : 0.0f;
    acc[item] = 0.0f;
  }
  for (uint32_t token = 0; token <= global_pos;) {
    const uint32_t logical_block = token / kKvCacheBlockTokens;
    const uint32_t logical_block_end =
        (logical_block + 1u) * kKvCacheBlockTokens;
    const uint32_t block_limit =
        logical_block_end <= global_pos ? logical_block_end : global_pos + 1u;
    uint64_t token_base = kv_cache_page_offset(
        layer_index, kv_block_count, kv_block_table[logical_block],
        token - logical_block * kKvCacheBlockTokens, kv_hidden, kv_start);
    for (; token < block_limit; ++token, token_base += kv_hidden) {
      float partial = 0.0f;
#pragma unroll
      for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
        const uint32_t offset = lane + item * kSharedWarpGqaThreadsPerHead;
        if (offset < head_dim) {
          partial += q_frag[item] *
                     encoded_to_f32_typed<DType>(kv_keys[token_base + offset]);
        }
      }
      const float score = warp_sum(partial) * scale;
      float old_scale = 0.0f;
      float new_scale = 0.0f;
      float next_m_value = local_m;
      float next_l_value = local_l;
      if (lane == 0) {
        const float next_m = fmaxf(local_m, score);
        old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
        new_scale = expf(score - next_m);
        next_l_value = local_l * old_scale + new_scale;
        next_m_value = next_m;
      }
      old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
      new_scale = __shfl_sync(0xffffffffu, new_scale, 0);
      local_l = __shfl_sync(0xffffffffu, next_l_value, 0);
      local_m = __shfl_sync(0xffffffffu, next_m_value, 0);
#pragma unroll
      for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
        const uint32_t offset = lane + item * kSharedWarpGqaThreadsPerHead;
        if (offset < head_dim) {
          acc[item] =
              acc[item] * old_scale +
              encoded_to_f32_typed<DType>(kv_values[token_base + offset]) *
                  new_scale;
        }
      }
    }
  }
  const bool normalize = local_l > 0.0f && isfinite(local_l);
#pragma unroll
  for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
    const uint32_t offset = lane + item * kSharedWarpGqaThreadsPerHead;
    if (offset < head_dim) {
      const float value = normalize ? acc[item] / local_l : acc[item];
      out[head_start + offset] = f32_to_encoded(value, DType);
    }
  }
}

void launch_hf_layer_attention_chunk_kernel(
    cudaStream_t stream, dim3 grid, uint32_t dtype,
    bool use_shared_warp_gqa, bool use_grouped_gqa, uint32_t dense_threads,
    uint32_t layer_index, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, uint32_t attention_chunks, float *scratch,
    const uint16_t *kv_keys, const uint16_t *kv_values, float *partial_values,
    float *partial_m, float *partial_l, uint32_t kv_block_count,
    const uint32_t *kv_block_table) {
  if (dtype == kDTypeBF16 && use_shared_warp_gqa) {
    hf_layer_shared_warp_gqa_attention_chunk_kernel<kDTypeBF16>
        <<<grid, kSharedWarpGqaThreads, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, attention_chunks, scratch, kv_keys,
            kv_values, partial_values, partial_m, partial_l, kv_block_count,
            kv_block_table);
  } else if (dtype == kDTypeBF16 && use_grouped_gqa) {
    hf_layer_grouped_gqa_attention_chunk_kernel<kDTypeBF16>
        <<<grid, kGroupedGqaThreads, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, attention_chunks, scratch, kv_keys,
            kv_values, partial_values, partial_m, partial_l, kv_block_count,
            kv_block_table);
  } else if (dtype == kDTypeBF16) {
    hf_layer_attention_chunk_kernel<kDTypeBF16>
        <<<grid, dense_threads, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, attention_chunks, scratch, kv_keys,
            kv_values, partial_values, partial_m, partial_l, kv_block_count,
            kv_block_table);
  } else if (use_shared_warp_gqa) {
    hf_layer_shared_warp_gqa_attention_chunk_kernel<kDTypeF16>
        <<<grid, kSharedWarpGqaThreads, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, attention_chunks, scratch, kv_keys,
            kv_values, partial_values, partial_m, partial_l, kv_block_count,
            kv_block_table);
  } else if (use_grouped_gqa) {
    hf_layer_grouped_gqa_attention_chunk_kernel<kDTypeF16>
        <<<grid, kGroupedGqaThreads, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, attention_chunks, scratch, kv_keys,
            kv_values, partial_values, partial_m, partial_l, kv_block_count,
            kv_block_table);
  } else {
    hf_layer_attention_chunk_kernel<kDTypeF16>
        <<<grid, dense_threads, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, attention_chunks, scratch, kv_keys,
            kv_values, partial_values, partial_m, partial_l, kv_block_count,
            kv_block_table);
  }
}

void launch_hf_prefill_grouped_gqa_attention_direct_kernel(
    cudaStream_t stream, dim3 grid, uint32_t dtype, uint32_t layer_index,
    uint32_t heads, uint32_t kv_heads, uint32_t head_dim, uint32_t max_steps,
    uint32_t chunk_start, uint32_t chunk_tokens, const float *qkv,
    const uint16_t *kv_keys, const uint16_t *kv_values,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint16_t *attn_out) {
  if (dtype == kDTypeBF16) {
    hf_prefill_grouped_gqa_attention_direct_kernel<kDTypeBF16>
        <<<grid, kSharedWarpGqaThreads, 0, stream>>>(
            layer_index, heads, kv_heads, head_dim, max_steps, chunk_start,
            chunk_tokens, qkv, kv_keys, kv_values, kv_block_count,
            kv_block_table, attn_out);
  } else {
    hf_prefill_grouped_gqa_attention_direct_kernel<kDTypeF16>
        <<<grid, kSharedWarpGqaThreads, 0, stream>>>(
            layer_index, heads, kv_heads, head_dim, max_steps, chunk_start,
            chunk_tokens, qkv, kv_keys, kv_values, kv_block_count,
            kv_block_table, attn_out);
  }
}
