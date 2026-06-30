#include "kernels.cuh"

#include "device_ops.cuh"

#include <stdint.h>

template <uint32_t DType>
__global__ void hf_experimental_qk_page_selector_kernel(
    uint32_t layer_index, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, uint32_t selected_pages, uint32_t local_window_tokens,
    uint32_t sink_tokens, float *scratch, const uint16_t *kv_keys,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint32_t *candidate_pages) {
  __shared__ uint32_t current_position_shared;
  __shared__ float best_score_shared;
  __shared__ uint32_t best_page_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
    best_score_shared = -INFINITY;
    best_page_shared = 0;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  (void)max_steps;
  if (kv_heads == 0 || selected_pages == 0 || candidate_pages == nullptr) {
    return;
  }
  const uint32_t kv_head = blockIdx.x;
  const uint32_t far_slot = blockIdx.y;
  if (kv_head >= kv_heads || heads % kv_heads != 0 ||
      head_dim > blockDim.x * kHeadThreadElements) {
    return;
  }

  const uint32_t active_pages =
      (current_position + kDecodeAttentionChunkTokens) /
      kDecodeAttentionChunkTokens;
  if (active_pages == 0) {
    return;
  }
  const uint32_t current_page =
      current_position / kDecodeAttentionChunkTokens;
  const uint32_t sink_pages =
      (sink_tokens + kDecodeAttentionChunkTokens - 1u) /
      kDecodeAttentionChunkTokens;
  const uint32_t raw_local_pages =
      (local_window_tokens + kDecodeAttentionChunkTokens - 1u) /
      kDecodeAttentionChunkTokens;
  uint32_t local_start =
      current_page + 1u > raw_local_pages
          ? current_page + 1u - raw_local_pages
          : 0u;
  if (local_start < sink_pages) {
    local_start = sink_pages;
  }
  const uint32_t local_pages =
      current_page >= local_start ? current_page - local_start + 1u : 0u;
  const uint32_t local_limit = sink_pages + local_pages;
  const uint64_t out_base = static_cast<uint64_t>(kv_head) * selected_pages;

  if (far_slot == 0 && threadIdx.x == 0) {
    const uint32_t sink_limit =
        sink_pages < selected_pages ? sink_pages : selected_pages;
    for (uint32_t slot = 0; slot < sink_limit; ++slot) {
      candidate_pages[out_base + slot] =
          slot < active_pages ? slot : active_pages - 1u;
    }
    const uint32_t local_slot_limit =
        local_limit < selected_pages ? local_limit : selected_pages;
    for (uint32_t slot = sink_pages; slot < local_slot_limit; ++slot) {
      const uint32_t page = local_start + (slot - sink_pages);
      candidate_pages[out_base + slot] =
          page < active_pages ? page : active_pages - 1u;
    }
    const uint32_t covered_far_slots = gridDim.y;
    for (uint32_t slot = local_limit + covered_far_slots; slot < selected_pages;
         ++slot) {
      candidate_pages[out_base + slot] =
          current_page < active_pages ? current_page : active_pages - 1u;
    }
  }

  const uint32_t selected_slot = local_limit + far_slot;
  if (selected_slot >= selected_pages) {
    return;
  }
  const uint64_t out_index = out_base + selected_slot;
  if (selected_slot < local_limit) {
    if (threadIdx.x == 0) {
      candidate_pages[out_index] =
          current_page < active_pages ? current_page : active_pages - 1u;
    }
    return;
  }

  const uint32_t far_start = sink_pages < active_pages ? sink_pages : active_pages;
  const uint32_t far_end = local_start < active_pages ? local_start : active_pages;
  const uint32_t far_pages = far_end > far_start ? far_end - far_start : 0u;
  const uint32_t far_slots =
      selected_pages > local_limit ? selected_pages - local_limit : 0u;
  if (far_pages == 0 || far_slots == 0 || far_slot >= far_slots) {
    if (threadIdx.x == 0) {
      candidate_pages[out_index] =
          current_page < active_pages ? current_page : active_pages - 1u;
    }
    return;
  }

  const uint32_t begin =
      (static_cast<uint64_t>(far_slot) * far_pages) / far_slots;
  const uint32_t end =
      (static_cast<uint64_t>(far_slot + 1u) * far_pages) / far_slots;
  const uint32_t group = heads / kv_heads;
  const uint32_t head = kv_head * group;
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  const uint32_t head_start = head * head_dim;
  const uint32_t kv_start = kv_head * head_dim;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);

  if (threadIdx.x == 0) {
    best_score_shared = -INFINITY;
    best_page_shared = far_start + begin;
  }
  __syncthreads();

  for (uint32_t relative = begin; relative < end; ++relative) {
    const uint32_t page = far_start + relative;
    const uint32_t page_begin = page * kDecodeAttentionChunkTokens;
    uint32_t token = page_begin + kDecodeAttentionChunkTokens / 2u;
    if (token > current_position) {
      token = current_position;
    }
    const uint32_t logical_block = token / kKvCacheBlockTokens;
    const uint32_t block_offset = token - logical_block * kKvCacheBlockTokens;
    const uint64_t token_base = kv_cache_page_offset(
        layer_index, kv_block_count, kv_block_table[logical_block],
        block_offset, kv_hidden, kv_start);
    float partial = 0.0f;
#pragma unroll
    for (uint32_t item = 0; item < kHeadThreadElements; ++item) {
      const uint32_t offset = threadIdx.x + item * blockDim.x;
      if (offset < head_dim) {
        partial += s.q[head_start + offset] *
                   encoded_to_f32_typed<DType>(kv_keys[token_base + offset]);
      }
    }
    const float score = block_sum(partial);
    if (threadIdx.x == 0 && score > best_score_shared) {
      best_score_shared = score;
      best_page_shared = page;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    candidate_pages[out_index] = best_page_shared;
  }
}

template <uint32_t DType>
__global__ void hf_layer_attention_chunk_kernel(
    uint32_t layer_index, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, uint32_t attention_chunks, float *scratch,
    const uint16_t *kv_keys, const uint16_t *kv_values, float *partial_values,
    float *partial_m, float *partial_l, uint32_t kv_block_count,
    const uint32_t *kv_block_table, const uint32_t *selected_chunks,
    uint32_t qk_fused_selector, uint32_t qk_local_window_tokens,
    uint32_t qk_sink_tokens) {
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
  const uint32_t selected_slot = blockIdx.y;
  if (head >= heads || selected_slot >= attention_chunks || kv_heads == 0) {
    return;
  }
  const uint32_t group = heads / kv_heads;
  const uint32_t kv_head = head / group;
  const uint32_t chunk =
      selected_chunks == nullptr
          ? selected_slot
          : selected_chunks[static_cast<uint64_t>(kv_head) * attention_chunks +
                            selected_slot];
  (void)qk_fused_selector;
  (void)qk_local_window_tokens;
  (void)qk_sink_tokens;
  const uint32_t chunk_start = chunk * kDecodeAttentionChunkTokens;
  const uint64_t slot =
      (static_cast<uint64_t>(head) * attention_chunks + selected_slot);
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
    const uint32_t *kv_block_table, const uint32_t *selected_chunks,
    uint32_t qk_fused_selector, uint32_t qk_local_window_tokens,
    uint32_t qk_sink_tokens) {
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
  const uint32_t selected_slot = blockIdx.y;
  if (kv_head >= kv_heads || selected_slot >= attention_chunks) {
    return;
  }
  const uint32_t chunk =
      selected_chunks == nullptr
          ? selected_slot
          : selected_chunks[static_cast<uint64_t>(kv_head) * attention_chunks +
                            selected_slot];
  (void)qk_fused_selector;
  (void)qk_local_window_tokens;
  (void)qk_sink_tokens;
  const uint32_t chunk_start = chunk * kDecodeAttentionChunkTokens;
  if (chunk_start > current_position) {
    if (threadIdx.x < kGroupedGqaHeads) {
      const uint32_t head = kv_head * kGroupedGqaHeads + threadIdx.x;
      const uint64_t slot =
          (static_cast<uint64_t>(head) * attention_chunks + selected_slot);
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
      (static_cast<uint64_t>(head) * attention_chunks + selected_slot);
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
    const uint32_t *kv_block_table, const uint32_t *selected_chunks,
    uint32_t qk_fused_selector, uint32_t qk_local_window_tokens,
    uint32_t qk_sink_tokens) {
  __shared__ uint32_t current_position_shared;
  __shared__ float qk_best_score_shared;
  __shared__ uint32_t qk_best_page_shared;
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
  const uint32_t selected_slot = blockIdx.y;
  if (kv_head >= kv_heads || selected_slot >= attention_chunks) {
    return;
  }
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  const uint32_t kv_start = kv_head * head_dim;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  uint32_t chunk =
      selected_chunks == nullptr
          ? selected_slot
          : selected_chunks[static_cast<uint64_t>(kv_head) * attention_chunks +
                            selected_slot];
  if (qk_fused_selector != 0) {
    const uint32_t active_pages =
        (current_position + kDecodeAttentionChunkTokens) /
        kDecodeAttentionChunkTokens;
    const uint32_t current_page =
        current_position / kDecodeAttentionChunkTokens;
    const uint32_t sink_pages =
        (qk_sink_tokens + kDecodeAttentionChunkTokens - 1u) /
        kDecodeAttentionChunkTokens;
    const uint32_t raw_local_pages =
        (qk_local_window_tokens + kDecodeAttentionChunkTokens - 1u) /
        kDecodeAttentionChunkTokens;
    uint32_t local_start =
        current_page + 1u > raw_local_pages
            ? current_page + 1u - raw_local_pages
            : 0u;
    if (local_start < sink_pages) {
      local_start = sink_pages;
    }
    const uint32_t local_pages =
        current_page >= local_start ? current_page - local_start + 1u : 0u;
    const uint32_t local_limit = sink_pages + local_pages;
    if (selected_slot < sink_pages) {
      chunk = selected_slot < active_pages ? selected_slot : active_pages - 1u;
    } else if (selected_slot < local_limit) {
      const uint32_t page = local_start + (selected_slot - sink_pages);
      chunk = page < active_pages ? page : active_pages - 1u;
    } else {
      const uint32_t far_start =
          sink_pages < active_pages ? sink_pages : active_pages;
      const uint32_t far_end =
          local_start < active_pages ? local_start : active_pages;
      const uint32_t far_pages = far_end > far_start ? far_end - far_start : 0u;
      const uint32_t far_slots =
          attention_chunks > local_limit ? attention_chunks - local_limit : 0u;
      const uint32_t far_slot = selected_slot - local_limit;
      if (far_pages == 0 || far_slots == 0 || far_slot >= far_slots) {
        chunk = current_page < active_pages ? current_page : active_pages - 1u;
      } else {
        const uint32_t begin =
            (static_cast<uint64_t>(far_slot) * far_pages) / far_slots;
        const uint32_t end =
            (static_cast<uint64_t>(far_slot + 1u) * far_pages) / far_slots;
        const uint32_t representative_head = kv_head * kGroupedGqaHeads;
        const uint32_t representative_q_start = representative_head * head_dim;
        if (threadIdx.x == 0) {
          qk_best_score_shared = -INFINITY;
          qk_best_page_shared = far_start + begin;
        }
        __syncthreads();
        for (uint32_t relative = begin; relative < end; ++relative) {
          const uint32_t page = far_start + relative;
          const uint32_t page_begin = page * kDecodeAttentionChunkTokens;
          uint32_t token = page_begin + kDecodeAttentionChunkTokens / 2u;
          if (token > current_position) {
            token = current_position;
          }
          const uint32_t logical_block = token / kKvCacheBlockTokens;
          const uint32_t block_offset =
              token - logical_block * kKvCacheBlockTokens;
          const uint64_t token_base = kv_cache_page_offset(
              layer_index, kv_block_count, kv_block_table[logical_block],
              block_offset, kv_hidden, kv_start);
          float partial = 0.0f;
          for (uint32_t offset = threadIdx.x; offset < head_dim;
               offset += blockDim.x) {
            partial += s.q[representative_q_start + offset] *
                       encoded_to_f32_typed<DType>(kv_keys[token_base + offset]);
          }
          const float score = block_sum(partial);
          if (threadIdx.x == 0 && score > qk_best_score_shared) {
            qk_best_score_shared = score;
            qk_best_page_shared = page;
          }
          __syncthreads();
        }
        chunk = qk_best_page_shared;
      }
    }
    __syncthreads();
  } else {
    (void)qk_local_window_tokens;
    (void)qk_sink_tokens;
  }
  const uint32_t q_in_group = threadIdx.x / kSharedWarpGqaThreadsPerHead;
  const uint32_t lane = threadIdx.x - q_in_group * kSharedWarpGqaThreadsPerHead;
  const uint32_t head = kv_head * kGroupedGqaHeads + q_in_group;
  const uint64_t slot =
      (static_cast<uint64_t>(head) * attention_chunks + selected_slot);
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
  const uint32_t head_start = head * head_dim;
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
    uint16_t *attn_out, uint32_t local_window_tokens) {
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
  uint32_t token_start =
      local_window_tokens == 0 || global_pos + 1u <= local_window_tokens
          ? 0u
          : global_pos + 1u - local_window_tokens;
  for (uint32_t token = token_start; token <= global_pos;) {
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
    const uint32_t *kv_block_table, const uint32_t *selected_chunks,
    uint32_t qk_fused_selector, uint32_t qk_local_window_tokens,
    uint32_t qk_sink_tokens) {
  if (dtype == kDTypeBF16 && use_shared_warp_gqa) {
    hf_layer_shared_warp_gqa_attention_chunk_kernel<kDTypeBF16>
        <<<grid, kSharedWarpGqaThreads, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, attention_chunks, scratch, kv_keys,
            kv_values, partial_values, partial_m, partial_l, kv_block_count,
            kv_block_table, selected_chunks, qk_fused_selector,
            qk_local_window_tokens, qk_sink_tokens);
  } else if (dtype == kDTypeBF16 && use_grouped_gqa) {
    hf_layer_grouped_gqa_attention_chunk_kernel<kDTypeBF16>
        <<<grid, kGroupedGqaThreads, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, attention_chunks, scratch, kv_keys,
            kv_values, partial_values, partial_m, partial_l, kv_block_count,
            kv_block_table, selected_chunks, 0, 0, 0);
  } else if (dtype == kDTypeBF16) {
    hf_layer_attention_chunk_kernel<kDTypeBF16>
        <<<grid, dense_threads, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, attention_chunks, scratch, kv_keys,
            kv_values, partial_values, partial_m, partial_l, kv_block_count,
            kv_block_table, selected_chunks, 0, 0, 0);
  } else if (use_shared_warp_gqa) {
    hf_layer_shared_warp_gqa_attention_chunk_kernel<kDTypeF16>
        <<<grid, kSharedWarpGqaThreads, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, attention_chunks, scratch, kv_keys,
            kv_values, partial_values, partial_m, partial_l, kv_block_count,
            kv_block_table, selected_chunks, qk_fused_selector,
            qk_local_window_tokens, qk_sink_tokens);
  } else if (use_grouped_gqa) {
    hf_layer_grouped_gqa_attention_chunk_kernel<kDTypeF16>
        <<<grid, kGroupedGqaThreads, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, attention_chunks, scratch, kv_keys,
            kv_values, partial_values, partial_m, partial_l, kv_block_count,
            kv_block_table, selected_chunks, 0, 0, 0);
  } else {
    hf_layer_attention_chunk_kernel<kDTypeF16>
        <<<grid, dense_threads, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, attention_chunks, scratch, kv_keys,
            kv_values, partial_values, partial_m, partial_l, kv_block_count,
            kv_block_table, selected_chunks, 0, 0, 0);
  }
}

void launch_hf_experimental_qk_page_selector_kernel(
    cudaStream_t stream, dim3 grid, uint32_t dtype, uint32_t layer_index,
    uint32_t hidden, uint32_t heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    uint32_t selected_pages, uint32_t local_window_tokens,
    uint32_t sink_tokens, float *scratch, const uint16_t *kv_keys,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint32_t *candidate_pages) {
  if (dtype == kDTypeBF16) {
    hf_experimental_qk_page_selector_kernel<kDTypeBF16>
        <<<grid, kHeadThreadsMax, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, selected_pages, local_window_tokens,
            sink_tokens, scratch, kv_keys, kv_block_count, kv_block_table,
            candidate_pages);
  } else {
    hf_experimental_qk_page_selector_kernel<kDTypeF16>
        <<<grid, kHeadThreadsMax, 0, stream>>>(
            layer_index, hidden, heads, kv_heads, head_dim, intermediate,
            step_cursor, max_steps, selected_pages, local_window_tokens,
            sink_tokens, scratch, kv_keys, kv_block_count, kv_block_table,
            candidate_pages);
  }
}

void launch_hf_prefill_grouped_gqa_attention_direct_kernel(
    cudaStream_t stream, dim3 grid, uint32_t dtype, uint32_t layer_index,
    uint32_t heads, uint32_t kv_heads, uint32_t head_dim, uint32_t max_steps,
    uint32_t chunk_start, uint32_t chunk_tokens, const float *qkv,
    const uint16_t *kv_keys, const uint16_t *kv_values,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint16_t *attn_out, uint32_t local_window_tokens) {
  if (dtype == kDTypeBF16) {
    hf_prefill_grouped_gqa_attention_direct_kernel<kDTypeBF16>
        <<<grid, kSharedWarpGqaThreads, 0, stream>>>(
            layer_index, heads, kv_heads, head_dim, max_steps, chunk_start,
            chunk_tokens, qkv, kv_keys, kv_values, kv_block_count,
            kv_block_table, attn_out, local_window_tokens);
  } else {
    hf_prefill_grouped_gqa_attention_direct_kernel<kDTypeF16>
        <<<grid, kSharedWarpGqaThreads, 0, stream>>>(
            layer_index, heads, kv_heads, head_dim, max_steps, chunk_start,
            chunk_tokens, qkv, kv_keys, kv_values, kv_block_count,
            kv_block_table, attn_out, local_window_tokens);
  }
}
