#include "nerva_cuda_api.h"
#include "deepseek_quant.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kDeepSeekScaleFormatE8M0 = 0;
constexpr uint32_t kDeepSeekScaleFormatF32 = 1;
constexpr uint32_t kDeepSeekScaleFormatMxfp4 = 2;
constexpr uint32_t kDeepSeekMaxCompressHeadSize = 512;

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t lsb = (bits >> 16u) & 1u;
  const uint32_t rounded = bits + 0x7fffu + lsb;
  return static_cast<uint16_t>(rounded >> 16u);
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16u);
}

__device__ uint8_t f32_to_f8_e4m3fn_bits_nearest(float value) {
  return nerva::deepseek::f32_to_f8_e4m3fn_bits(value);
}

__device__ __forceinline__ uint8_t encode_e8m0_scale(float scale) {
  int exponent = static_cast<int>(ceilf(log2f(scale)));
  exponent += 127;
  if (exponent < 0) {
    exponent = 0;
  }
  if (exponent > 255) {
    exponent = 255;
  }
  return static_cast<uint8_t>(exponent);
}

__device__ uint8_t f32_to_mxfp4_e2m1_nibble_nearest(float value) {
  uint8_t best_bits = 0;
  float best_error = INFINITY;
  for (uint32_t bits = 0; bits < 16u; ++bits) {
    const float candidate =
        nerva::deepseek::mxfp4_e2m1_nibble_to_f32(static_cast<uint8_t>(bits));
    const float error = fabsf(candidate - value);
    if (error < best_error ||
        (error == best_error && bits < static_cast<uint32_t>(best_bits))) {
      best_error = error;
      best_bits = static_cast<uint8_t>(bits);
    }
  }
  return best_bits;
}

__global__ void fp8_ds_mla_pack_kernel(const uint8_t *nope_fp8,
                                       const uint16_t *rope_bf16,
                                       const uint8_t *scales,
                                       uint8_t *output_block,
                                       uint32_t block_size,
                                       uint32_t token_index,
                                       uint32_t nope_bytes,
                                       uint32_t rope_bf16_values,
                                       uint32_t scale_dim,
                                       uint32_t token_stride) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t rope_bytes = rope_bf16_values * 2;
  const uint32_t data_bytes = nope_bytes + rope_bytes;
  const uint32_t total_bytes = data_bytes + scale_dim;
  if (idx >= total_bytes) {
    return;
  }

  const uint64_t token_base =
      static_cast<uint64_t>(token_index) * token_stride;
  const uint64_t scale_base =
      static_cast<uint64_t>(block_size) * token_stride +
      static_cast<uint64_t>(token_index) * scale_dim;

  if (idx < nope_bytes) {
    output_block[token_base + idx] = nope_fp8[idx];
    return;
  }
  if (idx < data_bytes) {
    const uint32_t rope_byte = idx - nope_bytes;
    const uint16_t value = rope_bf16[rope_byte / 2];
    output_block[token_base + idx] =
        static_cast<uint8_t>((value >> ((rope_byte & 1u) * 8u)) & 0xffu);
    return;
  }

  const uint32_t scale_idx = idx - data_bytes;
  output_block[scale_base + scale_idx] = scales[scale_idx];
}

__global__ void v32_fp8_ds_mla_pack_kernel(const uint8_t *nope_fp8,
                                           const uint16_t *rope_bf16,
                                           const uint8_t *scales,
                                           uint8_t *output_block,
                                           uint32_t token_index,
                                           uint32_t nope_bytes,
                                           uint32_t rope_bf16_values,
                                           uint32_t scale_dim,
                                           uint32_t token_stride) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t rope_bytes = rope_bf16_values * 2;
  if (idx >= token_stride) {
    return;
  }

  const uint64_t token_base =
      static_cast<uint64_t>(token_index) * token_stride;
  if (idx < nope_bytes) {
    output_block[token_base + idx] = nope_fp8[idx];
    return;
  }
  if (idx < nope_bytes + scale_dim) {
    output_block[token_base + idx] = scales[idx - nope_bytes];
    return;
  }

  const uint32_t rope_byte = idx - nope_bytes - scale_dim;
  if (rope_byte < rope_bytes) {
    const uint16_t value = rope_bf16[rope_byte / 2];
    output_block[token_base + idx] =
        static_cast<uint8_t>((value >> ((rope_byte & 1u) * 8u)) & 0xffu);
  }
}

__global__ void compressed_slot_mapping_kernel(
    int64_t *output_slots,
    const int32_t *query_start_loc,
    const int32_t *seq_lens,
    const int32_t *block_table,
    uint32_t num_tokens,
    uint32_t num_reqs,
    uint32_t block_table_stride,
    uint32_t block_size,
    uint32_t compress_ratio) {
  const uint32_t req_idx = blockIdx.x;
  if (req_idx >= num_reqs) {
    return;
  }

  const int32_t query_start = query_start_loc[req_idx];
  const int32_t query_end = query_start_loc[req_idx + 1];
  if (query_start < 0 || query_end < query_start) {
    return;
  }

  const uint32_t query_len = static_cast<uint32_t>(query_end - query_start);
  const int32_t start_pos = seq_lens[req_idx] - static_cast<int32_t>(query_len);
  for (uint32_t offset = threadIdx.x; offset < query_len; offset += blockDim.x) {
    const uint32_t output_idx = static_cast<uint32_t>(query_start) + offset;
    if (output_idx >= num_tokens) {
      continue;
    }

    int64_t slot = -1;
    const int32_t pos = start_pos + static_cast<int32_t>(offset);
    if (pos >= 0 && ((pos + 1) % static_cast<int32_t>(compress_ratio)) == 0) {
      const int32_t compressed_pos =
          pos / static_cast<int32_t>(compress_ratio);
      const int32_t block_id =
          compressed_pos / static_cast<int32_t>(block_size);
      const int32_t block_offset =
          compressed_pos % static_cast<int32_t>(block_size);
      if (block_id >= 0 &&
          block_id < static_cast<int32_t>(block_table_stride)) {
        const int32_t block_number =
            block_table[req_idx * block_table_stride + block_id];
        if (block_number >= 0) {
          slot = static_cast<int64_t>(block_number) * block_size + block_offset;
        }
      }
    }
    output_slots[output_idx] = slot;
  }
}

__global__ void c128_topk_metadata_kernel(
    int32_t *global_decode,
    int32_t *decode_lens,
    int32_t *prefill_local,
    const int64_t *positions,
    const int32_t *token_to_req_indices,
    const int32_t *block_table,
    const int64_t *slot_mapping,
    uint32_t num_tokens,
    uint32_t num_decode_tokens,
    uint32_t num_reqs,
    uint32_t block_table_stride,
    uint32_t block_size,
    uint32_t compress_ratio,
    uint32_t max_compressed_tokens) {
  const uint32_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  const int64_t position = positions[token_idx];
  uint32_t num_compressed = 0;
  if (position >= 0) {
    const uint64_t raw =
        (static_cast<uint64_t>(position) + 1ull) / compress_ratio;
    num_compressed = raw > max_compressed_tokens
                         ? max_compressed_tokens
                         : static_cast<uint32_t>(raw);
  }

  if (token_idx < num_decode_tokens) {
    const bool valid_token = slot_mapping[token_idx] >= 0;
    int32_t local_count = 0;
    const int32_t req_idx = token_to_req_indices[token_idx];
    for (uint32_t offset = threadIdx.x; offset < max_compressed_tokens;
         offset += blockDim.x) {
      int32_t slot = -1;
      const bool is_valid = offset < num_compressed;
      if (is_valid && req_idx >= 0 &&
          req_idx < static_cast<int32_t>(num_reqs)) {
        const uint32_t block_id = offset / block_size;
        const uint32_t block_offset = offset % block_size;
        if (block_id < block_table_stride) {
          const int32_t block_number =
              block_table[static_cast<uint32_t>(req_idx) * block_table_stride +
                          block_id];
          if (block_number >= 0) {
            slot = block_number * static_cast<int32_t>(block_size) +
                   static_cast<int32_t>(block_offset);
          }
        }
      }
      global_decode[token_idx * max_compressed_tokens + offset] = slot;
      local_count += is_valid ? 1 : 0;
    }

    __shared__ int32_t counts[256];
    counts[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        counts[threadIdx.x] += counts[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      decode_lens[token_idx] = valid_token ? counts[0] : 0;
    }
  } else {
    const uint32_t prefill_idx = token_idx - num_decode_tokens;
    for (uint32_t offset = threadIdx.x; offset < max_compressed_tokens;
         offset += blockDim.x) {
      prefill_local[prefill_idx * max_compressed_tokens + offset] =
          offset < num_compressed ? static_cast<int32_t>(offset) : -1;
    }
  }
}

__device__ __forceinline__ bool c4_indexer_score_is_better(float candidate,
                                                           int32_t slot,
                                                           float current,
                                                           int32_t current_slot) {
  if (!isfinite(candidate)) {
    return false;
  }
  if (current_slot < 0) {
    return true;
  }
  return candidate > current ||
         (candidate == current && slot >= 0 && slot < current_slot);
}

__global__ void c4_indexer_score_kernel(float *logits,
                                        const float *query,
                                        const float *key_cache,
                                        const float *weights,
                                        const int32_t *context_lens,
                                        uint32_t num_tokens,
                                        uint32_t num_heads,
                                        uint32_t head_dim,
                                        uint32_t max_compressed_tokens) {
  const uint32_t token_idx = blockIdx.x;
  const uint32_t slot = blockIdx.y * blockDim.x + threadIdx.x;
  if (token_idx >= num_tokens || slot >= max_compressed_tokens) {
    return;
  }

  const uint64_t logits_idx =
      static_cast<uint64_t>(token_idx) * max_compressed_tokens + slot;
  const int32_t raw_context_len = context_lens[token_idx];
  if (raw_context_len <= 0 ||
      slot >= static_cast<uint32_t>(raw_context_len)) {
    logits[logits_idx] = -INFINITY;
    return;
  }

  float score = 0.0f;
  for (uint32_t head = 0; head < num_heads; ++head) {
    const float head_weight = weights[token_idx * num_heads + head];
    float dot = 0.0f;
    const uint64_t query_base =
        (static_cast<uint64_t>(token_idx) * num_heads + head) * head_dim;
    const uint64_t key_base = static_cast<uint64_t>(slot) * head_dim;
    for (uint32_t dim = 0; dim < head_dim; ++dim) {
      dot += query[query_base + dim] * key_cache[key_base + dim];
    }
    score += head_weight * dot;
  }
  logits[logits_idx] = score;
}

__global__ void c4_indexer_topk_from_scores_kernel(
    int32_t *topk_indices,
    float *topk_scores,
    const float *logits,
    const int32_t *context_lens,
    uint32_t num_tokens,
    uint32_t max_compressed_tokens,
    uint32_t topk_tokens) {
  const uint32_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens || threadIdx.x != 0) {
    return;
  }
  const uint64_t output_base =
      static_cast<uint64_t>(token_idx) * topk_tokens;
  for (uint32_t rank = 0; rank < topk_tokens; ++rank) {
    topk_indices[output_base + rank] = -1;
    topk_scores[output_base + rank] = -INFINITY;
  }

  const int32_t raw_context_len = context_lens[token_idx];
  if (raw_context_len <= 0) {
    return;
  }
  const uint32_t context_len =
      static_cast<uint32_t>(raw_context_len) > max_compressed_tokens
          ? max_compressed_tokens
          : static_cast<uint32_t>(raw_context_len);

  for (uint32_t slot = 0; slot < context_len; ++slot) {
    const float score =
        logits[static_cast<uint64_t>(token_idx) * max_compressed_tokens + slot];
    const int32_t slot_i32 = static_cast<int32_t>(slot);
    for (uint32_t rank = 0; rank < topk_tokens; ++rank) {
      const uint64_t output_idx = output_base + rank;
      if (!c4_indexer_score_is_better(score,
                                      slot_i32,
                                      topk_scores[output_idx],
                                      topk_indices[output_idx])) {
        continue;
      }
      for (uint32_t shift = topk_tokens - 1; shift > rank; --shift) {
        topk_indices[output_base + shift] =
            topk_indices[output_base + shift - 1];
        topk_scores[output_base + shift] =
            topk_scores[output_base + shift - 1];
      }
      topk_indices[output_idx] = slot_i32;
      topk_scores[output_idx] = score;
      break;
    }
  }
}

__global__ void save_partial_states_kernel(float *state_cache,
                                           const float *kv,
                                           const float *score,
                                           const float *ape,
                                           const int64_t *positions,
                                           const int64_t *slot_mapping,
                                           uint32_t num_tokens,
                                           uint32_t block_size,
                                           uint32_t head_size,
                                           uint32_t state_width,
                                           uint32_t compress_ratio,
                                           uint32_t num_blocks,
                                           uint64_t state_cache_stride0,
                                           uint64_t state_cache_stride1,
                                           uint32_t *written_flags) {
  const uint32_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }
  const int64_t slot_id = slot_mapping[token_idx];
  if (slot_id < 0) {
    if (threadIdx.x == 0) {
      written_flags[token_idx] = 0;
    }
    return;
  }
  const uint64_t max_slots =
      static_cast<uint64_t>(num_blocks) * static_cast<uint64_t>(block_size);
  if (static_cast<uint64_t>(slot_id) >= max_slots) {
    if (threadIdx.x == 0) {
      written_flags[token_idx] = 0;
    }
    return;
  }
  if (threadIdx.x == 0) {
    written_flags[token_idx] = 1;
  }

  const uint64_t block_idx = static_cast<uint64_t>(slot_id) / block_size;
  const uint64_t pos_in_block = static_cast<uint64_t>(slot_id) % block_size;
  const uint64_t base =
      block_idx * state_cache_stride0 + pos_in_block * state_cache_stride1;
  uint32_t ape_row = 0;
  if (compress_ratio > 0) {
    const int64_t position = positions[token_idx];
    ape_row = position >= 0
                  ? static_cast<uint32_t>(
                        static_cast<uint64_t>(position) % compress_ratio)
                  : 0;
  }

  for (uint32_t dim = threadIdx.x; dim < head_size; dim += blockDim.x) {
    state_cache[base + dim] = kv[token_idx * head_size + dim];
    state_cache[base + state_width + dim] =
        score[token_idx * head_size + dim] + ape[ape_row * head_size + dim];
  }
}

__global__ void compress_norm_rope_fp8_cache_kernel(
    const float *state_cache,
    const int32_t *token_to_req_indices,
    const int64_t *positions,
    const int64_t *slot_mapping,
    const int32_t *block_table,
    const int64_t *kv_slot_mapping,
    const float *rms_norm_weight,
    const float *cos_sin_cache,
    uint8_t *kv_cache,
    uint32_t *written_flags,
    uint32_t num_tokens,
    uint32_t num_reqs,
    uint32_t block_table_stride,
    uint32_t state_block_size,
    uint32_t kv_cache_block_size,
    uint32_t head_size,
    uint32_t state_width,
    uint32_t rope_head_dim,
    uint32_t compress_ratio,
    uint32_t overlap,
    uint32_t quant_block,
    uint32_t token_stride,
    uint32_t scale_dim,
    uint32_t scale_format,
    uint32_t num_state_blocks,
    uint32_t num_kv_blocks,
    uint32_t kv_cache_block_stride,
    uint32_t cos_sin_stride,
    float rms_norm_eps,
    float fp8_max) {
  __shared__ float compressed[kDeepSeekMaxCompressHeadSize];
  __shared__ float normed[kDeepSeekMaxCompressHeadSize];
  __shared__ float rotated[kDeepSeekMaxCompressHeadSize];
  __shared__ float reduce[kDeepSeekMaxCompressHeadSize];
  __shared__ float scales[16];

  const uint32_t token_idx = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  if (tid == 0) {
    written_flags[token_idx] = 0;
  }

  const int64_t slot_id = slot_mapping[token_idx];
  const int64_t position = positions[token_idx];
  const int32_t req_idx = token_to_req_indices[token_idx];
  const int64_t kv_slot_idx = kv_slot_mapping[token_idx];
  const bool boundary =
      position >= 0 &&
      ((position + 1) % static_cast<int64_t>(compress_ratio)) == 0;
  const bool valid =
      slot_id >= 0 && boundary && req_idx >= 0 &&
      req_idx < static_cast<int32_t>(num_reqs) && kv_slot_idx >= 0 &&
      static_cast<uint64_t>(kv_slot_idx) <
          static_cast<uint64_t>(num_kv_blocks) * kv_cache_block_size;
  if (!valid) {
    return;
  }

  const uint32_t window_tokens = (1u + overlap) * compress_ratio;
  const int64_t start =
      position - static_cast<int64_t>(window_tokens) + 1ll;
  const uint64_t row_stride =
      static_cast<uint64_t>(state_width) * 2ull;
  const uint64_t state_block_stride =
      static_cast<uint64_t>(state_block_size) * row_stride;

  if (tid < head_size) {
    float max_score = -INFINITY;
    for (uint32_t window = 0; window < window_tokens; ++window) {
      const int64_t pos = start + static_cast<int64_t>(window);
      if (pos < 0) {
        continue;
      }
      const uint32_t block_index =
          static_cast<uint32_t>(pos) / state_block_size;
      if (block_index >= block_table_stride) {
        continue;
      }
      const int32_t block_number =
          block_table[static_cast<uint32_t>(req_idx) * block_table_stride +
                      block_index];
      if (block_number < 0 ||
          block_number >= static_cast<int32_t>(num_state_blocks)) {
        continue;
      }
      const uint32_t block_offset =
          static_cast<uint32_t>(pos) % state_block_size;
      const uint32_t head_offset = window >= compress_ratio ? head_size : 0u;
      const uint64_t base =
          static_cast<uint64_t>(block_number) * state_block_stride +
          static_cast<uint64_t>(block_offset) * row_stride + head_offset;
      const float score = state_cache[base + state_width + tid];
      max_score = fmaxf(max_score, score);
    }

    float weighted = 0.0f;
    float denom = 0.0f;
    for (uint32_t window = 0; window < window_tokens; ++window) {
      const int64_t pos = start + static_cast<int64_t>(window);
      if (pos < 0) {
        continue;
      }
      const uint32_t block_index =
          static_cast<uint32_t>(pos) / state_block_size;
      if (block_index >= block_table_stride) {
        continue;
      }
      const int32_t block_number =
          block_table[static_cast<uint32_t>(req_idx) * block_table_stride +
                      block_index];
      if (block_number < 0 ||
          block_number >= static_cast<int32_t>(num_state_blocks)) {
        continue;
      }
      const uint32_t block_offset =
          static_cast<uint32_t>(pos) % state_block_size;
      const uint32_t head_offset = window >= compress_ratio ? head_size : 0u;
      const uint64_t base =
          static_cast<uint64_t>(block_number) * state_block_stride +
          static_cast<uint64_t>(block_offset) * row_stride + head_offset;
      const float score = state_cache[base + state_width + tid];
      const float weight = expf(score - max_score);
      weighted += state_cache[base + tid] * weight;
      denom += weight;
    }
    compressed[tid] = denom > 0.0f ? weighted / denom : 0.0f;
    reduce[tid] = compressed[tid] * compressed[tid];
  } else if (tid < kDeepSeekMaxCompressHeadSize) {
    compressed[tid] = 0.0f;
    reduce[tid] = 0.0f;
  }
  __syncthreads();

  for (uint32_t stride = kDeepSeekMaxCompressHeadSize / 2u; stride > 0;
       stride >>= 1u) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float rrms = rsqrtf(reduce[0] / static_cast<float>(head_size) +
                           rms_norm_eps);
  if (tid < head_size) {
    normed[tid] = compressed[tid] * rrms * rms_norm_weight[tid];
  }
  __syncthreads();

  const uint32_t nope_head_dim = head_size - rope_head_dim;
  const uint32_t half_rope = rope_head_dim / 2u;
  const uint32_t compressed_pos =
      (static_cast<uint32_t>(position) / compress_ratio) * compress_ratio;
  if (tid < head_size) {
    float value = normed[tid];
    if (tid >= nope_head_dim) {
      const uint32_t rope_local = tid - nope_head_dim;
      const uint32_t pair_local = rope_local / 2u;
      const uint32_t pair_base = nope_head_dim + pair_local * 2u;
      const float even = normed[pair_base];
      const float odd = normed[pair_base + 1u];
      const uint64_t cs_base =
          static_cast<uint64_t>(compressed_pos) * cos_sin_stride;
      const float cos_v = cos_sin_cache[cs_base + pair_local];
      const float sin_v = cos_sin_cache[cs_base + half_rope + pair_local];
      value = (rope_local & 1u) == 0u ? even * cos_v - odd * sin_v
                                      : odd * cos_v + even * sin_v;
    }
    rotated[tid] = value;
  }
  __syncthreads();

  const uint64_t kv_block_idx =
      static_cast<uint64_t>(kv_slot_idx) / kv_cache_block_size;
  const uint64_t kv_pos_in_block =
      static_cast<uint64_t>(kv_slot_idx) % kv_cache_block_size;
  uint8_t *cache_block =
      kv_cache + kv_block_idx * static_cast<uint64_t>(kv_cache_block_stride);
  uint8_t *data_ptr = cache_block + kv_pos_in_block * token_stride;
  uint8_t *scale_ptr =
      cache_block + static_cast<uint64_t>(kv_cache_block_size) * token_stride +
      kv_pos_in_block * scale_dim;

  if (scale_format == kDeepSeekScaleFormatE8M0) {
    const uint32_t blocks = head_size / quant_block;
    for (uint32_t block = tid; block < blocks; block += blockDim.x) {
      float absmax = 0.0f;
      for (uint32_t offset = 0; offset < quant_block; ++offset) {
        const uint32_t dim = block * quant_block + offset;
        const float quant_input = bf16_bits_to_f32(f32_to_bf16_bits(normed[dim]));
        absmax = fmaxf(absmax, fabsf(quant_input));
      }
      const float raw = fmaxf(absmax, 1.0e-4f) / fp8_max;
      scales[block] = exp2f(ceilf(log2f(raw)));
    }
    __syncthreads();

    if (tid < nope_head_dim) {
      const uint32_t block = tid / quant_block;
      const float scale = scales[block];
      const float quant_input = bf16_bits_to_f32(f32_to_bf16_bits(normed[tid]));
      const float scaled = fminf(fmaxf(quant_input / scale, -fp8_max), fp8_max);
      data_ptr[tid] = f32_to_f8_e4m3fn_bits_nearest(scaled);
    }
    if (tid >= nope_head_dim && tid < head_size) {
      const uint32_t rope_local = tid - nope_head_dim;
      const uint16_t bits = f32_to_bf16_bits(rotated[tid]);
      uint8_t *rope_ptr = data_ptr + nope_head_dim + rope_local * 2u;
      rope_ptr[0] = static_cast<uint8_t>(bits & 0xffu);
      rope_ptr[1] = static_cast<uint8_t>(bits >> 8u);
    }
    if (tid < scale_dim) {
      const uint32_t nope_blocks = nope_head_dim / quant_block;
      scale_ptr[tid] = tid < nope_blocks ? encode_e8m0_scale(scales[tid]) : 0u;
    }
  } else if (scale_format == kDeepSeekScaleFormatF32) {
    float absmax = 0.0f;
    if (tid < head_size) {
      reduce[tid] = fabsf(bf16_bits_to_f32(f32_to_bf16_bits(rotated[tid])));
    } else if (tid < kDeepSeekMaxCompressHeadSize) {
      reduce[tid] = 0.0f;
    }
    __syncthreads();
    for (uint32_t stride = kDeepSeekMaxCompressHeadSize / 2u; stride > 0;
         stride >>= 1u) {
      if (tid < stride) {
        reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
      }
      __syncthreads();
    }
    absmax = reduce[0];
    const float raw = fmaxf(absmax, 1.0e-4f) / fp8_max;
    const float scale = exp2f(ceilf(log2f(raw)));
    if (tid == 0) {
      *reinterpret_cast<float *>(scale_ptr) = scale;
    }
    if (tid < head_size) {
      const float quant_input = bf16_bits_to_f32(f32_to_bf16_bits(rotated[tid]));
      const float scaled = fminf(fmaxf(quant_input / scale, -fp8_max), fp8_max);
      data_ptr[tid] = f32_to_f8_e4m3fn_bits_nearest(scaled);
    }
  } else if (scale_format == kDeepSeekScaleFormatMxfp4) {
    const uint32_t num_blocks = head_size / quant_block;
    const uint32_t half_block = quant_block / 2u;
    for (uint32_t block = tid; block < num_blocks; block += blockDim.x) {
      float absmax = 0.0f;
      const uint32_t pair_start = block * half_block;
      for (uint32_t pair = 0; pair < half_block; ++pair) {
        const uint32_t base = (pair_start + pair) * 2u;
        const float even = bf16_bits_to_f32(f32_to_bf16_bits(rotated[base]));
        const float odd = bf16_bits_to_f32(f32_to_bf16_bits(rotated[base + 1u]));
        absmax = fmaxf(absmax, fabsf(even));
        absmax = fmaxf(absmax, fabsf(odd));
      }
      const float raw = fmaxf(absmax, 6.0f * 1.1754943508222875e-38f) / 6.0f;
      const float exponent = fminf(fmaxf(ceilf(log2f(raw)), -127.0f), 127.0f);
      const float inv_scale = exp2f(-exponent);
      scale_ptr[block] = static_cast<uint8_t>(static_cast<int>(exponent) + 127);
      for (uint32_t pair = 0; pair < half_block; ++pair) {
        const uint32_t base = (pair_start + pair) * 2u;
        const float even = bf16_bits_to_f32(f32_to_bf16_bits(rotated[base]));
        const float odd = bf16_bits_to_f32(f32_to_bf16_bits(rotated[base + 1u]));
        const uint8_t lo = f32_to_mxfp4_e2m1_nibble_nearest(even * inv_scale);
        const uint8_t hi = f32_to_mxfp4_e2m1_nibble_nearest(odd * inv_scale);
        data_ptr[block * half_block + pair] =
            static_cast<uint8_t>((hi << 4u) | (lo & 0x0fu));
      }
    }
  }

  if (tid == 0) {
    written_flags[token_idx] = 1;
  }
}

uint64_t hash_bytes(const uint8_t *values, uint64_t len) {
  uint64_t hash = 1469598103934665603ull;
  for (uint64_t i = 0; i < len; ++i) {
    hash ^= values[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

void clear_result(NervaCudaDeepSeekKvFp8DsMlaPackResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

void clear_result(NervaCudaDeepSeekCompressedSlotMappingResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

void clear_result(NervaCudaDeepSeekC128TopkMetadataResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

void clear_result(NervaCudaDeepSeekC4IndexerTopkResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

void clear_result(NervaCudaDeepSeekSavePartialStatesResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

void clear_result(NervaCudaDeepSeekCompressNormRopeFp8CacheResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

int fail(NervaCudaDeepSeekKvFp8DsMlaPackResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

int fail(NervaCudaDeepSeekCompressedSlotMappingResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

int fail(NervaCudaDeepSeekC128TopkMetadataResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

int fail(NervaCudaDeepSeekC4IndexerTopkResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

int fail(NervaCudaDeepSeekSavePartialStatesResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

int fail(NervaCudaDeepSeekCompressNormRopeFp8CacheResult *out,
         cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

bool validate_request(const NervaCudaDeepSeekKvFp8DsMlaPackRequest *request) {
  return request != nullptr && request->nope_fp8 != nullptr &&
         request->rope_bf16 != nullptr && request->scales != nullptr &&
         request->output_block != nullptr && request->block_size > 0 &&
         request->token_index < request->block_size && request->nope_bytes > 0 &&
         request->rope_bf16_values > 0 && request->scale_dim > 0;
}

bool validate_request(
    const NervaCudaDeepSeekCompressedSlotMappingRequest *request) {
  return request != nullptr && request->query_start_loc != nullptr &&
         request->seq_lens != nullptr && request->block_table != nullptr &&
         request->output_slots != nullptr && request->num_tokens > 0 &&
         request->num_reqs > 0 && request->block_table_stride > 0 &&
         request->block_size > 0 && request->compress_ratio > 0;
}

bool validate_request(const NervaCudaDeepSeekC128TopkMetadataRequest *request) {
  return request != nullptr && request->positions != nullptr &&
         request->token_to_req_indices != nullptr &&
         request->block_table != nullptr && request->slot_mapping != nullptr &&
         request->global_decode != nullptr && request->decode_lens != nullptr &&
         request->prefill_local != nullptr && request->num_tokens > 0 &&
         request->num_decode_tokens <= request->num_tokens &&
         request->num_reqs > 0 && request->block_table_stride > 0 &&
         request->block_size > 0 && request->compress_ratio > 0 &&
         request->max_compressed_tokens > 0;
}

bool validate_request(const NervaCudaDeepSeekC4IndexerTopkRequest *request) {
  return request != nullptr && request->query != nullptr &&
         request->key_cache != nullptr && request->weights != nullptr &&
         request->context_lens != nullptr && request->topk_indices != nullptr &&
         request->topk_scores != nullptr && request->num_tokens > 0 &&
         request->num_heads > 0 && request->head_dim > 0 &&
         request->max_compressed_tokens > 0 && request->topk_tokens > 0;
}

bool validate_request(const NervaCudaDeepSeekSavePartialStatesRequest *request) {
  return request != nullptr && request->kv != nullptr &&
         request->score != nullptr && request->ape != nullptr &&
         request->positions != nullptr && request->slot_mapping != nullptr &&
         request->state_cache != nullptr && request->num_tokens > 0 &&
         request->block_size > 0 && request->head_size > 0 &&
         request->state_width >= request->head_size &&
         request->compress_ratio > 0 && request->num_blocks > 0;
}

bool validate_request(
    const NervaCudaDeepSeekCompressNormRopeFp8CacheRequest *request) {
  if (request == nullptr || request->state_cache == nullptr ||
      request->token_to_req_indices == nullptr || request->positions == nullptr ||
      request->slot_mapping == nullptr || request->block_table == nullptr ||
      request->kv_slot_mapping == nullptr || request->rms_norm_weight == nullptr ||
      request->cos_sin_cache == nullptr || request->kv_cache == nullptr) {
    return false;
  }
  if (request->num_tokens == 0 || request->num_reqs == 0 ||
      request->block_table_stride == 0 || request->state_block_size == 0 ||
      request->kv_cache_block_size == 0 || request->head_size == 0 ||
      request->head_size > kDeepSeekMaxCompressHeadSize ||
      request->state_width < request->head_size * (1u + request->overlap) ||
      request->rope_head_dim > request->head_size ||
      (request->rope_head_dim & 1u) != 0 || request->compress_ratio == 0 ||
      request->quant_block == 0 ||
      (request->head_size % request->quant_block) != 0 ||
      request->head_size / request->quant_block > 16u ||
      request->num_state_blocks == 0 || request->num_kv_blocks == 0 ||
      request->kv_cache_block_stride == 0 || request->cos_sin_stride == 0 ||
      request->cos_sin_values == 0 || !isfinite(request->rms_norm_eps) ||
      !isfinite(request->fp8_max) || request->rms_norm_eps <= 0.0f ||
      request->fp8_max <= 0.0f) {
    return false;
  }
  const uint32_t nope_head_dim = request->head_size - request->rope_head_dim;
  if (request->scale_format == kDeepSeekScaleFormatE8M0) {
    const uint32_t nope_blocks = nope_head_dim / request->quant_block;
    const uint32_t expected_stride = nope_head_dim + request->rope_head_dim * 2u;
    return request->rope_head_dim > 0 && request->token_stride == expected_stride &&
           request->scale_dim >= nope_blocks + 1u &&
           request->kv_cache_block_stride >=
               request->kv_cache_block_size * request->token_stride +
                   request->kv_cache_block_size * request->scale_dim &&
           request->cos_sin_stride >= request->rope_head_dim;
  }
  if (request->scale_format == kDeepSeekScaleFormatF32) {
    return request->token_stride == request->head_size &&
           request->scale_dim == sizeof(float) &&
           request->kv_cache_block_stride >=
               request->kv_cache_block_size * request->token_stride +
                   request->kv_cache_block_size * request->scale_dim &&
           request->cos_sin_stride >= request->rope_head_dim;
  }
  if (request->scale_format == kDeepSeekScaleFormatMxfp4) {
    return request->head_size == 128u && request->quant_block == 32u &&
           request->rope_head_dim > 0 && (request->quant_block & 1u) == 0 &&
           request->token_stride == request->head_size / 2u &&
           request->scale_dim == request->head_size / request->quant_block &&
           request->kv_cache_block_stride >=
               request->kv_cache_block_size * request->token_stride +
                   request->kv_cache_block_size * request->scale_dim &&
           request->cos_sin_stride >= request->rope_head_dim;
  }
  return false;
}

}  // namespace

extern "C" int nerva_cuda_deepseek_kv_fp8_ds_mla_pack(
    const NervaCudaDeepSeekKvFp8DsMlaPackRequest *request,
    NervaCudaDeepSeekKvFp8DsMlaPackResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  if (!validate_request(request)) {
    return -1;
  }

  const uint64_t rope_bytes = static_cast<uint64_t>(request->rope_bf16_values) * 2;
  const uint64_t token_stride =
      static_cast<uint64_t>(request->nope_bytes) + rope_bytes;
  const uint64_t block_bytes =
      static_cast<uint64_t>(request->block_size) *
      (token_stride + request->scale_dim);
  if (token_stride > UINT32_MAX || block_bytes == 0) {
    return -1;
  }

  out->block_size = request->block_size;
  out->token_index = request->token_index;
  out->token_stride = static_cast<uint32_t>(token_stride);
  out->scale_dim = request->scale_dim;
  out->block_bytes = block_bytes;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    return fail(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  uint8_t *d_nope = nullptr;
  uint16_t *d_rope = nullptr;
  uint8_t *d_scales = nullptr;
  uint8_t *d_output = nullptr;
  uint8_t *h_output = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t nope_bytes = request->nope_bytes;
  const uint64_t rope_input_bytes =
      static_cast<uint64_t>(request->rope_bf16_values) * sizeof(uint16_t);
  const uint64_t scale_bytes = request->scale_dim;

  err = cudaMalloc(reinterpret_cast<void **>(&d_nope), nope_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_rope), rope_input_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_scales), scale_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output), block_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      nope_bytes + rope_input_bytes + scale_bytes + block_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output),
                      block_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = block_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(
      d_nope, request->nope_fp8, nope_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_rope,
                        request->rope_bf16,
                        rope_input_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(
      d_scales, request->scales, scale_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = nope_bytes + rope_input_bytes + scale_bytes;

  err = cudaMemsetAsync(d_output, 0, block_bytes, stream);
  if (err != cudaSuccess) goto cleanup;

  {
    constexpr uint32_t threads = 256;
    const uint32_t copy_bytes =
        request->nope_bytes + request->rope_bf16_values * 2 +
        request->scale_dim;
    const uint32_t blocks = (copy_bytes + threads - 1) / threads;
    fp8_ds_mla_pack_kernel<<<blocks, threads, 0, stream>>>(
        d_nope,
        d_rope,
        d_scales,
        d_output,
        request->block_size,
        request->token_index,
        request->nope_bytes,
        request->rope_bf16_values,
        request->scale_dim,
        out->token_stride);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(
      h_output, d_output, block_bytes, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = block_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->output_block, h_output, block_bytes);
  out->output_hash = hash_bytes(request->output_block, block_bytes);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output != nullptr) cudaFreeHost(h_output);
  if (d_output != nullptr) cudaFree(d_output);
  if (d_scales != nullptr) cudaFree(d_scales);
  if (d_rope != nullptr) cudaFree(d_rope);
  if (d_nope != nullptr) cudaFree(d_nope);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_v32_kv_fp8_ds_mla_pack(
    const NervaCudaDeepSeekKvFp8DsMlaPackRequest *request,
    NervaCudaDeepSeekKvFp8DsMlaPackResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  if (!validate_request(request) || request->block_size != 64u ||
      request->nope_bytes != 512u || request->rope_bf16_values != 64u ||
      request->scale_dim != 16u) {
    return -1;
  }

  const uint64_t rope_bytes = static_cast<uint64_t>(request->rope_bf16_values) * 2;
  const uint64_t token_stride =
      static_cast<uint64_t>(request->nope_bytes) + request->scale_dim +
      rope_bytes;
  const uint64_t block_bytes =
      static_cast<uint64_t>(request->block_size) * token_stride;
  if (token_stride != 656u || block_bytes == 0 || token_stride > UINT32_MAX) {
    return -1;
  }

  out->block_size = request->block_size;
  out->token_index = request->token_index;
  out->token_stride = static_cast<uint32_t>(token_stride);
  out->scale_dim = request->scale_dim;
  out->block_bytes = block_bytes;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    return fail(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  uint8_t *d_nope = nullptr;
  uint16_t *d_rope = nullptr;
  uint8_t *d_scales = nullptr;
  uint8_t *d_output = nullptr;
  uint8_t *h_output = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t nope_bytes = request->nope_bytes;
  const uint64_t rope_input_bytes =
      static_cast<uint64_t>(request->rope_bf16_values) * sizeof(uint16_t);
  const uint64_t scale_bytes = request->scale_dim;

  err = cudaMalloc(reinterpret_cast<void **>(&d_nope), nope_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_rope), rope_input_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_scales), scale_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output), block_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      nope_bytes + rope_input_bytes + scale_bytes + block_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output),
                      block_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = block_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(
      d_nope, request->nope_fp8, nope_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_rope,
                        request->rope_bf16,
                        rope_input_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(
      d_scales, request->scales, scale_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = nope_bytes + rope_input_bytes + scale_bytes;

  err = cudaMemsetAsync(d_output, 0, block_bytes, stream);
  if (err != cudaSuccess) goto cleanup;

  {
    constexpr uint32_t threads = 256;
    const uint32_t blocks = (out->token_stride + threads - 1) / threads;
    v32_fp8_ds_mla_pack_kernel<<<blocks, threads, 0, stream>>>(
        d_nope,
        d_rope,
        d_scales,
        d_output,
        request->token_index,
        request->nope_bytes,
        request->rope_bf16_values,
        request->scale_dim,
        out->token_stride);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(
      h_output, d_output, block_bytes, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = block_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->output_block, h_output, block_bytes);
  out->output_hash = hash_bytes(request->output_block, block_bytes);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output != nullptr) cudaFreeHost(h_output);
  if (d_output != nullptr) cudaFree(d_output);
  if (d_scales != nullptr) cudaFree(d_scales);
  if (d_rope != nullptr) cudaFree(d_rope);
  if (d_nope != nullptr) cudaFree(d_nope);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_compressed_slot_mapping(
    const NervaCudaDeepSeekCompressedSlotMappingRequest *request,
    NervaCudaDeepSeekCompressedSlotMappingResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  if (!validate_request(request)) {
    return -1;
  }

  out->num_tokens = request->num_tokens;
  out->num_reqs = request->num_reqs;
  out->block_table_stride = request->block_table_stride;
  out->block_size = request->block_size;
  out->compress_ratio = request->compress_ratio;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    return fail(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  int32_t *d_query_start_loc = nullptr;
  int32_t *d_seq_lens = nullptr;
  int32_t *d_block_table = nullptr;
  int64_t *d_output_slots = nullptr;
  int64_t *h_output_slots = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t query_bytes =
      static_cast<uint64_t>(request->num_reqs + 1) * sizeof(int32_t);
  const uint64_t seq_bytes =
      static_cast<uint64_t>(request->num_reqs) * sizeof(int32_t);
  const uint64_t table_values =
      static_cast<uint64_t>(request->num_reqs) * request->block_table_stride;
  const uint64_t table_bytes = table_values * sizeof(int32_t);
  const uint64_t output_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int64_t);

  err = cudaMalloc(reinterpret_cast<void **>(&d_query_start_loc), query_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_seq_lens), seq_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_block_table), table_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output_slots), output_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      query_bytes + seq_bytes + table_bytes + output_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output_slots),
                      output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(d_query_start_loc,
                        request->query_start_loc,
                        query_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(
      d_seq_lens, request->seq_lens, seq_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_block_table,
                        request->block_table,
                        table_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = query_bytes + seq_bytes + table_bytes;

  err = cudaMemsetAsync(d_output_slots, 0xff, output_bytes, stream);
  if (err != cudaSuccess) goto cleanup;

  {
    constexpr uint32_t threads = 256;
    compressed_slot_mapping_kernel<<<request->num_reqs, threads, 0, stream>>>(
        d_output_slots,
        d_query_start_loc,
        d_seq_lens,
        d_block_table,
        request->num_tokens,
        request->num_reqs,
        request->block_table_stride,
        request->block_size,
        request->compress_ratio);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_output_slots,
                        d_output_slots,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->output_slots, h_output_slots, output_bytes);
  for (uint32_t idx = 0; idx < request->num_tokens; ++idx) {
    if (request->output_slots[idx] >= 0) {
      out->valid_slots += 1;
    } else {
      out->pad_slots += 1;
    }
  }
  out->output_hash = hash_bytes(
      reinterpret_cast<const uint8_t *>(request->output_slots), output_bytes);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output_slots != nullptr) cudaFreeHost(h_output_slots);
  if (d_output_slots != nullptr) cudaFree(d_output_slots);
  if (d_block_table != nullptr) cudaFree(d_block_table);
  if (d_seq_lens != nullptr) cudaFree(d_seq_lens);
  if (d_query_start_loc != nullptr) cudaFree(d_query_start_loc);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_c128_topk_metadata(
    const NervaCudaDeepSeekC128TopkMetadataRequest *request,
    NervaCudaDeepSeekC128TopkMetadataResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  if (!validate_request(request)) {
    return -1;
  }

  out->num_tokens = request->num_tokens;
  out->num_decode_tokens = request->num_decode_tokens;
  out->num_prefill_tokens = request->num_tokens - request->num_decode_tokens;
  out->num_reqs = request->num_reqs;
  out->block_table_stride = request->block_table_stride;
  out->block_size = request->block_size;
  out->compress_ratio = request->compress_ratio;
  out->max_compressed_tokens = request->max_compressed_tokens;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    return fail(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  int64_t *d_positions = nullptr;
  int32_t *d_token_to_req = nullptr;
  int32_t *d_block_table = nullptr;
  int64_t *d_slot_mapping = nullptr;
  int32_t *d_global_decode = nullptr;
  int32_t *d_decode_lens = nullptr;
  int32_t *d_prefill_local = nullptr;
  int32_t *h_global_decode = nullptr;
  int32_t *h_decode_lens = nullptr;
  int32_t *h_prefill_local = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t positions_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int64_t);
  const uint64_t token_to_req_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int32_t);
  const uint64_t table_bytes =
      static_cast<uint64_t>(request->num_reqs) *
      request->block_table_stride * sizeof(int32_t);
  const uint64_t slot_mapping_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int64_t);
  const uint64_t global_decode_values =
      static_cast<uint64_t>(request->num_decode_tokens) *
      request->max_compressed_tokens;
  const uint64_t prefill_values =
      static_cast<uint64_t>(out->num_prefill_tokens) *
      request->max_compressed_tokens;
  const uint64_t global_decode_bytes =
      global_decode_values * sizeof(int32_t);
  const uint64_t decode_lens_bytes =
      static_cast<uint64_t>(request->num_decode_tokens) * sizeof(int32_t);
  const uint64_t prefill_bytes = prefill_values * sizeof(int32_t);

  err = cudaMalloc(reinterpret_cast<void **>(&d_positions), positions_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_token_to_req),
                   token_to_req_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_block_table), table_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_slot_mapping),
                   slot_mapping_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_global_decode),
                   global_decode_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_decode_lens),
                   decode_lens_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_prefill_local), prefill_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes = positions_bytes + token_to_req_bytes + table_bytes +
                            slot_mapping_bytes + global_decode_bytes +
                            decode_lens_bytes + prefill_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_global_decode),
                      global_decode_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_decode_lens),
                      decode_lens_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_prefill_local),
                      prefill_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes =
      global_decode_bytes + decode_lens_bytes + prefill_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(d_positions,
                        request->positions,
                        positions_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_token_to_req,
                        request->token_to_req_indices,
                        token_to_req_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_block_table,
                        request->block_table,
                        table_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_slot_mapping,
                        request->slot_mapping,
                        slot_mapping_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes =
      positions_bytes + token_to_req_bytes + table_bytes + slot_mapping_bytes;

  err = cudaMemsetAsync(d_global_decode, 0xff, global_decode_bytes, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemsetAsync(d_decode_lens, 0, decode_lens_bytes, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemsetAsync(d_prefill_local, 0xff, prefill_bytes, stream);
  if (err != cudaSuccess) goto cleanup;

  {
    constexpr uint32_t threads = 256;
    c128_topk_metadata_kernel<<<request->num_tokens, threads, 0, stream>>>(
        d_global_decode,
        d_decode_lens,
        d_prefill_local,
        d_positions,
        d_token_to_req,
        d_block_table,
        d_slot_mapping,
        request->num_tokens,
        request->num_decode_tokens,
        request->num_reqs,
        request->block_table_stride,
        request->block_size,
        request->compress_ratio,
        request->max_compressed_tokens);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_global_decode,
                        d_global_decode,
                        global_decode_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_decode_lens,
                        d_decode_lens,
                        decode_lens_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_prefill_local,
                        d_prefill_local,
                        prefill_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = global_decode_bytes + decode_lens_bytes + prefill_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->global_decode, h_global_decode, global_decode_bytes);
  memcpy(request->decode_lens, h_decode_lens, decode_lens_bytes);
  memcpy(request->prefill_local, h_prefill_local, prefill_bytes);
  for (uint32_t idx = 0; idx < request->num_decode_tokens; ++idx) {
    if (request->slot_mapping[idx] >= 0) {
      out->valid_decode_tokens += 1;
    }
    out->decode_entries += request->decode_lens[idx];
  }
  for (uint64_t idx = 0; idx < prefill_values; ++idx) {
    if (request->prefill_local[idx] >= 0) {
      out->prefill_entries += 1;
    }
  }
  out->output_hash = hash_bytes(
      reinterpret_cast<const uint8_t *>(request->global_decode),
      global_decode_bytes);
  out->output_hash ^= hash_bytes(
      reinterpret_cast<const uint8_t *>(request->decode_lens),
      decode_lens_bytes);
  out->output_hash *= 1099511628211ull;
  out->output_hash ^= hash_bytes(
      reinterpret_cast<const uint8_t *>(request->prefill_local),
      prefill_bytes);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_prefill_local != nullptr) cudaFreeHost(h_prefill_local);
  if (h_decode_lens != nullptr) cudaFreeHost(h_decode_lens);
  if (h_global_decode != nullptr) cudaFreeHost(h_global_decode);
  if (d_prefill_local != nullptr) cudaFree(d_prefill_local);
  if (d_decode_lens != nullptr) cudaFree(d_decode_lens);
  if (d_global_decode != nullptr) cudaFree(d_global_decode);
  if (d_slot_mapping != nullptr) cudaFree(d_slot_mapping);
  if (d_block_table != nullptr) cudaFree(d_block_table);
  if (d_token_to_req != nullptr) cudaFree(d_token_to_req);
  if (d_positions != nullptr) cudaFree(d_positions);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_c4_indexer_topk(
    const NervaCudaDeepSeekC4IndexerTopkRequest *request,
    NervaCudaDeepSeekC4IndexerTopkResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  if (!validate_request(request)) {
    return -1;
  }

  out->num_tokens = request->num_tokens;
  out->num_heads = request->num_heads;
  out->head_dim = request->head_dim;
  out->max_compressed_tokens = request->max_compressed_tokens;
  out->topk_tokens = request->topk_tokens;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    return fail(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  float *d_query = nullptr;
  float *d_key_cache = nullptr;
  float *d_weights = nullptr;
  float *d_logits = nullptr;
  int32_t *d_context_lens = nullptr;
  int32_t *d_topk_indices = nullptr;
  float *d_topk_scores = nullptr;
  int32_t *h_topk_indices = nullptr;
  float *h_topk_scores = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t query_values =
      static_cast<uint64_t>(request->num_tokens) * request->num_heads *
      request->head_dim;
  const uint64_t key_values =
      static_cast<uint64_t>(request->max_compressed_tokens) * request->head_dim;
  const uint64_t weight_values =
      static_cast<uint64_t>(request->num_tokens) * request->num_heads;
  const uint64_t logits_values =
      static_cast<uint64_t>(request->num_tokens) *
      request->max_compressed_tokens;
  const uint64_t output_values =
      static_cast<uint64_t>(request->num_tokens) * request->topk_tokens;
  const uint64_t query_bytes = query_values * sizeof(float);
  const uint64_t key_bytes = key_values * sizeof(float);
  const uint64_t weight_bytes = weight_values * sizeof(float);
  const uint64_t logits_bytes = logits_values * sizeof(float);
  const uint64_t context_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int32_t);
  const uint64_t output_index_bytes = output_values * sizeof(int32_t);
  const uint64_t output_score_bytes = output_values * sizeof(float);

  err = cudaMalloc(reinterpret_cast<void **>(&d_query), query_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_key_cache), key_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_weights), weight_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_logits), logits_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_context_lens), context_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_topk_indices),
                   output_index_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_topk_scores),
                   output_score_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes = query_bytes + key_bytes + weight_bytes +
                            logits_bytes +
                            context_bytes + output_index_bytes +
                            output_score_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_topk_indices),
                      output_index_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_topk_scores),
                      output_score_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_index_bytes + output_score_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(
      d_query, request->query, query_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_key_cache,
                        request->key_cache,
                        key_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(
      d_weights, request->weights, weight_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_context_lens,
                        request->context_lens,
                        context_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = query_bytes + key_bytes + weight_bytes + context_bytes;

  {
    constexpr uint32_t kScoreThreads = 128;
    const dim3 score_grid(
        request->num_tokens,
        (request->max_compressed_tokens + kScoreThreads - 1u) /
            kScoreThreads);
    c4_indexer_score_kernel<<<score_grid, kScoreThreads, 0, stream>>>(
        d_logits,
        d_query,
        d_key_cache,
        d_weights,
        d_context_lens,
        request->num_tokens,
        request->num_heads,
        request->head_dim,
        request->max_compressed_tokens);
    out->kernel_launches += 1;
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
    c4_indexer_topk_from_scores_kernel<<<request->num_tokens, 1, 0, stream>>>(
        d_topk_indices,
        d_topk_scores,
        d_logits,
        d_context_lens,
        request->num_tokens,
        request->max_compressed_tokens,
        request->topk_tokens);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_topk_indices,
                        d_topk_indices,
                        output_index_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_topk_scores,
                        d_topk_scores,
                        output_score_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_index_bytes + output_score_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->topk_indices, h_topk_indices, output_index_bytes);
  memcpy(request->topk_scores, h_topk_scores, output_score_bytes);
  for (uint32_t token_idx = 0; token_idx < request->num_tokens; ++token_idx) {
    if (request->context_lens[token_idx] > 0) {
      out->valid_tokens += 1;
    }
  }
  for (uint64_t idx = 0; idx < output_values; ++idx) {
    if (request->topk_indices[idx] >= 0) {
      out->selected_entries += 1;
    }
  }
  out->output_hash = hash_bytes(
      reinterpret_cast<const uint8_t *>(request->topk_indices),
      output_index_bytes);
  out->output_hash ^= hash_bytes(
      reinterpret_cast<const uint8_t *>(request->topk_scores),
      output_score_bytes);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_topk_scores != nullptr) cudaFreeHost(h_topk_scores);
  if (h_topk_indices != nullptr) cudaFreeHost(h_topk_indices);
  if (d_topk_scores != nullptr) cudaFree(d_topk_scores);
  if (d_topk_indices != nullptr) cudaFree(d_topk_indices);
  if (d_context_lens != nullptr) cudaFree(d_context_lens);
  if (d_logits != nullptr) cudaFree(d_logits);
  if (d_weights != nullptr) cudaFree(d_weights);
  if (d_key_cache != nullptr) cudaFree(d_key_cache);
  if (d_query != nullptr) cudaFree(d_query);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_save_partial_states(
    const NervaCudaDeepSeekSavePartialStatesRequest *request,
    NervaCudaDeepSeekSavePartialStatesResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  if (!validate_request(request)) {
    return -1;
  }

  out->num_tokens = request->num_tokens;
  out->block_size = request->block_size;
  out->head_size = request->head_size;
  out->state_width = request->state_width;
  out->compress_ratio = request->compress_ratio;
  out->num_blocks = request->num_blocks;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    return fail(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  float *d_kv = nullptr;
  float *d_score = nullptr;
  float *d_ape = nullptr;
  int64_t *d_positions = nullptr;
  int64_t *d_slot_mapping = nullptr;
  float *d_state_cache = nullptr;
  uint32_t *d_written_flags = nullptr;
  float *h_state_cache = nullptr;
  uint32_t *h_written_flags = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t token_values =
      static_cast<uint64_t>(request->num_tokens) * request->head_size;
  const uint64_t kv_bytes = token_values * sizeof(float);
  const uint64_t score_bytes = token_values * sizeof(float);
  const uint64_t ape_bytes =
      static_cast<uint64_t>(request->compress_ratio) * request->head_size *
      sizeof(float);
  const uint64_t positions_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int64_t);
  const uint64_t slot_mapping_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int64_t);
  const uint64_t state_values =
      static_cast<uint64_t>(request->num_blocks) * request->block_size *
      request->state_width * 2ull;
  const uint64_t state_bytes = state_values * sizeof(float);
  const uint64_t flags_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(uint32_t);

  err = cudaMalloc(reinterpret_cast<void **>(&d_kv), kv_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_score), score_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_ape), ape_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_positions), positions_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_slot_mapping),
                   slot_mapping_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_state_cache), state_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_written_flags), flags_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes = kv_bytes + score_bytes + ape_bytes +
                            positions_bytes + slot_mapping_bytes +
                            state_bytes + flags_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_state_cache),
                      state_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_written_flags),
                      flags_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = state_bytes + flags_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err =
      cudaMemcpyAsync(d_kv, request->kv, kv_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(
      d_score, request->score, score_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(
      d_ape, request->ape, ape_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_positions,
                        request->positions,
                        positions_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_slot_mapping,
                        request->slot_mapping,
                        slot_mapping_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes =
      kv_bytes + score_bytes + ape_bytes + positions_bytes + slot_mapping_bytes;

  err = cudaMemsetAsync(d_state_cache, 0, state_bytes, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemsetAsync(d_written_flags, 0, flags_bytes, stream);
  if (err != cudaSuccess) goto cleanup;

  {
    constexpr uint32_t threads = 256;
    const uint64_t stride1 = static_cast<uint64_t>(request->state_width) * 2ull;
    const uint64_t stride0 = static_cast<uint64_t>(request->block_size) * stride1;
    save_partial_states_kernel<<<request->num_tokens, threads, 0, stream>>>(
        d_state_cache,
        d_kv,
        d_score,
        d_ape,
        d_positions,
        d_slot_mapping,
        request->num_tokens,
        request->block_size,
        request->head_size,
        request->state_width,
        request->compress_ratio,
        request->num_blocks,
        stride0,
        stride1,
        d_written_flags);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_state_cache,
                        d_state_cache,
                        state_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_written_flags,
                        d_written_flags,
                        flags_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = state_bytes + flags_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->state_cache, h_state_cache, state_bytes);
  for (uint32_t idx = 0; idx < request->num_tokens; ++idx) {
    if (h_written_flags[idx] != 0) {
      out->written_tokens += 1;
    } else {
      out->skipped_tokens += 1;
    }
  }
  out->output_hash = hash_bytes(
      reinterpret_cast<const uint8_t *>(request->state_cache), state_bytes);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_written_flags != nullptr) cudaFreeHost(h_written_flags);
  if (h_state_cache != nullptr) cudaFreeHost(h_state_cache);
  if (d_written_flags != nullptr) cudaFree(d_written_flags);
  if (d_state_cache != nullptr) cudaFree(d_state_cache);
  if (d_slot_mapping != nullptr) cudaFree(d_slot_mapping);
  if (d_positions != nullptr) cudaFree(d_positions);
  if (d_ape != nullptr) cudaFree(d_ape);
  if (d_score != nullptr) cudaFree(d_score);
  if (d_kv != nullptr) cudaFree(d_kv);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_compress_norm_rope_fp8_cache(
    const NervaCudaDeepSeekCompressNormRopeFp8CacheRequest *request,
    NervaCudaDeepSeekCompressNormRopeFp8CacheResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  if (!validate_request(request)) {
    return -1;
  }

  out->num_tokens = request->num_tokens;
  out->head_size = request->head_size;
  out->rope_head_dim = request->rope_head_dim;
  out->compress_ratio = request->compress_ratio;
  out->quant_block = request->quant_block;
  out->token_stride = request->token_stride;
  out->scale_dim = request->scale_dim;
  out->scale_format = request->scale_format;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    return fail(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  float *d_state_cache = nullptr;
  int32_t *d_token_to_req = nullptr;
  int64_t *d_positions = nullptr;
  int64_t *d_slot_mapping = nullptr;
  int32_t *d_block_table = nullptr;
  int64_t *d_kv_slot_mapping = nullptr;
  float *d_rms_weight = nullptr;
  float *d_cos_sin_cache = nullptr;
  uint8_t *d_kv_cache = nullptr;
  uint32_t *d_written_flags = nullptr;
  uint8_t *h_kv_cache = nullptr;
  uint32_t *h_written_flags = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t state_values =
      static_cast<uint64_t>(request->num_state_blocks) *
      request->state_block_size * request->state_width * 2ull;
  const uint64_t state_bytes = state_values * sizeof(float);
  const uint64_t token_to_req_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int32_t);
  const uint64_t positions_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int64_t);
  const uint64_t slot_mapping_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int64_t);
  const uint64_t block_table_bytes =
      static_cast<uint64_t>(request->num_reqs) * request->block_table_stride *
      sizeof(int32_t);
  const uint64_t kv_slot_mapping_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int64_t);
  const uint64_t rms_weight_bytes =
      static_cast<uint64_t>(request->head_size) * sizeof(float);
  const uint64_t cos_sin_bytes =
      static_cast<uint64_t>(request->cos_sin_values) * sizeof(float);
  const uint64_t kv_cache_bytes =
      static_cast<uint64_t>(request->num_kv_blocks) *
      request->kv_cache_block_stride;
  const uint64_t flags_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(uint32_t);
  out->kv_cache_bytes = kv_cache_bytes;

  err = cudaMalloc(reinterpret_cast<void **>(&d_state_cache), state_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_token_to_req),
                   token_to_req_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_positions), positions_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_slot_mapping),
                   slot_mapping_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_block_table),
                   block_table_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_kv_slot_mapping),
                   kv_slot_mapping_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_rms_weight),
                   rms_weight_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_cos_sin_cache),
                   cos_sin_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_kv_cache), kv_cache_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_written_flags), flags_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      state_bytes + token_to_req_bytes + positions_bytes + slot_mapping_bytes +
      block_table_bytes + kv_slot_mapping_bytes + rms_weight_bytes +
      cos_sin_bytes + kv_cache_bytes + flags_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_kv_cache),
                      kv_cache_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_written_flags),
                      flags_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = kv_cache_bytes + flags_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(d_state_cache,
                        request->state_cache,
                        state_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_token_to_req,
                        request->token_to_req_indices,
                        token_to_req_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_positions,
                        request->positions,
                        positions_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_slot_mapping,
                        request->slot_mapping,
                        slot_mapping_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_block_table,
                        request->block_table,
                        block_table_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_kv_slot_mapping,
                        request->kv_slot_mapping,
                        kv_slot_mapping_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_rms_weight,
                        request->rms_norm_weight,
                        rms_weight_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_cos_sin_cache,
                        request->cos_sin_cache,
                        cos_sin_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = state_bytes + token_to_req_bytes + positions_bytes +
                   slot_mapping_bytes + block_table_bytes +
                   kv_slot_mapping_bytes + rms_weight_bytes + cos_sin_bytes;

  err = cudaMemsetAsync(d_kv_cache, 0, kv_cache_bytes, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemsetAsync(d_written_flags, 0, flags_bytes, stream);
  if (err != cudaSuccess) goto cleanup;

  {
    constexpr uint32_t threads = kDeepSeekMaxCompressHeadSize;
    compress_norm_rope_fp8_cache_kernel<<<request->num_tokens,
                                           threads,
                                           0,
                                           stream>>>(
        d_state_cache,
        d_token_to_req,
        d_positions,
        d_slot_mapping,
        d_block_table,
        d_kv_slot_mapping,
        d_rms_weight,
        d_cos_sin_cache,
        d_kv_cache,
        d_written_flags,
        request->num_tokens,
        request->num_reqs,
        request->block_table_stride,
        request->state_block_size,
        request->kv_cache_block_size,
        request->head_size,
        request->state_width,
        request->rope_head_dim,
        request->compress_ratio,
        request->overlap,
        request->quant_block,
        request->token_stride,
        request->scale_dim,
        request->scale_format,
        request->num_state_blocks,
        request->num_kv_blocks,
        request->kv_cache_block_stride,
        request->cos_sin_stride,
        request->rms_norm_eps,
        request->fp8_max);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_kv_cache,
                        d_kv_cache,
                        kv_cache_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_written_flags,
                        d_written_flags,
                        flags_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = kv_cache_bytes + flags_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->kv_cache, h_kv_cache, kv_cache_bytes);
  for (uint32_t idx = 0; idx < request->num_tokens; ++idx) {
    if (h_written_flags[idx] != 0) {
      out->written_tokens += 1;
    } else {
      out->skipped_tokens += 1;
    }
  }
  out->output_hash = hash_bytes(request->kv_cache, kv_cache_bytes);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_written_flags != nullptr) cudaFreeHost(h_written_flags);
  if (h_kv_cache != nullptr) cudaFreeHost(h_kv_cache);
  if (d_written_flags != nullptr) cudaFree(d_written_flags);
  if (d_kv_cache != nullptr) cudaFree(d_kv_cache);
  if (d_cos_sin_cache != nullptr) cudaFree(d_cos_sin_cache);
  if (d_rms_weight != nullptr) cudaFree(d_rms_weight);
  if (d_kv_slot_mapping != nullptr) cudaFree(d_kv_slot_mapping);
  if (d_block_table != nullptr) cudaFree(d_block_table);
  if (d_slot_mapping != nullptr) cudaFree(d_slot_mapping);
  if (d_positions != nullptr) cudaFree(d_positions);
  if (d_token_to_req != nullptr) cudaFree(d_token_to_req);
  if (d_state_cache != nullptr) cudaFree(d_state_cache);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}
