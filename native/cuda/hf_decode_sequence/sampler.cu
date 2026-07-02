#include "sampler.cuh"

#include <math.h>

namespace {

constexpr uint32_t kRequestId = 1;
constexpr uint32_t kSequenceId = 1;
constexpr uint32_t kCompletionDeviceComplete = 1;
constexpr uint32_t kDecodeSampleThreads = 1024;
constexpr uint32_t kSamplerTopKMax = 128;
constexpr uint32_t kSamplerRadixBits = 11;
constexpr uint32_t kSamplerRadixBins = 1u << kSamplerRadixBits;
constexpr uint32_t kSamplerCandidateCapacity = 1024;

bool sampler_isfinite(float value) { return isfinite(value) != 0; }

__device__ uint64_t sampler_mix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

__device__ float sampler_uniform_open(uint64_t seed, uint32_t position,
                                      uint32_t token, uint32_t salt) {
  uint64_t value = seed ^ (static_cast<uint64_t>(position) << 32) ^
                   static_cast<uint64_t>(token) ^
                   (static_cast<uint64_t>(salt) << 48);
  value = sampler_mix64(value);
  const uint32_t mantissa = static_cast<uint32_t>((value >> 40) & 0x00ffffffu);
  return (static_cast<float>(mantissa) + 1.0f) / 16777217.0f;
}

__device__ bool sampler_candidate_better(float lhs_value, uint32_t lhs_index,
                                         float rhs_value,
                                         uint32_t rhs_index) {
  return lhs_value > rhs_value ||
         (lhs_value == rhs_value && lhs_index < rhs_index);
}

// Monotone map from float to uint32 so that larger floats map to larger
// unsigned keys. Packs with the bitwise-inverted index so one u64 key orders
// by (value desc, index asc).
__device__ __forceinline__ uint32_t sampler_ordered_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint64_t sampler_candidate_key(float value,
                                                          uint32_t index) {
  return (static_cast<uint64_t>(sampler_ordered_bits(value)) << 32) |
         static_cast<uint64_t>(~index);
}

__device__ __forceinline__ float sampler_key_value(uint64_t key) {
  const uint32_t ordered = static_cast<uint32_t>(key >> 32);
  const uint32_t bits =
      (ordered & 0x80000000u) != 0u ? (ordered & 0x7fffffffu) : ~ordered;
  return __uint_as_float(bits);
}

__device__ __forceinline__ uint32_t sampler_key_index(uint64_t key) {
  return ~static_cast<uint32_t>(key);
}

__global__ void hf_decode_final_head_sample_kernel(
    uint32_t *step_cursor, uint32_t max_steps, uint32_t has_eos_token,
    uint32_t eos_token, const float *scores, uint32_t vocab_size,
    NervaCudaSyntheticTokenSlot *slots, float temperature, float top_p,
    uint32_t top_k, uint64_t sampler_seed) {
  __shared__ float best_values[kDecodeSampleThreads];
  __shared__ uint32_t best_indices[kDecodeSampleThreads];
  __shared__ float top_values[kSamplerTopKMax];
  __shared__ uint32_t top_indices[kSamplerTopKMax];
  __shared__ uint32_t current_position_shared;
  __shared__ uint32_t sampled_index_shared;
  __shared__ uint32_t selected_count_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
    sampled_index_shared = 0;
    selected_count_shared = 0;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  (void)max_steps;
  (void)has_eos_token;
  (void)eos_token;
  const bool greedy = !(temperature > 0.0f) || top_k == 1;
  if (greedy) {
    float best_value = -INFINITY;
    uint32_t best_index = 0;
    for (uint32_t index = threadIdx.x; index < vocab_size; index += blockDim.x) {
      const float value = scores[index];
      if (isfinite(value) &&
          sampler_candidate_better(value, index, best_value, best_index)) {
        best_value = value;
        best_index = index;
      }
    }
    best_values[threadIdx.x] = best_value;
    best_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        const float other_value = best_values[threadIdx.x + stride];
        const uint32_t other_index = best_indices[threadIdx.x + stride];
        if (sampler_candidate_better(other_value, other_index,
                                     best_values[threadIdx.x],
                                     best_indices[threadIdx.x])) {
          best_values[threadIdx.x] = other_value;
          best_indices[threadIdx.x] = other_index;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      sampled_index_shared = best_indices[0];
    }
  } else if (top_k == 0 && top_p >= 0.999999f) {
    float best_value = -INFINITY;
    uint32_t best_index = 0;
    const float inv_temperature = 1.0f / temperature;
    for (uint32_t index = threadIdx.x; index < vocab_size; index += blockDim.x) {
      const float value = scores[index];
      if (!isfinite(value)) {
        continue;
      }
      const float u = sampler_uniform_open(sampler_seed, current_position, index, 0);
      const float gumbel = -logf(-logf(u));
      const float sampled_value = value * inv_temperature + gumbel;
      if (sampler_candidate_better(sampled_value, index, best_value, best_index)) {
        best_value = sampled_value;
        best_index = index;
      }
    }
    best_values[threadIdx.x] = best_value;
    best_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        const float other_value = best_values[threadIdx.x + stride];
        const uint32_t other_index = best_indices[threadIdx.x + stride];
        if (sampler_candidate_better(other_value, other_index,
                                     best_values[threadIdx.x],
                                     best_indices[threadIdx.x])) {
          best_values[threadIdx.x] = other_value;
          best_indices[threadIdx.x] = other_index;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      sampled_index_shared = best_indices[0];
    }
  } else {
    // Exact top-k selection by (value desc, index asc) via radix refinement
    // over 64-bit candidate keys, instead of k full vocab rescans.
    __shared__ uint32_t key_hist[kSamplerRadixBins];
    __shared__ uint64_t candidate_keys[kSamplerCandidateCapacity];
    __shared__ uint32_t candidate_count_shared;
    __shared__ uint64_t key_prefix_shared;
    __shared__ uint32_t need_shared;
    __shared__ int32_t resolved_shift_shared;
    const uint32_t requested_top_k =
        top_k == 0 ? kSamplerTopKMax
                   : (top_k < kSamplerTopKMax ? top_k : kSamplerTopKMax);
    const uint32_t candidate_count =
        requested_top_k < vocab_size ? requested_top_k : vocab_size;

    if (threadIdx.x == 0) {
      candidate_count_shared = 0;
      key_prefix_shared = 0;
      need_shared = candidate_count;
      resolved_shift_shared = -1;
    }
    __syncthreads();

    // Refine an 11-bit digit at a time (64 = 6 levels) until the candidates
    // sharing the running prefix fit in shared memory. Keys are unique (the
    // index occupies the low bits), so this always terminates.
    for (int32_t shift = 64 - kSamplerRadixBits;
         resolved_shift_shared < 0 && shift >= 0;
         shift -= kSamplerRadixBits) {
      for (uint32_t bin = threadIdx.x; bin < kSamplerRadixBins;
           bin += blockDim.x) {
        key_hist[bin] = 0;
      }
      __syncthreads();
      const uint64_t prefix = key_prefix_shared;
      const uint64_t prefix_mask =
          shift + kSamplerRadixBits >= 64
              ? 0ull
              : (~0ull << (shift + kSamplerRadixBits));
      for (uint32_t index = threadIdx.x; index < vocab_size;
           index += blockDim.x) {
        const float value = scores[index];
        if (!isfinite(value)) {
          continue;
        }
        const uint64_t key = sampler_candidate_key(value, index);
        if ((key & prefix_mask) != prefix) {
          continue;
        }
        atomicAdd(key_hist + ((key >> shift) & (kSamplerRadixBins - 1u)), 1u);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        uint32_t cumulative = 0;
        int32_t threshold_bin = -1;
        for (int32_t bin = kSamplerRadixBins - 1; bin >= 0; --bin) {
          const uint32_t next = cumulative + key_hist[bin];
          if (next >= need_shared) {
            threshold_bin = bin;
            break;
          }
          cumulative = next;
        }
        if (threshold_bin < 0) {
          // Fewer finite candidates than requested: take everything with the
          // current prefix.
          resolved_shift_shared = shift;
        } else if (cumulative + key_hist[threshold_bin] <=
                       kSamplerCandidateCapacity - candidate_count_shared ||
                   shift == 0) {
          // Everything at or above the threshold digit fits (or keys are
          // fully resolved); stop refining.
          key_prefix_shared =
              key_prefix_shared |
              (static_cast<uint64_t>(threshold_bin) << shift);
          resolved_shift_shared = shift;
        } else {
          // Threshold digit is too crowded; descend into it.
          key_prefix_shared =
              key_prefix_shared |
              (static_cast<uint64_t>(threshold_bin) << shift);
          need_shared -= cumulative;
          // Candidates strictly above the threshold digit are final
          // selections; stash them now so the deeper levels only track the
          // contested bin. They are fewer than `need`, which is <= capacity.
          // (Compacted below, outside this single-thread section.)
        }
      }
      __syncthreads();
      if (resolved_shift_shared < 0) {
        // Compact finalists strictly above the refined prefix at this level.
        const uint64_t prefix = key_prefix_shared;
        const uint64_t prefix_mask =
            shift + kSamplerRadixBits >= 64
                ? 0ull
                : (~0ull << (shift + kSamplerRadixBits));
        const uint64_t digit_mask = ~0ull << shift;
        for (uint32_t index = threadIdx.x; index < vocab_size;
             index += blockDim.x) {
          const float value = scores[index];
          if (!isfinite(value)) {
            continue;
          }
          const uint64_t key = sampler_candidate_key(value, index);
          if ((key & prefix_mask) != (prefix & prefix_mask)) {
            continue;
          }
          if ((key & digit_mask) > (prefix & digit_mask)) {
            const uint32_t slot = atomicAdd(&candidate_count_shared, 1u);
            if (slot < kSamplerCandidateCapacity) {
              candidate_keys[slot] = key;
            }
          }
        }
        __syncthreads();
      }
    }

    // Compact every candidate whose digit at the resolved level is at or
    // above the threshold digit, under the same higher-level prefix.
    // Finalists from descended levels were already stashed above.
    {
      const int32_t shift =
          resolved_shift_shared < 0 ? 0 : resolved_shift_shared;
      const uint64_t threshold = key_prefix_shared;
      const uint64_t prefix_mask =
          shift + kSamplerRadixBits >= 64
              ? 0ull
              : (~0ull << (shift + kSamplerRadixBits));
      for (uint32_t index = threadIdx.x; index < vocab_size;
           index += blockDim.x) {
        const float value = scores[index];
        if (!isfinite(value)) {
          continue;
        }
        const uint64_t key = sampler_candidate_key(value, index);
        if ((key & prefix_mask) == (threshold & prefix_mask) &&
            (key >> shift) >= (threshold >> shift)) {
          const uint32_t slot = atomicAdd(&candidate_count_shared, 1u);
          if (slot < kSamplerCandidateCapacity) {
            candidate_keys[slot] = key;
          }
        }
      }
      __syncthreads();
    }

    // Pad to a power of two and bitonic-sort descending (key order already
    // encodes value desc, index asc).
    uint32_t compacted = candidate_count_shared;
    if (compacted > kSamplerCandidateCapacity) {
      compacted = kSamplerCandidateCapacity;
    }
    uint32_t sort_size = 1;
    while (sort_size < compacted) {
      sort_size <<= 1;
    }
    for (uint32_t slot = compacted + threadIdx.x; slot < sort_size;
         slot += blockDim.x) {
      candidate_keys[slot] = 0;
    }
    __syncthreads();
    for (uint32_t width = 2; width <= sort_size; width <<= 1) {
      for (uint32_t stride = width >> 1; stride != 0; stride >>= 1) {
        for (uint32_t index = threadIdx.x; index < sort_size;
             index += blockDim.x) {
          const uint32_t other = index ^ stride;
          if (other <= index) {
            continue;
          }
          const bool descending = (index & width) == 0;
          const uint64_t left = candidate_keys[index];
          const uint64_t right = candidate_keys[other];
          if (descending ? (left < right) : (left > right)) {
            candidate_keys[index] = right;
            candidate_keys[other] = left;
          }
        }
        __syncthreads();
      }
    }

    const uint32_t selected =
        compacted < candidate_count ? compacted : candidate_count;
    for (uint32_t rank = threadIdx.x; rank < selected; rank += blockDim.x) {
      const uint64_t key = candidate_keys[rank];
      top_values[rank] = sampler_key_value(key);
      top_indices[rank] = sampler_key_index(key);
    }
    if (threadIdx.x == 0) {
      selected_count_shared = selected;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const uint32_t selected_count = selected_count_shared;
      if (selected_count == 0) {
        sampled_index_shared = 0;
      } else {
        const float inv_temperature = 1.0f / temperature;
        const float max_value = top_values[0];
        float total = 0.0f;
        for (uint32_t index = 0; index < selected_count; ++index) {
          total += expf((top_values[index] - max_value) * inv_temperature);
        }
        float kept_total = 0.0f;
        uint32_t kept_count = selected_count;
        if (top_p < 0.999999f) {
          kept_count = 0;
          for (uint32_t index = 0; index < selected_count; ++index) {
            kept_total += expf((top_values[index] - max_value) * inv_temperature);
            kept_count = index + 1;
            if (total > 0.0f && kept_total / total >= top_p) {
              break;
            }
          }
        } else {
          kept_total = total;
        }
        if (kept_total <= 0.0f || kept_count == 0) {
          sampled_index_shared = top_indices[0];
        } else {
          const float draw =
              sampler_uniform_open(sampler_seed, current_position, 0, 1) *
              kept_total;
          float cumulative = 0.0f;
          sampled_index_shared = top_indices[kept_count - 1];
          for (uint32_t index = 0; index < kept_count; ++index) {
            cumulative += expf((top_values[index] - max_value) * inv_temperature);
            if (draw <= cumulative) {
              sampled_index_shared = top_indices[index];
              break;
            }
          }
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t token = sampled_index_shared;
    NervaCudaSyntheticTokenSlot *slot = slots + current_position;
    slot->request_id = kRequestId;
    slot->sequence_id = kSequenceId;
    slot->token_index = current_position;
    slot->token = token;
    slot->version = current_position + 1;
    slot->completion = kCompletionDeviceComplete;
    slot->host_copied = 0;
    if (step_cursor != nullptr) {
      *step_cursor = current_position + 1;
    }
  }
}

}  // namespace

NervaCudaHfDecodeSamplerConfig default_hf_decode_sampler_config() {
  NervaCudaHfDecodeSamplerConfig config{};
  config.temperature = 0.7f;
  config.top_p = 0.9f;
  config.top_k = 0;
  config.reserved = 0;
  config.seed = 0;
  return config;
}

NervaCudaHfDecodeSamplerConfig normalize_hf_decode_sampler_config(
    NervaCudaHfDecodeSamplerConfig config) {
  if (!sampler_isfinite(config.temperature) || config.temperature < 0.0f) {
    config.temperature = 0.0f;
  }
  if (!sampler_isfinite(config.top_p) || config.top_p <= 0.0f ||
      config.top_p > 1.0f) {
    config.top_p = 1.0f;
  }
  config.reserved = 0;
  return config;
}

bool hf_decode_sampler_config_matches(
    const NervaCudaHfDecodeSamplerConfig &lhs,
    const NervaCudaHfDecodeSamplerConfig &rhs) {
  return lhs.temperature == rhs.temperature && lhs.top_p == rhs.top_p &&
         lhs.top_k == rhs.top_k && lhs.seed == rhs.seed;
}

cudaError_t launch_hf_decode_final_head_sampler(
    cudaStream_t stream, uint32_t *step_cursor, uint32_t max_steps,
    uint32_t has_eos_token, uint32_t eos_token, const float *scores,
    uint32_t vocab_size, NervaCudaSyntheticTokenSlot *slots,
    NervaCudaHfDecodeSamplerConfig sampler) {
  hf_decode_final_head_sample_kernel<<<1, kDecodeSampleThreads, 0, stream>>>(
      step_cursor, max_steps, has_eos_token, eos_token, scores, vocab_size,
      slots, sampler.temperature, sampler.top_p, sampler.top_k, sampler.seed);
  return cudaGetLastError();
}
