#include "nerva_cuda_api.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#if NERVA_HAVE_CUDNN_FRONTEND
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <memory>
#endif

namespace {

constexpr uint32_t kDTypeF16 = 0;
constexpr uint32_t kDTypeBF16 = 1;
constexpr uint32_t kRequestId = 1;
constexpr uint32_t kSequenceId = 1;
constexpr uint32_t kCompletionDeviceComplete = 1;
constexpr uint32_t kWeightStrategyGpuResident = 1;
constexpr uint32_t kWeightStrategyGpuStaged = 2;
constexpr uint32_t kDecodeThreads = 256;
constexpr uint32_t kDecodeNormThreads = 1024;
constexpr uint32_t kDecodeSampleThreads = 1024;
constexpr uint32_t kHeadThreadsMax = 256;
constexpr uint32_t kHeadThreadElements = 4;
constexpr uint32_t kPrefillChunkBaseTokens = 1024;
constexpr uint32_t kPrefillChunkMaxTokens = 8192;
constexpr uint32_t kKvCacheBlockTokens = 16;
constexpr uint32_t kDecodeAttentionChunkTokens = 64;
constexpr uint32_t kGroupedGqaHeads = 4;
constexpr uint32_t kGroupedGqaThreadsPerHead = 64;
constexpr uint32_t kGroupedGqaThreads =
    kGroupedGqaHeads * kGroupedGqaThreadsPerHead;
constexpr uint32_t kGroupedGqaHeadDimMax =
    kGroupedGqaThreadsPerHead * kHeadThreadElements;
constexpr uint32_t kSharedWarpGqaThreadsPerHead = 32;
constexpr uint32_t kSharedWarpGqaThreads =
    kGroupedGqaHeads * kSharedWarpGqaThreadsPerHead;
constexpr uint32_t kSharedWarpGqaHeadDimMax =
    kSharedWarpGqaThreadsPerHead * kHeadThreadElements;
constexpr uint32_t kSharedWarpGqaTileTokens = kKvCacheBlockTokens;
constexpr uint32_t kSharedWarpGqaTileElements =
    kSharedWarpGqaTileTokens * kSharedWarpGqaHeadDimMax;
constexpr uint32_t kLtGemvMaxHeuristics = 32;
constexpr uint32_t kLtGemvAutotuneWarmups = 1;
constexpr uint32_t kLtGemvAutotuneIterations = 3;
constexpr uint32_t kChunkedDecodeAttentionThreshold = 128;
constexpr uint64_t kMissingOffset = UINT64_MAX;
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x00000100000001b3ull;
constexpr size_t kCublasWorkspaceBytes = 64ull * 1024ull * 1024ull;
constexpr uint64_t kDescriptorStreamStagingBytes = 64ull * 1024ull * 1024ull;
constexpr uint64_t kPrefillAutotuneSafetyBytes = 256ull * 1024ull * 1024ull;

enum CreateFailureStage : int32_t {
  kCreateStageNone = 0,
  kCreateStageInvalidRequest = 1,
  kCreateStageGetDeviceCount = 2,
  kCreateStageSetDevice = 3,
  kCreateStageSessionAlloc = 4,
  kCreateStageHostWeightAlloc = 5,
  kCreateStageHostSlotsAlloc = 6,
  kCreateStageDeviceArenaAlloc = 7,
  kCreateStageDeviceLayoutsAlloc = 8,
  kCreateStageDeviceScratchAlloc = 9,
  kCreateStageProjectionInputAlloc = 10,
  kCreateStagePackedQkvAlloc = 11,
  kCreateStagePackedGateUpAlloc = 12,
  kCreateStageKvKeysAlloc = 13,
  kCreateStageKvValuesAlloc = 14,
  kCreateStagePromptTokensAlloc = 15,
  kCreateStageDeviceSlotsAlloc = 16,
  kCreateStageDeviceStepAlloc = 17,
  kCreateStageCublasWorkspaceAlloc = 18,
  kCreateStageStreamCreate = 19,
  kCreateStageCublasCreate = 20,
  kCreateStageCublasLtCreate = 21,
  kCreateStageCublasConfigure = 22,
  kCreateStageStartEventCreate = 23,
  kCreateStageStopEventCreate = 24,
  kCreateStageDescriptorCopy = 25,
  kCreateStageLayoutCopy = 26,
  kCreateStagePackReplicas = 27,
  kCreateStageWarmCublas = 28,
  kCreateStageSetupSynchronize = 29,
  kCreateStagePrefillHiddenAlloc = 30,
  kCreateStagePrefillChunkAlloc = 31,
  kCreateStageDecodeAttentionAlloc = 32,
  kCreateStageDecodeSdpaAlloc = 33,
  kCreateStageProjectionPlanAutotune = 34,
};

struct SequenceArenaLayout {
  uint64_t embeddings;
  uint64_t input;
  uint64_t scratch;
  uint64_t final_norm;
  uint64_t lm_head;
};

struct SequenceLayerLayout {
  uint64_t rms_attn;
  uint64_t rms_mlp;
  uint64_t w_q;
  uint64_t w_k;
  uint64_t q_norm;
  uint64_t k_norm;
  uint64_t w_v;
  uint64_t w_o;
  uint64_t q_bias;
  uint64_t k_bias;
  uint64_t v_bias;
  uint64_t o_bias;
  uint64_t w_gate;
  uint64_t w_up;
  uint64_t w_down;
};

struct PackedProjectionShape {
  uint64_t qkv_rows;
  uint64_t gate_up_rows;
  uint64_t qkv_elements_per_layer;
  uint64_t gate_up_elements_per_layer;
};

struct LayerScratch {
  float *input;
  float *attn_norm;
  float *q;
  float *k;
  float *v;
  float *attn;
  float *residual;
  float *mlp_norm;
  float *gate;
  float *up;
  float *ff;
  float *down;
};

struct LtGemvPlan {
  uint32_t rows = 0;
  uint32_t cols = 0;
  uint32_t dtype = 0;
  uint32_t backend = 0;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulAlgo_t algo{};
  bool ready = false;
  bool has_algo = false;
  uint32_t heuristic_count = 0;
  uint32_t selected_heuristic = UINT32_MAX;
  uint64_t tuned_avg_ns = 0;
};

struct LtGemmTokensPlan {
  uint32_t rows = 0;
  uint32_t cols = 0;
  uint32_t tokens = 0;
  uint32_t dtype = 0;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  bool ready = false;
};

constexpr uint32_t kGemvBackendLt = 0;
constexpr uint32_t kGemvBackendCublas = 1;

constexpr uint32_t kProjectionBatchPlanReady = 0;
constexpr uint32_t kProjectionBatchPlanInvalidRequest = 1;
constexpr uint32_t kProjectionBatchPlanNoSessions = 2;
constexpr uint32_t kProjectionBatchPlanNoReadySessions = 3;
constexpr uint32_t kProjectionBatchPlanSharedWeightsUnproven = 4;
constexpr uint32_t kProjectionBatchPlanInsufficientCompatibleReady = 5;
constexpr uint32_t kProjectionBatchPlanUnsupportedProjection = 6;
constexpr uint32_t kProjectionBatchPlanInvalidLayer = 7;
constexpr uint32_t kProjectionBatchPlanInsufficientScratch = 8;
constexpr uint32_t kProjectionBatchKindQkv = 1;
constexpr uint32_t kProjectionBatchKindAttentionOutput = 2;
constexpr uint32_t kProjectionBatchKindGateUp = 3;
constexpr uint32_t kProjectionBatchKindDown = 4;
constexpr uint32_t kProjectionBatchKindLmHead = 5;

__host__ __device__ LayerScratch layer_scratch_ptrs(
    float *scratch, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate) {
  LayerScratch out{};
  out.input = scratch;
  out.attn_norm = out.input + hidden;
  out.q = out.attn_norm + hidden;
  out.k = out.q + attention_hidden;
  out.v = out.k + kv_hidden;
  out.attn = out.v + kv_hidden;
  out.residual = out.attn + attention_hidden;
  out.mlp_norm = out.residual + hidden;
  out.gate = out.mlp_norm + hidden;
  out.up = out.gate + intermediate;
  out.ff = out.up + intermediate;
  out.down = out.ff + intermediate;
  return out;
}

__device__ __forceinline__ float encoded_to_f32(uint16_t value, uint32_t dtype) {
  if (dtype == kDTypeBF16) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
  }
  return __half2float(__ushort_as_half(value));
}

template <uint32_t DType>
__device__ __forceinline__ float encoded_to_f32_typed(uint16_t value) {
  if (DType == kDTypeBF16) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
  }
  return __half2float(__ushort_as_half(value));
}

__device__ __forceinline__ uint16_t f32_to_encoded(float value, uint32_t dtype) {
  if (dtype == kDTypeBF16) {
    uint32_t bits = __float_as_uint(value);
    uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
  }
  return __half_as_ushort(__float2half_rn(value));
}

__host__ __device__ __forceinline__ uint64_t kv_cache_page_offset(
    uint32_t layer_index, uint32_t kv_block_count, uint32_t physical_block,
    uint32_t block_offset, uint32_t kv_hidden, uint32_t kv_offset) {
  return (((static_cast<uint64_t>(layer_index) * kv_block_count +
            physical_block) *
               kKvCacheBlockTokens +
           block_offset) *
              kv_hidden +
          kv_offset);
}

__device__ __forceinline__ uint64_t kv_cache_token_offset(
    uint32_t layer_index, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint32_t token, uint32_t kv_hidden,
    uint32_t kv_offset) {
  const uint32_t logical_block = token / kKvCacheBlockTokens;
  const uint32_t block_offset = token - logical_block * kKvCacheBlockTokens;
  const uint32_t physical_block = kv_block_table[logical_block];
  return kv_cache_page_offset(layer_index, kv_block_count, physical_block,
                              block_offset, kv_hidden, kv_offset);
}

__device__ __forceinline__ uint64_t kv_cache_token_base(
    uint32_t layer_index, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint32_t token, uint32_t kv_hidden,
    uint32_t kv_offset) {
  const uint32_t logical_block = token / kKvCacheBlockTokens;
  const uint32_t block_offset = token - logical_block * kKvCacheBlockTokens;
  const uint32_t physical_block = kv_block_table[logical_block];
  return kv_cache_page_offset(layer_index, kv_block_count, physical_block,
                              block_offset, kv_hidden, kv_offset);
}

__device__ float silu(float value) {
  return value / (1.0f + expf(-value));
}

__device__ float block_sum(float value) {
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

__device__ __forceinline__ float warp_sum(float value) {
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ void encoded_slice_to_f32(const uint16_t *input, uint32_t len,
                                     uint32_t dtype, float *output) {
  for (uint32_t index = threadIdx.x; index < len; index += blockDim.x) {
    output[index] = encoded_to_f32(input[index], dtype);
  }
  __syncthreads();
}

__device__ void f32_slice_to_encoded(const float *input, uint16_t *output,
                                     uint32_t len, uint32_t dtype) {
  for (uint32_t index = threadIdx.x; index < len; index += blockDim.x) {
    output[index] = f32_to_encoded(input[index], dtype);
  }
  __syncthreads();
}

__device__ void copy_encoded_slice(uint16_t *dst, const uint16_t *src, uint32_t len) {
  for (uint32_t index = threadIdx.x; index < len; index += blockDim.x) {
    dst[index] = src[index];
  }
  __syncthreads();
}

__device__ void mat_vec(const uint16_t *matrix, const float *input, uint32_t rows,
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

__device__ void rms_norm(const float *input, const uint16_t *weight, uint32_t hidden,
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

__device__ void rms_norm_to_encoded(const float *input, const uint16_t *weight,
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

__device__ void add_bias(const uint16_t *arena, uint64_t offset, uint32_t len,
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

__device__ void per_head_rms_norm(uint16_t *arena, uint64_t offset, float *values,
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

__device__ void add_optional_head_bias(const uint16_t *arena, uint64_t offset,
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

__device__ void per_head_rms_norm_block(uint16_t *arena, uint64_t offset,
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

__device__ void apply_rope_head(float *values, uint32_t head_dim,
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

__device__ void apply_rope(float *values, uint32_t heads, uint32_t head_dim,
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

__device__ void run_layer(uint16_t *arena, SequenceLayerLayout layout,
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

__global__ void hf_decode_final_head_reduce_kernel(
    uint32_t *step_cursor, uint32_t max_steps, uint32_t has_eos_token,
    uint32_t eos_token, const float *scores, uint32_t vocab_size,
    NervaCudaSyntheticTokenSlot *slots) {
  __shared__ float best_values[kDecodeSampleThreads];
  __shared__ uint32_t best_indices[kDecodeSampleThreads];
  __shared__ uint32_t current_position_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  (void)max_steps;
  (void)has_eos_token;
  (void)eos_token;
  float best_value = -INFINITY;
  uint32_t best_index = 0;
  for (uint32_t index = threadIdx.x; index < vocab_size; index += blockDim.x) {
    const float value = scores[index];
    if (isfinite(value) && (value > best_value ||
                            (value == best_value && index < best_index))) {
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
      if (other_value > best_values[threadIdx.x] ||
          (other_value == best_values[threadIdx.x] &&
           other_index < best_indices[threadIdx.x])) {
        best_values[threadIdx.x] = other_value;
        best_indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const uint32_t best_index = best_indices[0];
    NervaCudaSyntheticTokenSlot *slot = slots + current_position;
    slot->request_id = kRequestId;
    slot->sequence_id = kSequenceId;
    slot->token_index = current_position;
    slot->token = best_index;
    slot->version = current_position + 1;
    slot->completion = kCompletionDeviceComplete;
    slot->host_copied = 0;
    if (step_cursor != nullptr) {
      *step_cursor = current_position + 1;
    }
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
    const NervaCudaSyntheticTokenSlot *slots) {
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
              kv_values, kv_block_count, kv_block_table);
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
    arena[arena_layout.input + index] = encoded;
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

__global__ void hf_projection_batch_scatter_f32_kernel(
    const float *src, float *dst, uint32_t rows, uint32_t token_index) {
  const uint64_t offset = static_cast<uint64_t>(token_index) * rows;
  const uint32_t stride = static_cast<uint32_t>(gridDim.x) * blockDim.x;
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += stride) {
    dst[row] = src[offset + row];
  }
}

uint64_t push(uint64_t &cursor, uint64_t len) {
  const uint64_t offset = cursor;
  cursor += len;
  return offset;
}

uint64_t push_optional(uint64_t &cursor, uint64_t len, const uint16_t *ptr) {
  if (ptr == nullptr) {
    return kMissingOffset;
  }
  return push(cursor, len);
}

uint64_t hash_tokens(const uint32_t *tokens, uint32_t count) {
  uint64_t hash = kFnvOffset;
  for (uint32_t index = 0; index < count; ++index) {
    uint32_t token = tokens[index];
    for (uint32_t byte = 0; byte < 4; ++byte) {
      hash ^= static_cast<uint64_t>((token >> (8u * byte)) & 0xffu);
      hash *= kFnvPrime;
    }
  }
  return hash;
}

void hash_u32(uint64_t &hash, uint32_t value) {
  for (uint32_t byte = 0; byte < 4; ++byte) {
    hash ^= static_cast<uint64_t>((value >> (8u * byte)) & 0xffu);
    hash *= kFnvPrime;
  }
}

void hash_u64(uint64_t &hash, uint64_t value) {
  for (uint32_t byte = 0; byte < 8; ++byte) {
    hash ^= static_cast<uint64_t>((value >> (8u * byte)) & 0xffu);
    hash *= kFnvPrime;
  }
}

void hash_descriptor(uint64_t &hash,
                     const NervaCudaHfDecodeSequenceWeightBlock &descriptor) {
  hash_u64(hash, descriptor.block_id);
  hash_u64(hash, descriptor.block_version);
  hash_u64(hash, descriptor.offset_bytes);
  hash_u64(hash, descriptor.bytes);
  hash_u32(hash, descriptor.strategy);
}

bool descriptor_has_memory_source(
    const NervaCudaHfDecodeSequenceWeightBlock &descriptor) {
  return descriptor.host_source != nullptr;
}

bool descriptor_has_file_source(
    const NervaCudaHfDecodeSequenceWeightBlock &descriptor) {
  return descriptor.source_file != nullptr && descriptor.source_file_len != 0;
}

bool descriptor_has_source(
    const NervaCudaHfDecodeSequenceWeightBlock &descriptor) {
  return descriptor_has_memory_source(descriptor) ||
         descriptor_has_file_source(descriptor);
}

template <typename Request>
bool descriptors_require_file_staging(const Request *request) {
  if (request == nullptr || request->planned_weight_descriptor_count == 0 ||
      request->planned_weight_descriptors == nullptr) {
    return false;
  }
  for (uint32_t index = 0; index < request->planned_weight_descriptor_count; ++index) {
    if (descriptor_has_file_source(request->planned_weight_descriptors[index])) {
      return true;
    }
  }
  return false;
}

template <typename Request>
uint64_t pinned_weight_staging_bytes(const Request *request,
                                     uint64_t full_weight_bytes) {
  if (request->planned_weight_blocks == 0 && request->planned_weight_bytes == 0) {
    return full_weight_bytes;
  }
  if (!descriptors_require_file_staging(request)) {
    return sizeof(uint16_t);
  }
  uint64_t bytes = std::min(full_weight_bytes, kDescriptorStreamStagingBytes);
  bytes -= bytes % sizeof(uint16_t);
  return bytes == 0 ? sizeof(uint16_t) : bytes;
}

template <typename Request>
bool has_declared_weight_plan(const Request *request) {
  return request->planned_weight_blocks != 0 || request->planned_weight_bytes != 0;
}

bool valid_layer(const NervaCudaHfDecodeChainLayer &layer, bool require_sources) {
  if (!require_sources) {
    return true;
  }
  return layer.rms_attn_weight != nullptr && layer.rms_mlp_weight != nullptr &&
         layer.w_q != nullptr && layer.w_k != nullptr && layer.w_v != nullptr &&
         layer.w_o != nullptr && layer.w_gate != nullptr && layer.w_up != nullptr &&
         layer.w_down != nullptr;
}

bool valid_request(const NervaCudaHfDecodeSequenceRequest *request) {
  if (request == nullptr) {
    return false;
  }
  const bool declared_weight_plan = has_declared_weight_plan(request);
  if (request->layers == nullptr || request->output_tokens == nullptr ||
      request->prompt_tokens == nullptr ||
      (!declared_weight_plan &&
       (request->embeddings == nullptr || request->final_norm_weight == nullptr ||
        request->lm_head == nullptr)) ||
      request->output_token_capacity < request->steps || request->layer_count == 0 ||
      request->steps == 0 || request->prompt_token_count == 0 ||
      request->hidden == 0 || request->heads == 0 || request->kv_heads == 0 ||
      request->head_dim == 0 || request->intermediate == 0 ||
      request->vocab_size == 0 || request->seed_token >= request->vocab_size ||
      request->kv_heads > request->heads || request->heads % request->kv_heads != 0 ||
      request->dtype > kDTypeBF16 ||
      (request->rope_theta > 0.0f && request->head_dim % 2 != 0) ||
      request->prompt_token_count > UINT32_MAX - request->steps + 1u) {
    return false;
  }
  for (uint32_t index = 0; index < request->prompt_token_count; ++index) {
    if (request->prompt_tokens[index] >= request->vocab_size) {
      return false;
    }
  }
  if (request->prompt_tokens[request->prompt_token_count - 1u] != request->seed_token) {
    return false;
  }
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    if (!valid_layer(request->layers[index], !declared_weight_plan)) {
      return false;
    }
  }
  if (declared_weight_plan) {
    if (request->planned_weight_blocks == 0 || request->planned_weight_bytes == 0) {
      return false;
    }
    if (request->planned_weight_descriptors == nullptr ||
        request->planned_weight_descriptor_count != request->planned_weight_blocks ||
        request->planned_weight_descriptor_hash == 0) {
      return false;
    }
    if (request->planned_gpu_resident_blocks > request->planned_weight_blocks ||
        request->planned_gpu_staged_blocks >
            request->planned_weight_blocks - request->planned_gpu_resident_blocks) {
      return false;
    }
    if (request->planned_gpu_resident_weight_bytes > request->planned_weight_bytes ||
        request->planned_gpu_staged_weight_bytes >
            request->planned_weight_bytes - request->planned_gpu_resident_weight_bytes) {
      return false;
    }
  }
  return true;
}

void clear_result(const NervaCudaHfDecodeSequenceRequest *request,
                  NervaCudaHfDecodeSequenceResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request != nullptr) {
    out->dtype = request->dtype;
    out->hidden = request->hidden;
    out->heads = request->heads;
    out->kv_heads = request->kv_heads;
    out->head_dim = request->head_dim;
    out->intermediate = request->intermediate;
    out->vocab_size = request->vocab_size;
    out->layer_count = request->layer_count;
    out->steps = request->steps;
    out->seed_token = request->seed_token;
    out->planned_weight_blocks = request->planned_weight_blocks;
    out->planned_gpu_resident_blocks = request->planned_gpu_resident_blocks;
    out->planned_gpu_staged_blocks = request->planned_gpu_staged_blocks;
    out->planned_weight_bytes = request->planned_weight_bytes;
    out->planned_gpu_resident_weight_bytes =
        request->planned_gpu_resident_weight_bytes;
    out->planned_gpu_staged_weight_bytes =
        request->planned_gpu_staged_weight_bytes;
    out->planned_weight_descriptor_count =
        request->planned_weight_descriptor_count;
    out->planned_weight_descriptor_hash =
        request->planned_weight_descriptor_hash;
  }
}

void clear_session_create_result(
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request,
    NervaCudaHfDecodeSequenceSessionCreateResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request != nullptr) {
    out->dtype = request->dtype;
    out->hidden = request->hidden;
    out->heads = request->heads;
    out->kv_heads = request->kv_heads;
    out->head_dim = request->head_dim;
    out->intermediate = request->intermediate;
    out->vocab_size = request->vocab_size;
    out->layer_count = request->layer_count;
    out->max_context_tokens = request->max_context_tokens;
    out->planned_weight_blocks = request->planned_weight_blocks;
    out->planned_gpu_resident_blocks = request->planned_gpu_resident_blocks;
    out->planned_gpu_staged_blocks = request->planned_gpu_staged_blocks;
    out->planned_weight_bytes = request->planned_weight_bytes;
    out->planned_gpu_resident_weight_bytes =
        request->planned_gpu_resident_weight_bytes;
    out->planned_gpu_staged_weight_bytes =
        request->planned_gpu_staged_weight_bytes;
    out->planned_weight_descriptor_count =
        request->planned_weight_descriptor_count;
    out->planned_weight_descriptor_hash =
        request->planned_weight_descriptor_hash;
  }
}

template <typename Request, typename Result>
bool validate_weight_descriptors(const Request *request,
                                 uint64_t resident_weight_bytes,
                                 Result *out) {
  if (request->planned_weight_blocks == 0) {
    return true;
  }
  uint64_t cursor = 0;
  uint64_t descriptor_hash = kFnvOffset;
  uint64_t resident_bytes = 0;
  uint64_t staged_bytes = 0;
  uint32_t resident_blocks = 0;
  uint32_t staged_blocks = 0;
  for (uint32_t index = 0; index < request->planned_weight_descriptor_count; ++index) {
    const auto &descriptor = request->planned_weight_descriptors[index];
    if (descriptor.bytes == 0 || descriptor.reserved != 0 ||
        !descriptor_has_source(descriptor) || descriptor.offset_bytes != cursor ||
        descriptor.offset_bytes % sizeof(uint16_t) != 0 ||
        descriptor.bytes % sizeof(uint16_t) != 0) {
      return false;
    }
    const uint64_t next_cursor = cursor + descriptor.bytes;
    if (next_cursor < cursor) {
      return false;
    }
    cursor = next_cursor;
    hash_descriptor(descriptor_hash, descriptor);
    if (descriptor.strategy == kWeightStrategyGpuResident) {
      resident_blocks += 1;
      const uint64_t next_resident_bytes = resident_bytes + descriptor.bytes;
      if (next_resident_bytes < resident_bytes) {
        return false;
      }
      resident_bytes = next_resident_bytes;
    } else if (descriptor.strategy == kWeightStrategyGpuStaged) {
      staged_blocks += 1;
      const uint64_t next_staged_bytes = staged_bytes + descriptor.bytes;
      if (next_staged_bytes < staged_bytes) {
        return false;
      }
      staged_bytes = next_staged_bytes;
    } else {
      return false;
    }
  }
  if (cursor != resident_weight_bytes || cursor != request->planned_weight_bytes ||
      descriptor_hash != request->planned_weight_descriptor_hash ||
      resident_blocks != request->planned_gpu_resident_blocks ||
      staged_blocks != request->planned_gpu_staged_blocks ||
      resident_bytes != request->planned_gpu_resident_weight_bytes ||
      staged_bytes != request->planned_gpu_staged_weight_bytes) {
    return false;
  }
  out->planned_weight_descriptor_hash = descriptor_hash;
  return true;
}

int fail(NervaCudaHfDecodeSequenceResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  return -1;
}

cudaError_t cublas_to_cuda(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return cudaSuccess;
    case CUBLAS_STATUS_ALLOC_FAILED:
      return cudaErrorMemoryAllocation;
    case CUBLAS_STATUS_INVALID_VALUE:
      return cudaErrorInvalidValue;
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return cudaErrorInvalidDeviceFunction;
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return cudaErrorLaunchFailure;
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return cudaErrorNotSupported;
    default:
      return cudaErrorUnknown;
  }
}

#if NERVA_HAVE_CUDNN_FRONTEND
cudaError_t cudnn_to_cuda(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return cudaSuccess;
    case CUDNN_STATUS_ALLOC_FAILED:
      return cudaErrorMemoryAllocation;
    case CUDNN_STATUS_BAD_PARAM:
      return cudaErrorInvalidValue;
    case CUDNN_STATUS_NOT_SUPPORTED:
      return cudaErrorNotSupported;
    case CUDNN_STATUS_EXECUTION_FAILED:
      return cudaErrorLaunchFailure;
    default:
      return cudaErrorUnknown;
  }
}
#endif

cudaDataType_t encoded_cuda_type(uint32_t dtype) {
  return dtype == kDTypeBF16 ? CUDA_R_16BF : CUDA_R_16F;
}

cudaError_t configure_cublas(cublasHandle_t handle, cudaStream_t stream,
                             void *workspace, size_t workspace_bytes) {
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_to_cuda(status);
  }
  status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_to_cuda(status);
  }
  status = cublasSetWorkspace(handle, workspace, workspace_bytes);
  return cublas_to_cuda(status);
}

cudaError_t encoded_row_major_gemv_beta(
    cublasHandle_t handle, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t dtype, float beta,
    float *output) {
  if (rows == 0 || cols == 0 || rows > INT32_MAX || cols > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  const float alpha = 1.0f;
  const cudaDataType_t data_type = encoded_cuda_type(dtype);
  cublasStatus_t status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rows), 1,
      static_cast<int>(cols), &alpha, matrix, data_type, static_cast<int>(cols),
      input, data_type, static_cast<int>(cols), &beta, output, CUDA_R_32F,
      static_cast<int>(rows), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return cublas_to_cuda(status);
}

cudaError_t encoded_row_major_gemv(cublasHandle_t handle, const uint16_t *matrix,
                                   const uint16_t *input, uint32_t rows,
                                   uint32_t cols, uint32_t dtype,
                                   float *output) {
  return encoded_row_major_gemv_beta(handle, matrix, input, rows, cols, dtype,
                                     0.0f, output);
}

void destroy_lt_descriptors(cublasLtMatmulDesc_t op_desc,
                            cublasLtMatrixLayout_t a_desc,
                            cublasLtMatrixLayout_t b_desc,
                            cublasLtMatrixLayout_t c_desc,
                            cublasLtMatrixLayout_t d_desc) {
  if (d_desc != nullptr) cublasLtMatrixLayoutDestroy(d_desc);
  if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
  if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
  if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
  if (op_desc != nullptr) cublasLtMatmulDescDestroy(op_desc);
}

void destroy_lt_gemv_plan(LtGemvPlan *plan) {
  if (plan == nullptr) {
    return;
  }
  destroy_lt_descriptors(plan->op_desc, plan->a_desc, plan->b_desc,
                         plan->c_desc, plan->d_desc);
  *plan = LtGemvPlan{};
}

void destroy_lt_gemm_tokens_plan(LtGemmTokensPlan *plan) {
  if (plan == nullptr) {
    return;
  }
  destroy_lt_descriptors(plan->op_desc, plan->a_desc, plan->b_desc,
                         plan->c_desc, plan->d_desc);
  *plan = LtGemmTokensPlan{};
}

cudaError_t create_lt_gemv_descriptors(
    uint32_t rows, uint32_t cols, uint32_t dtype,
    cublasLtMatmulDesc_t *op_desc, cublasLtMatrixLayout_t *a_desc,
    cublasLtMatrixLayout_t *b_desc, cublasLtMatrixLayout_t *c_desc,
    cublasLtMatrixLayout_t *d_desc) {
  if (rows == 0 || cols == 0 || op_desc == nullptr || a_desc == nullptr ||
      b_desc == nullptr || c_desc == nullptr || d_desc == nullptr) {
    return cudaErrorInvalidValue;
  }
  const cudaDataType_t data_type = encoded_cuda_type(dtype);
  cublasStatus_t status =
      cublasLtMatmulDescCreate(op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t op_a = CUBLAS_OP_N;
  cublasOperation_t op_b = CUBLAS_OP_T;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        *op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        *op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(a_desc, data_type, 1, cols, cols);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(b_desc, data_type, rows, cols, cols);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(c_desc, CUDA_R_32F, 1, rows, rows);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(d_desc, CUDA_R_32F, 1, rows, rows);
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        *a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        *b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        *c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        *d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_lt_descriptors(*op_desc, *a_desc, *b_desc, *c_desc, *d_desc);
    *op_desc = nullptr;
    *a_desc = nullptr;
    *b_desc = nullptr;
    *c_desc = nullptr;
    *d_desc = nullptr;
  }
  return cublas_to_cuda(status);
}

cudaError_t create_lt_gemm_tokens_plan(LtGemmTokensPlan *plan, uint32_t rows,
                                       uint32_t cols, uint32_t tokens,
                                       uint32_t dtype) {
  if (plan == nullptr || rows == 0 || cols == 0 || tokens == 0 ||
      rows > INT32_MAX || cols > INT32_MAX || tokens > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  destroy_lt_gemm_tokens_plan(plan);
  const cudaDataType_t data_type = encoded_cuda_type(dtype);
  cublasStatus_t status =
      cublasLtMatmulDescCreate(&plan->op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t op_a = CUBLAS_OP_N;
  cublasOperation_t op_b = CUBLAS_OP_T;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(
        &plan->a_desc, data_type, tokens, cols, cols);
  if (status == CUBLAS_STATUS_SUCCESS)
    status =
        cublasLtMatrixLayoutCreate(&plan->b_desc, data_type, rows, cols, cols);
  if (status == CUBLAS_STATUS_SUCCESS)
    status =
        cublasLtMatrixLayoutCreate(&plan->c_desc, CUDA_R_32F, tokens, rows, rows);
  if (status == CUBLAS_STATUS_SUCCESS)
    status =
        cublasLtMatrixLayoutCreate(&plan->d_desc, CUDA_R_32F, tokens, rows, rows);
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        plan->a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        plan->b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        plan->c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        plan->d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_lt_gemm_tokens_plan(plan);
    return cublas_to_cuda(status);
  }
  plan->rows = rows;
  plan->cols = cols;
  plan->tokens = tokens;
  plan->dtype = dtype;
  plan->ready = true;
  return cudaSuccess;
}

cudaError_t launch_lt_gemm_tokens_plan(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const LtGemmTokensPlan *plan,
    const uint16_t *matrix, const uint16_t *input, float beta, float *output) {
  if (handle == nullptr || plan == nullptr || !plan->ready ||
      matrix == nullptr || input == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  const float alpha = 1.0f;
  const cublasStatus_t status = cublasLtMatmul(
      handle, plan->op_desc, &alpha, input, plan->a_desc, matrix, plan->b_desc,
      &beta, output, plan->c_desc, output, plan->d_desc, nullptr, workspace,
      workspace_bytes, stream);
  return cublas_to_cuda(status);
}

cudaError_t create_lt_gemv_plan(LtGemvPlan *plan, uint32_t rows,
                                uint32_t cols, uint32_t dtype) {
  if (plan == nullptr) {
    return cudaErrorInvalidValue;
  }
  destroy_lt_gemv_plan(plan);
  cudaError_t err = create_lt_gemv_descriptors(
      rows, cols, dtype, &plan->op_desc, &plan->a_desc, &plan->b_desc,
      &plan->c_desc, &plan->d_desc);
  if (err != cudaSuccess) {
    return err;
  }
  plan->rows = rows;
  plan->cols = cols;
  plan->dtype = dtype;
  plan->ready = true;
  return cudaSuccess;
}

const cublasLtMatmulAlgo_t *lt_gemv_algo(const LtGemvPlan *plan) {
  return plan != nullptr && plan->has_algo ? &plan->algo : nullptr;
}

cudaError_t launch_lt_gemv_plan(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const LtGemvPlan *plan, const uint16_t *matrix,
    const uint16_t *input, float beta, float *output,
    const cublasLtMatmulAlgo_t *algo) {
  if (handle == nullptr || plan == nullptr || !plan->ready ||
      matrix == nullptr || input == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  const float alpha = 1.0f;
  const cublasStatus_t status = cublasLtMatmul(
      handle, plan->op_desc, &alpha, input, plan->a_desc, matrix, plan->b_desc,
      &beta, output, plan->c_desc, output, plan->d_desc, algo, workspace,
      workspace_bytes, stream);
  return cublas_to_cuda(status);
}

cudaError_t encoded_row_major_gemv_lt(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t dtype, float beta, float *output) {
  if (handle == nullptr || rows == 0 || cols == 0) {
    return cudaErrorInvalidValue;
  }
  const float alpha = 1.0f;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cudaError_t err = create_lt_gemv_descriptors(
      rows, cols, dtype, &op_desc, &a_desc, &b_desc, &c_desc, &d_desc);
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  if (err == cudaSuccess)
    status = cublasLtMatmul(handle, op_desc, &alpha, input, a_desc, matrix,
                            b_desc, &beta, output, c_desc, output, d_desc,
                            nullptr, workspace, workspace_bytes, stream);
  destroy_lt_descriptors(op_desc, a_desc, b_desc, c_desc, d_desc);
  return err == cudaSuccess ? cublas_to_cuda(status) : err;
}

cudaError_t encoded_row_major_gemv_lt_planned(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const LtGemvPlan *plan, const uint16_t *matrix,
    const uint16_t *input, float beta, float *output) {
  if (plan == nullptr || !plan->ready) {
    return cudaErrorInvalidValue;
  }
  return launch_lt_gemv_plan(handle, stream, workspace, workspace_bytes, plan,
                             matrix, input, beta, output, lt_gemv_algo(plan));
}

cudaError_t encoded_row_major_gemv_planned(
    cublasHandle_t cublas, cublasLtHandle_t cublas_lt, cudaStream_t stream,
    void *workspace, size_t workspace_bytes, const LtGemvPlan *plan,
    const uint16_t *matrix, const uint16_t *input, float beta, float *output) {
  if (plan == nullptr || !plan->ready) {
    return cudaErrorInvalidValue;
  }
  if (plan->backend == kGemvBackendCublas) {
    return encoded_row_major_gemv_beta(cublas, matrix, input, plan->rows,
                                       plan->cols, plan->dtype, beta, output);
  }
  return encoded_row_major_gemv_lt_planned(cublas_lt, stream, workspace,
                                           workspace_bytes, plan, matrix,
                                           input, beta, output);
}

cudaError_t find_lt_gemv_heuristics(
    cublasLtHandle_t handle, const LtGemvPlan *plan,
    size_t workspace_bytes, cublasLtMatmulHeuristicResult_t *heuristics,
    uint32_t *heuristic_count) {
  if (handle == nullptr || plan == nullptr || !plan->ready ||
      heuristics == nullptr || heuristic_count == nullptr) {
    return cudaErrorInvalidValue;
  }
  *heuristic_count = 0;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasStatus_t status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_to_cuda(status);
  }
  status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes));
  int returned_count = 0;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoGetHeuristic(
        handle, plan->op_desc, plan->a_desc, plan->b_desc, plan->c_desc,
        plan->d_desc, preference, kLtGemvMaxHeuristics, heuristics,
        &returned_count);
  }
  cublasLtMatmulPreferenceDestroy(preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_to_cuda(status);
  }
  *heuristic_count = returned_count > 0
                         ? static_cast<uint32_t>(returned_count)
                         : 0;
  return cudaSuccess;
}

uint64_t cuda_event_elapsed_ns(cudaEvent_t start, cudaEvent_t stop) {
  float elapsed_ms = 0.0f;
  cudaError_t err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (err != cudaSuccess || elapsed_ms <= 0.0f) {
    return 0;
  }
  const uint64_t ns = static_cast<uint64_t>(elapsed_ms * 1000000.0f);
  return ns == 0 ? 1 : ns;
}

cudaError_t time_lt_gemv_candidate(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const LtGemvPlan *plan, const uint16_t *matrix,
    const uint16_t *input, float *output, const cublasLtMatmulAlgo_t *algo,
    uint64_t *avg_ns) {
  if (avg_ns == nullptr) {
    return cudaErrorInvalidValue;
  }
  *avg_ns = 0;
  for (uint32_t index = 0; index < kLtGemvAutotuneWarmups; ++index) {
    cudaError_t err = launch_lt_gemv_plan(
        handle, stream, workspace, workspace_bytes, plan, matrix, input, 0.0f,
        output, algo);
    if (err != cudaSuccess) return err;
  }
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return err;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  err = cudaEventCreate(&start);
  if (err == cudaSuccess) err = cudaEventCreate(&stop);
  if (err == cudaSuccess) err = cudaEventRecord(start, stream);
  for (uint32_t index = 0;
       err == cudaSuccess && index < kLtGemvAutotuneIterations; ++index) {
    err = launch_lt_gemv_plan(handle, stream, workspace, workspace_bytes, plan,
                              matrix, input, 0.0f, output, algo);
  }
  if (err == cudaSuccess) err = cudaEventRecord(stop, stream);
  if (err == cudaSuccess) err = cudaEventSynchronize(stop);
  if (err == cudaSuccess) {
    const uint64_t total_ns = cuda_event_elapsed_ns(start, stop);
    *avg_ns = total_ns / kLtGemvAutotuneIterations;
  }
  if (stop != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(stop);
    if (err == cudaSuccess) err = cleanup_err;
  }
  if (start != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(start);
    if (err == cudaSuccess) err = cleanup_err;
  }
  return err;
}

cudaError_t time_cublas_gemv_candidate(
    cublasHandle_t handle, cudaStream_t stream, const LtGemvPlan *plan,
    const uint16_t *matrix, const uint16_t *input, float *output,
    uint64_t *avg_ns) {
  if (avg_ns == nullptr || plan == nullptr || !plan->ready) {
    return cudaErrorInvalidValue;
  }
  *avg_ns = 0;
  for (uint32_t index = 0; index < kLtGemvAutotuneWarmups; ++index) {
    cudaError_t err = encoded_row_major_gemv_beta(
        handle, matrix, input, plan->rows, plan->cols, plan->dtype, 0.0f,
        output);
    if (err != cudaSuccess) return err;
  }
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return err;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  err = cudaEventCreate(&start);
  if (err == cudaSuccess) err = cudaEventCreate(&stop);
  if (err == cudaSuccess) err = cudaEventRecord(start, stream);
  for (uint32_t index = 0;
       err == cudaSuccess && index < kLtGemvAutotuneIterations; ++index) {
    err = encoded_row_major_gemv_beta(handle, matrix, input, plan->rows,
                                      plan->cols, plan->dtype, 0.0f, output);
  }
  if (err == cudaSuccess) err = cudaEventRecord(stop, stream);
  if (err == cudaSuccess) err = cudaEventSynchronize(stop);
  if (err == cudaSuccess) {
    const uint64_t total_ns = cuda_event_elapsed_ns(start, stop);
    *avg_ns = total_ns / kLtGemvAutotuneIterations;
  }
  if (stop != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(stop);
    if (err == cudaSuccess) err = cleanup_err;
  }
  if (start != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(start);
    if (err == cudaSuccess) err = cleanup_err;
  }
  return err;
}

cudaError_t time_lt_gemv_graph_candidate(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const LtGemvPlan *plan, const uint16_t *matrix,
    const uint16_t *input, float *output, const cublasLtMatmulAlgo_t *algo,
    uint64_t *avg_ns) {
  if (avg_ns == nullptr) {
    return cudaErrorInvalidValue;
  }
  *avg_ns = 0;
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return err;
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (err != cudaSuccess) {
    return err;
  }
  err = launch_lt_gemv_plan(handle, stream, workspace, workspace_bytes, plan,
                            matrix, input, 0.0f, output, algo);
  cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
  if (err != cudaSuccess) {
    if (end_err == cudaSuccess && graph != nullptr) cudaGraphDestroy(graph);
    return err;
  }
  if (end_err != cudaSuccess) {
    if (graph != nullptr) cudaGraphDestroy(graph);
    return end_err;
  }
  err = cudaGraphInstantiate(&graph_exec, graph, 0);
  if (err != cudaSuccess) {
    if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
    if (graph != nullptr) cudaGraphDestroy(graph);
    return err;
  }

  for (uint32_t index = 0; index < kLtGemvAutotuneWarmups; ++index) {
    err = cudaGraphLaunch(graph_exec, stream);
    if (err != cudaSuccess) break;
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (err == cudaSuccess) err = cudaEventCreate(&start);
  if (err == cudaSuccess) err = cudaEventCreate(&stop);
  if (err == cudaSuccess) err = cudaEventRecord(start, stream);
  for (uint32_t index = 0;
       err == cudaSuccess && index < kLtGemvAutotuneIterations; ++index) {
    err = cudaGraphLaunch(graph_exec, stream);
  }
  if (err == cudaSuccess) err = cudaEventRecord(stop, stream);
  if (err == cudaSuccess) err = cudaEventSynchronize(stop);
  if (err == cudaSuccess) {
    const uint64_t total_ns = cuda_event_elapsed_ns(start, stop);
    *avg_ns = total_ns / kLtGemvAutotuneIterations;
  }

  if (stop != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(stop);
    if (err == cudaSuccess) err = cleanup_err;
  }
  if (start != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(start);
    if (err == cudaSuccess) err = cleanup_err;
  }
  cudaError_t cleanup_err = cudaGraphExecDestroy(graph_exec);
  if (err == cudaSuccess && cleanup_err != cudaSuccess) err = cleanup_err;
  cleanup_err = cudaGraphDestroy(graph);
  if (err == cudaSuccess && cleanup_err != cudaSuccess) err = cleanup_err;
  return err;
}

cudaError_t time_cublas_gemv_graph_candidate(
    cublasHandle_t handle, cudaStream_t stream, const LtGemvPlan *plan,
    const uint16_t *matrix, const uint16_t *input, float *output,
    uint64_t *avg_ns) {
  if (avg_ns == nullptr || plan == nullptr || !plan->ready) {
    return cudaErrorInvalidValue;
  }
  *avg_ns = 0;
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return err;
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (err != cudaSuccess) {
    return err;
  }
  err = encoded_row_major_gemv_beta(handle, matrix, input, plan->rows,
                                    plan->cols, plan->dtype, 0.0f, output);
  cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
  if (err != cudaSuccess) {
    if (end_err == cudaSuccess && graph != nullptr) cudaGraphDestroy(graph);
    return err;
  }
  if (end_err != cudaSuccess) {
    if (graph != nullptr) cudaGraphDestroy(graph);
    return end_err;
  }
  err = cudaGraphInstantiate(&graph_exec, graph, 0);
  if (err != cudaSuccess) {
    if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
    if (graph != nullptr) cudaGraphDestroy(graph);
    return err;
  }

  for (uint32_t index = 0; index < kLtGemvAutotuneWarmups; ++index) {
    err = cudaGraphLaunch(graph_exec, stream);
    if (err != cudaSuccess) break;
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (err == cudaSuccess) err = cudaEventCreate(&start);
  if (err == cudaSuccess) err = cudaEventCreate(&stop);
  if (err == cudaSuccess) err = cudaEventRecord(start, stream);
  for (uint32_t index = 0;
       err == cudaSuccess && index < kLtGemvAutotuneIterations; ++index) {
    err = cudaGraphLaunch(graph_exec, stream);
  }
  if (err == cudaSuccess) err = cudaEventRecord(stop, stream);
  if (err == cudaSuccess) err = cudaEventSynchronize(stop);
  if (err == cudaSuccess) {
    const uint64_t total_ns = cuda_event_elapsed_ns(start, stop);
    *avg_ns = total_ns / kLtGemvAutotuneIterations;
  }

  if (stop != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(stop);
    if (err == cudaSuccess) err = cleanup_err;
  }
  if (start != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(start);
    if (err == cudaSuccess) err = cleanup_err;
  }
  cudaError_t cleanup_err = cudaGraphExecDestroy(graph_exec);
  if (err == cudaSuccess && cleanup_err != cudaSuccess) err = cleanup_err;
  cleanup_err = cudaGraphDestroy(graph);
  if (err == cudaSuccess && cleanup_err != cudaSuccess) err = cleanup_err;
  return err;
}

cudaError_t autotune_lt_gemv_plan(
    cublasHandle_t cublas, cublasLtHandle_t cublas_lt, cudaStream_t stream,
    void *workspace, size_t workspace_bytes, LtGemvPlan *plan,
    const uint16_t *matrix, const uint16_t *input, float *output) {
  if (plan == nullptr || !plan->ready) {
    return cudaErrorInvalidValue;
  }
  uint64_t best_avg_ns = 0;
  cudaError_t err = time_lt_gemv_candidate(
      cublas_lt, stream, workspace, workspace_bytes, plan, matrix, input,
      output, nullptr, &best_avg_ns);
  if (err != cudaSuccess) {
    return err;
  }
  plan->backend = kGemvBackendLt;
  plan->has_algo = false;
  plan->selected_heuristic = UINT32_MAX;
  plan->tuned_avg_ns = best_avg_ns;

  uint64_t cublas_avg_ns = 0;
  const cudaError_t cublas_err =
      time_cublas_gemv_candidate(cublas, stream, plan, matrix, input, output,
                                 &cublas_avg_ns);
  if (cublas_err == cudaSuccess && cublas_avg_ns != 0 &&
      (best_avg_ns == 0 || cublas_avg_ns < best_avg_ns)) {
    best_avg_ns = cublas_avg_ns;
    plan->backend = kGemvBackendCublas;
    plan->has_algo = false;
    plan->selected_heuristic = UINT32_MAX;
    plan->tuned_avg_ns = cublas_avg_ns;
  }

  cublasLtMatmulHeuristicResult_t heuristics[kLtGemvMaxHeuristics]{};
  uint32_t heuristic_count = 0;
  const cudaError_t heuristic_err = find_lt_gemv_heuristics(
      cublas_lt, plan, workspace_bytes, heuristics, &heuristic_count);
  if (heuristic_err != cudaSuccess) {
    return cudaSuccess;
  }
  plan->heuristic_count = heuristic_count;
  for (uint32_t index = 0; index < heuristic_count; ++index) {
    uint64_t avg_ns = 0;
    err = time_lt_gemv_candidate(
        cublas_lt, stream, workspace, workspace_bytes, plan, matrix, input,
        output, &heuristics[index].algo, &avg_ns);
    if (err != cudaSuccess || avg_ns == 0) {
      continue;
    }
    if (best_avg_ns == 0 || avg_ns < best_avg_ns) {
      best_avg_ns = avg_ns;
      plan->backend = kGemvBackendLt;
      plan->algo = heuristics[index].algo;
      plan->has_algo = true;
      plan->selected_heuristic = index;
      plan->tuned_avg_ns = avg_ns;
    }
  }
  uint64_t graph_best_avg_ns = 0;
  uint32_t graph_best_backend = plan->backend;
  bool graph_best_has_algo = plan->has_algo;
  uint32_t graph_best_heuristic = plan->selected_heuristic;
  cublasLtMatmulAlgo_t graph_best_algo = plan->algo;

  auto consider_graph_candidate = [&](uint32_t backend, bool has_algo,
                                      uint32_t heuristic_index,
                                      const cublasLtMatmulAlgo_t *algo,
                                      uint64_t avg_ns) {
    if (avg_ns == 0) {
      return;
    }
    if (graph_best_avg_ns == 0 || avg_ns < graph_best_avg_ns) {
      graph_best_avg_ns = avg_ns;
      graph_best_backend = backend;
      graph_best_has_algo = has_algo;
      graph_best_heuristic = heuristic_index;
      if (algo != nullptr) {
        graph_best_algo = *algo;
      }
    }
  };

  uint64_t graph_avg_ns = 0;
  cudaError_t graph_err = time_cublas_gemv_graph_candidate(
      cublas, stream, plan, matrix, input, output, &graph_avg_ns);
  if (graph_err == cudaSuccess) {
    consider_graph_candidate(kGemvBackendCublas, false, UINT32_MAX, nullptr,
                             graph_avg_ns);
  }
  graph_avg_ns = 0;
  graph_err = time_lt_gemv_graph_candidate(
      cublas_lt, stream, workspace, workspace_bytes, plan, matrix, input,
      output, nullptr, &graph_avg_ns);
  if (graph_err == cudaSuccess) {
    consider_graph_candidate(kGemvBackendLt, false, UINT32_MAX, nullptr,
                             graph_avg_ns);
  }
  for (uint32_t index = 0; index < heuristic_count; ++index) {
    graph_avg_ns = 0;
    graph_err = time_lt_gemv_graph_candidate(
        cublas_lt, stream, workspace, workspace_bytes, plan, matrix, input,
        output, &heuristics[index].algo, &graph_avg_ns);
    if (graph_err == cudaSuccess) {
      consider_graph_candidate(kGemvBackendLt, true, index,
                               &heuristics[index].algo, graph_avg_ns);
    }
  }
  if (graph_best_avg_ns != 0) {
    plan->backend = graph_best_backend;
    plan->has_algo = graph_best_has_algo;
    plan->selected_heuristic = graph_best_heuristic;
    plan->tuned_avg_ns = graph_best_avg_ns;
    if (graph_best_has_algo) {
      plan->algo = graph_best_algo;
    }
  }
  return cudaSuccess;
}

cudaError_t encoded_row_major_gemm_tokens(
    cublasHandle_t handle, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t tokens, uint32_t dtype, float beta,
    float *output) {
  if (rows == 0 || cols == 0 || tokens == 0 || rows > INT32_MAX ||
      cols > INT32_MAX || tokens > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  const float alpha = 1.0f;
  const cudaDataType_t data_type = encoded_cuda_type(dtype);
  cublasStatus_t status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rows),
      static_cast<int>(tokens), static_cast<int>(cols), &alpha, matrix,
      data_type, static_cast<int>(cols), input, data_type,
      static_cast<int>(cols), &beta, output, CUDA_R_32F,
      static_cast<int>(rows), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return cublas_to_cuda(status);
}

cudaError_t encoded_row_major_gemm_tokens_lt(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t tokens, uint32_t dtype, float beta,
    float *output) {
  if (handle == nullptr || rows == 0 || cols == 0 || tokens == 0 ||
      rows > INT32_MAX || cols > INT32_MAX || tokens > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  const cudaDataType_t data_type = encoded_cuda_type(dtype);
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasStatus_t status =
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t op_a = CUBLAS_OP_N;
  cublasOperation_t op_b = CUBLAS_OP_T;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(
        &a_desc, data_type, tokens, cols, cols);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&b_desc, data_type, rows, cols, cols);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, tokens, rows, rows);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_32F, tokens, rows, rows);
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS) {
    const float alpha = 1.0f;
    status = cublasLtMatmul(handle, op_desc, &alpha, input, a_desc, matrix,
                            b_desc, &beta, output, c_desc, output, d_desc,
                            nullptr, workspace, workspace_bytes, stream);
  }
  destroy_lt_descriptors(op_desc, a_desc, b_desc, c_desc, d_desc);
  return cublas_to_cuda(status);
}

cudaError_t encoded_row_major_gemm_tokens_best(
    cublasHandle_t cublas, cublasLtHandle_t cublas_lt, cudaStream_t stream,
    void *workspace, size_t workspace_bytes, const uint16_t *matrix,
    const uint16_t *input, uint32_t rows, uint32_t cols, uint32_t tokens,
    uint32_t dtype, float beta, float *output) {
  cudaError_t err = encoded_row_major_gemm_tokens_lt(
      cublas_lt, stream, workspace, workspace_bytes, matrix, input, rows, cols,
      tokens, dtype, beta, output);
  if (err == cudaSuccess) {
    return err;
  }
  return encoded_row_major_gemm_tokens(cublas, matrix, input, rows, cols,
                                       tokens, dtype, beta, output);
}

cudaError_t final_head_gemv(cublasHandle_t handle, uint16_t *arena,
                            SequenceArenaLayout arena_layout, uint32_t dtype,
                            uint32_t hidden, uint32_t vocab_size,
                            float *device_logits) {
  return encoded_row_major_gemv(handle, arena + arena_layout.lm_head,
                                arena + arena_layout.input, vocab_size, hidden,
                                dtype, device_logits);
}

cudaError_t warm_cublas_gemv(cublasHandle_t handle, uint16_t *arena,
                             SequenceArenaLayout arena_layout, uint32_t dtype,
                             float *scratch, cudaStream_t stream) {
  cudaError_t err = cudaMemsetAsync(arena + arena_layout.input, 0,
                                    sizeof(uint16_t), stream);
  if (err != cudaSuccess) {
    return err;
  }
  return encoded_row_major_gemv(handle, arena + arena_layout.lm_head,
                                arena + arena_layout.input, 1, 1, dtype,
                                scratch);
}

bool should_pack_cublas_weights(uint32_t hidden, uint32_t attention_hidden) {
  return hidden >= 128 && attention_hidden == hidden;
}

PackedProjectionShape packed_projection_shape(uint64_t hidden,
                                              uint64_t attention_hidden,
                                              uint64_t kv_hidden,
                                              uint64_t intermediate) {
  PackedProjectionShape shape{};
  shape.qkv_rows = attention_hidden + kv_hidden * 2;
  shape.gate_up_rows = intermediate * 2;
  shape.qkv_elements_per_layer = shape.qkv_rows * hidden;
  shape.gate_up_elements_per_layer = shape.gate_up_rows * hidden;
  return shape;
}

void pack_layer(SequenceLayerLayout &layout, uint64_t &cursor,
                const NervaCudaHfDecodeChainLayer &layer, uint64_t hidden,
                uint64_t attention_hidden, uint64_t kv_hidden, uint64_t head_dim,
                uint64_t intermediate) {
  layout.rms_attn = push(cursor, hidden);
  layout.w_q = push(cursor, attention_hidden * hidden);
  layout.q_norm = push_optional(cursor, head_dim, layer.q_norm_weight);
  layout.w_k = push(cursor, kv_hidden * hidden);
  layout.k_norm = push_optional(cursor, head_dim, layer.k_norm_weight);
  layout.w_v = push(cursor, kv_hidden * hidden);
  layout.w_o = push(cursor, hidden * attention_hidden);
  layout.rms_mlp = push(cursor, hidden);
  layout.w_gate = push(cursor, intermediate * hidden);
  layout.w_up = push(cursor, intermediate * hidden);
  layout.w_down = push(cursor, hidden * intermediate);
  layout.q_bias = push_optional(cursor, attention_hidden, layer.q_bias);
  layout.k_bias = push_optional(cursor, kv_hidden, layer.k_bias);
  layout.v_bias = push_optional(cursor, kv_hidden, layer.v_bias);
  layout.o_bias = push_optional(cursor, hidden, layer.o_bias);
}

void copy_optional(uint16_t *arena, uint64_t offset, const uint16_t *src, uint64_t elements) {
  if (src != nullptr) {
    memcpy(arena + offset, src, elements * sizeof(uint16_t));
  }
}

void copy_layer(uint16_t *arena, const SequenceLayerLayout &layout,
                const NervaCudaHfDecodeChainLayer &layer, uint64_t hidden,
                uint64_t attention_hidden, uint64_t kv_hidden, uint64_t head_dim,
                uint64_t intermediate) {
  memcpy(arena + layout.rms_attn, layer.rms_attn_weight, hidden * sizeof(uint16_t));
  memcpy(arena + layout.rms_mlp, layer.rms_mlp_weight, hidden * sizeof(uint16_t));
  memcpy(arena + layout.w_q, layer.w_q, attention_hidden * hidden * sizeof(uint16_t));
  copy_optional(arena, layout.q_norm, layer.q_norm_weight, head_dim);
  memcpy(arena + layout.w_k, layer.w_k, kv_hidden * hidden * sizeof(uint16_t));
  copy_optional(arena, layout.k_norm, layer.k_norm_weight, head_dim);
  memcpy(arena + layout.w_v, layer.w_v, kv_hidden * hidden * sizeof(uint16_t));
  memcpy(arena + layout.w_o, layer.w_o, hidden * attention_hidden * sizeof(uint16_t));
  copy_optional(arena, layout.q_bias, layer.q_bias, attention_hidden);
  copy_optional(arena, layout.k_bias, layer.k_bias, kv_hidden);
  copy_optional(arena, layout.v_bias, layer.v_bias, kv_hidden);
  copy_optional(arena, layout.o_bias, layer.o_bias, hidden);
  memcpy(arena + layout.w_gate, layer.w_gate, intermediate * hidden * sizeof(uint16_t));
  memcpy(arena + layout.w_up, layer.w_up, intermediate * hidden * sizeof(uint16_t));
  memcpy(arena + layout.w_down, layer.w_down, hidden * intermediate * sizeof(uint16_t));
}

bool descriptor_destination_bytes(
    const NervaCudaHfDecodeSequenceWeightBlock &descriptor,
    uint64_t arena_bytes, uint64_t embedding_bytes, uint64_t scratch_gap_bytes,
    uint64_t *destination_bytes) {
  if (descriptor.offset_bytes % sizeof(uint16_t) != 0 ||
      descriptor.bytes % sizeof(uint16_t) != 0) {
    return false;
  }
  uint64_t translated = descriptor.offset_bytes;
  if (translated >= embedding_bytes) {
    translated += scratch_gap_bytes;
  }
  if (translated > arena_bytes || descriptor.bytes > arena_bytes - translated) {
    return false;
  }
  *destination_bytes = translated;
  return true;
}

struct NativeLoadProgress {
  std::chrono::steady_clock::time_point start;
  uint32_t last_percent;
};

void report_native_load_progress(uint64_t done, uint64_t total,
                                 NativeLoadProgress *progress) {
  const char *mode = getenv("NERVA_NATIVE_LOAD_PROGRESS");
  if (mode != nullptr && strcmp(mode, "quiet") == 0) {
    return;
  }
  if (total == 0 || progress == nullptr) {
    return;
  }
  const uint32_t percent =
      done >= total ? 100u : static_cast<uint32_t>((done * 100u) / total);
  const uint32_t displayed_percent =
      percent >= 100u ? 100u : (percent / 5u) * 5u;
  if (displayed_percent == progress->last_percent) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  const double elapsed_s =
      std::chrono::duration<double>(now - progress->start).count();
  const double done_gb = static_cast<double>(done) / 1000000000.0;
  const double total_gb = static_cast<double>(total) / 1000000000.0;
  const double gb_s = elapsed_s > 0.0 ? done_gb / elapsed_s : 0.0;
  if (mode != nullptr && strcmp(mode, "color") == 0) {
    fprintf(stderr,
            "\x1b[2m[nerva-load]\x1b[0m "
            "\x1b[38;2;255;106;42mweights H2D\x1b[0m "
            "\x1b[38;2;112;223;158m%3u%%\x1b[0m  "
            "%.2f/%.2f GB  \x1b[38;2;87;190;255m%.2f GB/s\x1b[0m\n",
            displayed_percent, done_gb, total_gb, gb_s);
  } else if (mode != nullptr && strcmp(mode, "ansi") == 0) {
    fprintf(stderr,
            "\x1b[2m[nerva-load]\x1b[0m "
            "\x1b[93mweights H2D\x1b[0m "
            "\x1b[92m%3u%%\x1b[0m  "
            "%.2f/%.2f GB  \x1b[96m%.2f GB/s\x1b[0m\n",
            displayed_percent, done_gb, total_gb, gb_s);
  } else {
    fprintf(stderr,
            "[nerva-load] weights H2D %3u%%  %.2f/%.2f GB  %.2f GB/s\n",
            displayed_percent, done_gb, total_gb, gb_s);
  }
  fflush(stderr);
  progress->last_percent = displayed_percent;
}

cudaError_t copy_file_descriptor_to_device(
    uint16_t *device_destination, uint16_t *staging, uint64_t staging_bytes,
    const NervaCudaHfDecodeSequenceWeightBlock &descriptor, cudaStream_t stream,
    uint64_t *setup_sync_calls, uint64_t *progress_done,
    uint64_t progress_total, NativeLoadProgress *progress) {
  if (device_destination == nullptr || staging == nullptr || staging_bytes == 0 ||
      staging_bytes % sizeof(uint16_t) != 0 ||
      !descriptor_has_file_source(descriptor)) {
    return cudaErrorInvalidValue;
  }
  std::string path(descriptor.source_file, descriptor.source_file_len);
  FILE *file = fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (fseek(file, static_cast<long>(descriptor.file_offset_begin), SEEK_SET) != 0) {
    fclose(file);
    return cudaErrorInvalidValue;
  }
  uint64_t remaining = descriptor.bytes;
  uint64_t destination_offset_elements = 0;
  while (remaining != 0) {
    const uint64_t chunk_bytes = std::min(remaining, staging_bytes);
    const size_t read = fread(staging, 1, static_cast<size_t>(chunk_bytes), file);
    if (read != static_cast<size_t>(chunk_bytes)) {
      fclose(file);
      return cudaErrorInvalidValue;
    }
    cudaError_t err = cudaMemcpyAsync(
        device_destination + destination_offset_elements, staging, chunk_bytes,
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      fclose(file);
      return err;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      fclose(file);
      return err;
    }
    if (setup_sync_calls != nullptr) {
      *setup_sync_calls += 1;
    }
    remaining -= chunk_bytes;
    destination_offset_elements += chunk_bytes / sizeof(uint16_t);
    if (progress_done != nullptr) {
      *progress_done += chunk_bytes;
      report_native_load_progress(*progress_done, progress_total, progress);
    }
  }
  fclose(file);
  return cudaSuccess;
}

template <typename Request, typename Result>
cudaError_t copy_weight_descriptors_to_device(
    uint16_t *device_arena, uint16_t *staging, uint64_t staging_bytes,
    const Request *request, uint64_t arena_bytes,
    uint64_t embedding_bytes, uint64_t scratch_gap_bytes, cudaStream_t stream,
    Result *out, uint64_t *setup_sync_calls) {
  uint64_t progress_done = 0;
  NativeLoadProgress progress = {std::chrono::steady_clock::now(), UINT32_MAX};
  const bool report_progress = descriptors_require_file_staging(request);
  if (report_progress) {
    report_native_load_progress(0, request->planned_weight_bytes, &progress);
  }
  for (uint32_t index = 0; index < request->planned_weight_descriptor_count; ++index) {
    const auto &descriptor = request->planned_weight_descriptors[index];
    uint64_t destination_bytes = 0;
    if (!descriptor_destination_bytes(descriptor, arena_bytes, embedding_bytes,
                                      scratch_gap_bytes, &destination_bytes)) {
      return cudaErrorInvalidValue;
    }
    uint16_t *destination = device_arena + destination_bytes / sizeof(uint16_t);
    if (descriptor_has_file_source(descriptor)) {
      cudaError_t err = copy_file_descriptor_to_device(
          destination, staging, staging_bytes, descriptor, stream, setup_sync_calls,
          &progress_done, request->planned_weight_bytes,
          &progress);
      if (err != cudaSuccess) {
        return err;
      }
    } else if (descriptor_has_memory_source(descriptor)) {
      cudaError_t err = cudaMemcpyAsync(destination, descriptor.host_source,
                                        descriptor.bytes, cudaMemcpyHostToDevice,
                                        stream);
      if (err != cudaSuccess) {
        return err;
      }
      if (report_progress) {
        progress_done += descriptor.bytes;
        report_native_load_progress(progress_done, request->planned_weight_bytes,
                                    &progress);
      }
    } else {
      return cudaErrorInvalidValue;
    }
    out->h2d_bytes += descriptor.bytes;
    if (descriptor.strategy == kWeightStrategyGpuResident) {
      out->descriptor_gpu_resident_h2d_bytes += descriptor.bytes;
    } else if (descriptor.strategy == kWeightStrategyGpuStaged) {
      out->descriptor_gpu_staged_h2d_bytes += descriptor.bytes;
    }
  }
  if (report_progress) {
    report_native_load_progress(request->planned_weight_bytes,
                                request->planned_weight_bytes,
                                &progress);
  }
  return cudaSuccess;
}

uint32_t observed_count_for(uint32_t steps, uint32_t prompt_token_count,
                            uint32_t has_eos_token, uint32_t eos_token,
                            const NervaCudaSyntheticTokenSlot *slots) {
  uint32_t count = steps;
  if (has_eos_token == 0) {
    return count;
  }
  const uint32_t output_start = prompt_token_count - 1u;
  for (uint32_t index = 0; index < steps; ++index) {
    if (slots[output_start + index].token == eos_token) {
      count = index + 1;
      break;
    }
  }
  return count;
}

uint32_t observed_count(const NervaCudaHfDecodeSequenceRequest *request,
                        const NervaCudaSyntheticTokenSlot *slots) {
  return observed_count_for(request->steps, request->prompt_token_count,
                            request->has_eos_token, request->eos_token, slots);
}

uint64_t saturating_mul_profile_value(uint64_t value, uint64_t scale) {
  if (value != 0 && scale > UINT64_MAX / value) {
    return UINT64_MAX;
  }
  return value * scale;
}

void scale_profile_counters(NervaCudaHfDecodeSequenceResult *out,
                            uint64_t scale) {
  if (out == nullptr || scale <= 1) {
    return;
  }
  out->projection_ns = saturating_mul_profile_value(out->projection_ns, scale);
  out->qkv_projection_ns =
      saturating_mul_profile_value(out->qkv_projection_ns, scale);
  out->attention_output_projection_ns =
      saturating_mul_profile_value(out->attention_output_projection_ns, scale);
  out->gate_up_projection_ns =
      saturating_mul_profile_value(out->gate_up_projection_ns, scale);
  out->down_projection_ns =
      saturating_mul_profile_value(out->down_projection_ns, scale);
  out->lm_head_projection_ns =
      saturating_mul_profile_value(out->lm_head_projection_ns, scale);
  out->attention_ns = saturating_mul_profile_value(out->attention_ns, scale);
  out->mlp_ns = saturating_mul_profile_value(out->mlp_ns, scale);
  out->norm_ns = saturating_mul_profile_value(out->norm_ns, scale);
  out->sampling_ns = saturating_mul_profile_value(out->sampling_ns, scale);
}

#if NERVA_HAVE_CUDNN_FRONTEND
struct CudnnPrefillSdpaPlan {
  std::unique_ptr<cudnn_frontend::graph::Graph> graph;
  uint32_t seq_tokens = 0;
  uint32_t heads = 0;
  uint32_t kv_heads = 0;
  uint32_t head_dim = 0;
  uint64_t rows = 0;
  size_t workspace_bytes = 0;
};

struct CudnnDecodeSdpaPlan {
  std::unique_ptr<cudnn_frontend::graph::Graph> graph;
  uint32_t max_context_tokens = 0;
  uint32_t kv_token_capacity = 0;
  uint32_t heads = 0;
  uint32_t kv_heads = 0;
  uint32_t head_dim = 0;
  size_t workspace_bytes = 0;
};
#endif

}  // namespace

struct NervaCudaHfDecodeSequenceSession {
  uint32_t dtype = 0;
  uint32_t hidden = 0;
  uint32_t heads = 0;
  uint32_t kv_heads = 0;
  uint32_t head_dim = 0;
  uint32_t head_threads = kHeadThreadsMax;
  uint32_t intermediate = 0;
  uint32_t vocab_size = 0;
  uint32_t layer_count = 0;
  uint32_t max_context_tokens = 0;
  uint32_t kv_block_count = 0;
  uint32_t kv_token_capacity = 0;
  uint32_t prefill_chunk_tokens = 0;
  uint32_t detailed_profile = 0;
  float rms_eps = 0.0f;
  float rope_theta = 0.0f;
  SequenceArenaLayout arena_layout{};
  uint64_t arena_bytes = 0;
  uint64_t resident_weight_bytes = 0;
  uint64_t layout_bytes = 0;
  uint64_t scratch_bytes = 0;
  uint64_t projection_input_bytes = 0;
  uint64_t prefill_hidden_bytes = 0;
  uint64_t prefill_norm_bytes = 0;
  uint64_t prefill_qkv_bytes = 0;
  uint64_t prefill_qkv_encoded_bytes = 0;
  uint64_t prefill_attn_bytes = 0;
  uint64_t prefill_o_bytes = 0;
  uint64_t prefill_gate_up_bytes = 0;
  uint64_t prefill_ff_bytes = 0;
  uint64_t prefill_down_bytes = 0;
  uint64_t decode_attention_values_bytes = 0;
  uint64_t decode_attention_stats_bytes = 0;
  uint32_t decode_attention_max_chunks = 0;
  uint64_t decode_q_bytes = 0;
  uint64_t decode_seq_len_bytes = 0;
  uint64_t packed_qkv_bytes = 0;
  uint64_t packed_gate_up_bytes = 0;
  uint64_t kv_bytes = 0;
  uint64_t kv_block_table_bytes = 0;
  uint64_t slots_bytes = 0;
  uint64_t prompt_bytes = 0;
  uint64_t h2d_bytes = 0;
  uint64_t load_staging_bytes = 0;
  uint64_t setup_sync_calls = 0;
  uint64_t descriptor_gpu_resident_h2d_bytes = 0;
  uint64_t descriptor_gpu_staged_h2d_bytes = 0;
  uint32_t planned_weight_blocks = 0;
  uint32_t planned_gpu_resident_blocks = 0;
  uint32_t planned_gpu_staged_blocks = 0;
  uint64_t planned_weight_bytes = 0;
  uint64_t planned_gpu_resident_weight_bytes = 0;
  uint64_t planned_gpu_staged_weight_bytes = 0;
  uint32_t planned_weight_descriptor_count = 0;
  uint64_t planned_weight_descriptor_hash = 0;
  std::vector<SequenceLayerLayout> host_layouts;
  uint16_t *device_arena = nullptr;
  SequenceLayerLayout *device_layouts = nullptr;
  float *device_scratch = nullptr;
  uint16_t *device_projection_input = nullptr;
  uint16_t *device_prefill_hidden_a = nullptr;
  uint16_t *device_prefill_hidden_b = nullptr;
  uint16_t *device_prefill_norm = nullptr;
  float *device_prefill_qkv = nullptr;
  uint16_t *device_prefill_qkv_encoded = nullptr;
  uint16_t *device_prefill_attn = nullptr;
  float *device_prefill_o = nullptr;
  float *device_prefill_gate_up = nullptr;
  uint16_t *device_prefill_ff = nullptr;
  float *device_prefill_down = nullptr;
  float *device_decode_attention_values = nullptr;
  float *device_decode_attention_m = nullptr;
  float *device_decode_attention_l = nullptr;
  uint16_t *device_decode_q = nullptr;
  int32_t *device_decode_seq_len_q = nullptr;
  int32_t *device_decode_seq_len_kv = nullptr;
  uint16_t *device_qkv_packed = nullptr;
  uint16_t *device_gate_up_packed = nullptr;
  uint16_t *device_kv_keys = nullptr;
  uint16_t *device_kv_values = nullptr;
  uint32_t *device_kv_block_table = nullptr;
  uint32_t *device_prompt_tokens = nullptr;
  NervaCudaSyntheticTokenSlot *host_slots = nullptr;
  NervaCudaSyntheticTokenSlot *device_slots = nullptr;
  uint32_t *device_step = nullptr;
  void *cublas_workspace = nullptr;
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas = nullptr;
  cublasLtHandle_t cublas_lt = nullptr;
#if NERVA_HAVE_CUDNN_FRONTEND
  cudnnHandle_t cudnn = nullptr;
  CudnnPrefillSdpaPlan *cudnn_prefill_sdpa = nullptr;
  uint32_t cudnn_prefill_sdpa_disabled = 0;
  CudnnDecodeSdpaPlan *cudnn_decode_sdpa = nullptr;
  uint32_t cudnn_decode_sdpa_disabled = 0;
#endif
  LtGemvPlan qkv_plan;
  LtGemvPlan attention_output_plan;
  LtGemvPlan gate_up_plan;
  LtGemvPlan down_plan;
  LtGemvPlan lm_head_plan;
  std::vector<LtGemmTokensPlan> projection_block_plans;
  cudaEvent_t device_start = nullptr;
  cudaEvent_t device_stop = nullptr;
  cudaEvent_t profile_start = nullptr;
  cudaEvent_t profile_stop = nullptr;
  cudaGraph_t cached_graph = nullptr;
  cudaGraphExec_t cached_graph_exec = nullptr;
  uint32_t cached_context_steps = 0;
  uint32_t cached_prompt_token_count = 0;
  uint32_t cached_has_eos_token = 0;
  uint32_t cached_eos_token = 0;
  uint32_t cached_attention_chunks = 0;
  uint64_t cached_graph_nodes = 0;
  uint64_t cached_projection_ns = 0;
  uint64_t cached_qkv_projection_ns = 0;
  uint64_t cached_attention_output_projection_ns = 0;
  uint64_t cached_gate_up_projection_ns = 0;
  uint64_t cached_down_projection_ns = 0;
  uint64_t cached_lm_head_projection_ns = 0;
  uint64_t cached_attention_ns = 0;
  uint64_t cached_mlp_ns = 0;
  uint64_t cached_norm_ns = 0;
  uint64_t cached_sampling_ns = 0;
  uint64_t pending_prefill_kernel_launches = 0;
  uint64_t pending_prefill_device_elapsed_ns = 0;
  uint64_t pending_prefill_sync_calls = 0;
  uint64_t pending_prefill_graph_replays = 0;
  uint64_t pending_prefill_graph_launches = 0;
  uint64_t pending_prefill_graph_nodes = 0;
  uint32_t pending_prefill_available = 0;
  uint32_t active_prompt_token_count = 0;
  uint32_t active_has_eos_token = 0;
  uint32_t active_eos_token = 0;
  uint32_t active_seed_token = 0;
  uint32_t active_observed_tokens = 0;
  uint32_t active_cursor = 0;
  bool active_started = false;
  bool active_finished = false;
};

namespace {

void free_session_fields(NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr) {
    return;
  }
  if (session->cached_graph_exec != nullptr) {
    cudaGraphExecDestroy(session->cached_graph_exec);
  }
  if (session->cached_graph != nullptr) {
    cudaGraphDestroy(session->cached_graph);
  }
#if NERVA_HAVE_CUDNN_FRONTEND
  delete session->cudnn_prefill_sdpa;
  session->cudnn_prefill_sdpa = nullptr;
  delete session->cudnn_decode_sdpa;
  session->cudnn_decode_sdpa = nullptr;
  if (session->cudnn != nullptr) cudnnDestroy(session->cudnn);
#endif
  if (session->profile_stop != nullptr) cudaEventDestroy(session->profile_stop);
  if (session->profile_start != nullptr) cudaEventDestroy(session->profile_start);
  if (session->device_stop != nullptr) cudaEventDestroy(session->device_stop);
  if (session->device_start != nullptr) cudaEventDestroy(session->device_start);
  for (LtGemmTokensPlan &plan : session->projection_block_plans) {
    destroy_lt_gemm_tokens_plan(&plan);
  }
  session->projection_block_plans.clear();
  destroy_lt_gemv_plan(&session->lm_head_plan);
  destroy_lt_gemv_plan(&session->down_plan);
  destroy_lt_gemv_plan(&session->gate_up_plan);
  destroy_lt_gemv_plan(&session->attention_output_plan);
  destroy_lt_gemv_plan(&session->qkv_plan);
  if (session->cublas_lt != nullptr) cublasLtDestroy(session->cublas_lt);
  if (session->cublas != nullptr) cublasDestroy(session->cublas);
  if (session->stream != nullptr) cudaStreamDestroy(session->stream);
  cudaFree(session->cublas_workspace);
  cudaFree(session->device_step);
  cudaFree(session->device_slots);
  cudaFree(session->device_prompt_tokens);
  cudaFree(session->device_kv_block_table);
  cudaFree(session->device_kv_values);
  cudaFree(session->device_kv_keys);
  cudaFree(session->device_gate_up_packed);
  cudaFree(session->device_qkv_packed);
  cudaFree(session->device_prefill_down);
  cudaFree(session->device_prefill_ff);
  cudaFree(session->device_prefill_gate_up);
  cudaFree(session->device_prefill_o);
  cudaFree(session->device_prefill_attn);
  cudaFree(session->device_prefill_qkv_encoded);
  cudaFree(session->device_prefill_qkv);
  cudaFree(session->device_prefill_norm);
  cudaFree(session->device_prefill_hidden_b);
  cudaFree(session->device_prefill_hidden_a);
  cudaFree(session->device_decode_attention_l);
  cudaFree(session->device_decode_attention_m);
  cudaFree(session->device_decode_attention_values);
  cudaFree(session->device_decode_seq_len_kv);
  cudaFree(session->device_decode_seq_len_q);
  cudaFree(session->device_decode_q);
  cudaFree(session->device_projection_input);
  cudaFree(session->device_scratch);
  cudaFree(session->device_layouts);
  cudaFree(session->device_arena);
  cudaFreeHost(session->host_slots);
}

void reset_session_graph(NervaCudaHfDecodeSequenceSession *session) {
  if (session->cached_graph_exec != nullptr) {
    cudaGraphExecDestroy(session->cached_graph_exec);
    session->cached_graph_exec = nullptr;
  }
  if (session->cached_graph != nullptr) {
    cudaGraphDestroy(session->cached_graph);
    session->cached_graph = nullptr;
  }
  session->cached_context_steps = 0;
  session->cached_prompt_token_count = 0;
  session->cached_has_eos_token = 0;
  session->cached_eos_token = 0;
  session->cached_attention_chunks = 0;
  session->cached_graph_nodes = 0;
  session->cached_projection_ns = 0;
  session->cached_qkv_projection_ns = 0;
  session->cached_attention_output_projection_ns = 0;
  session->cached_gate_up_projection_ns = 0;
  session->cached_down_projection_ns = 0;
  session->cached_lm_head_projection_ns = 0;
  session->cached_attention_ns = 0;
  session->cached_mlp_ns = 0;
  session->cached_norm_ns = 0;
  session->cached_sampling_ns = 0;
}

uint64_t session_device_footprint(
    const NervaCudaHfDecodeSequenceSession *session) {
  return session->arena_bytes + session->layout_bytes + session->scratch_bytes +
         session->projection_input_bytes + session->prefill_hidden_bytes * 2 +
         session->prefill_norm_bytes + session->prefill_qkv_bytes +
         session->prefill_qkv_encoded_bytes +
         session->prefill_attn_bytes + session->prefill_o_bytes +
         session->prefill_gate_up_bytes + session->prefill_ff_bytes +
         session->prefill_down_bytes + session->decode_attention_values_bytes +
         session->decode_attention_stats_bytes * 2 + session->decode_q_bytes +
         session->decode_seq_len_bytes +
         session->packed_qkv_bytes + session->packed_gate_up_bytes + session->kv_bytes +
         session->kv_block_table_bytes +
         session->prompt_bytes + session->slots_bytes + sizeof(uint32_t) +
         kCublasWorkspaceBytes;
}

uint64_t session_fixed_footprint_without_prefill_chunk(
    const NervaCudaHfDecodeSequenceSession *session) {
  return session->arena_bytes + session->layout_bytes + session->scratch_bytes +
         session->projection_input_bytes + session->prefill_hidden_bytes * 2 +
         session->decode_attention_values_bytes +
         session->decode_attention_stats_bytes * 2 + session->decode_q_bytes +
         session->decode_seq_len_bytes +
         session->packed_qkv_bytes + session->packed_gate_up_bytes + session->kv_bytes +
         session->kv_block_table_bytes +
         session->prompt_bytes + session->slots_bytes + sizeof(uint32_t) +
         kCublasWorkspaceBytes;
}

uint64_t sat_add_u64(uint64_t lhs, uint64_t rhs) {
  if (UINT64_MAX - lhs < rhs) return UINT64_MAX;
  return lhs + rhs;
}

uint64_t sat_mul_u64(uint64_t lhs, uint64_t rhs) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs) return UINT64_MAX;
  return lhs * rhs;
}

uint64_t prefill_chunk_scratch_bytes(uint64_t chunk_tokens,
                                     uint64_t projection_input_elements,
                                     uint64_t prefill_qkv_rows,
                                     uint64_t attention_hidden,
                                     uint64_t hidden,
                                     uint64_t prefill_gate_up_rows,
                                     uint64_t intermediate) {
  uint64_t total = 0;
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(projection_input_elements, chunk_tokens), sizeof(uint16_t)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(prefill_qkv_rows, chunk_tokens), sizeof(float)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(prefill_qkv_rows, chunk_tokens), sizeof(uint16_t)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(attention_hidden, chunk_tokens), sizeof(uint16_t)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(hidden, chunk_tokens), sizeof(float)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(prefill_gate_up_rows, chunk_tokens), sizeof(float)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(intermediate, chunk_tokens), sizeof(uint16_t)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(hidden, chunk_tokens), sizeof(float)));
  return total;
}

uint32_t tune_prefill_chunk_tokens(uint64_t max_context_tokens,
                                   uint64_t fixed_device_bytes,
                                   uint64_t projection_input_elements,
                                   uint64_t prefill_qkv_rows,
                                   uint64_t attention_hidden,
                                   uint64_t hidden,
                                   uint64_t prefill_gate_up_rows,
                                   uint64_t intermediate,
                                   uint64_t free_device_bytes) {
  if (max_context_tokens == 0) return 0;
  const uint64_t base =
      std::min<uint64_t>(kPrefillChunkBaseTokens, max_context_tokens);
  const uint64_t max_target =
      std::min<uint64_t>(kPrefillChunkMaxTokens, max_context_tokens);
  const uint64_t min_chunk = std::min<uint64_t>(base, max_context_tokens);
  if (free_device_bytes == 0) {
    return static_cast<uint32_t>(base);
  }
  const uint64_t budget =
      free_device_bytes > kPrefillAutotuneSafetyBytes
          ? free_device_bytes - kPrefillAutotuneSafetyBytes
          : free_device_bytes;
  auto fits = [&](uint64_t candidate) {
    const uint64_t footprint = sat_add_u64(
        fixed_device_bytes,
        prefill_chunk_scratch_bytes(candidate, projection_input_elements,
                                    prefill_qkv_rows, attention_hidden, hidden,
                                    prefill_gate_up_rows, intermediate));
    return footprint <= budget;
  };
  uint64_t chunk = base;
  while (chunk > min_chunk && !fits(chunk)) {
    chunk = std::max<uint64_t>(min_chunk, chunk / 2);
  }
  while (chunk < max_target) {
    const uint64_t next = std::min<uint64_t>(max_target, chunk * 2);
    if (next == chunk || !fits(next)) break;
    chunk = next;
  }
  return static_cast<uint32_t>(chunk);
}

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
  return divisor == 0 ? 0 : (value + divisor - 1u) / divisor;
}

uint32_t next_pow2_at_least(uint32_t value, uint32_t minimum,
                            uint32_t maximum) {
  uint32_t out = minimum;
  while (out < value && out < maximum) {
    out <<= 1;
  }
  return out > maximum ? maximum : out;
}

uint32_t tuned_head_threads(uint32_t head_dim, const cudaDeviceProp &props) {
  const uint32_t warp_threads = props.warpSize > 0 ? props.warpSize : 32u;
  const uint32_t minimum = props.major >= 9 ? std::max(warp_threads, 64u)
                                            : warp_threads;
  const uint32_t exact_head_threads =
      next_pow2_at_least(head_dim, minimum, kHeadThreadsMax);
  const uint32_t compact_threads = next_pow2_at_least(
      ceil_div_u32(head_dim, kHeadThreadElements), minimum, kHeadThreadsMax);
  if (props.major >= 9 && compact_threads < exact_head_threads) {
    return compact_threads;
  }
  return exact_head_threads;
}

uint32_t decode_attention_chunks_for_cursor(
    const NervaCudaHfDecodeSequenceSession *session, uint32_t cursor) {
  const uint32_t kv_tokens = cursor >= session->max_context_tokens
                                 ? session->max_context_tokens
                                 : cursor + 1u;
  if (kv_tokens <= kChunkedDecodeAttentionThreshold ||
      session->decode_attention_max_chunks == 0 ||
      session->device_decode_attention_values == nullptr ||
      session->device_decode_attention_m == nullptr ||
      session->device_decode_attention_l == nullptr ||
      session->head_dim > kDecodeThreads) {
    return 0;
  }
  const uint32_t chunks =
      ceil_div_u32(kv_tokens, kDecodeAttentionChunkTokens);
  return std::min(chunks, session->decode_attention_max_chunks);
}

uint32_t decode_head_threads_for_session(
    const NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr) {
    return kHeadThreadsMax;
  }
  return next_pow2_at_least(session->head_dim, session->head_threads,
                            kHeadThreadsMax);
}

bool session_graph_matches(const NervaCudaHfDecodeSequenceSession *session,
                           uint32_t context_steps,
                           uint32_t prompt_token_count,
                           uint32_t has_eos_token,
                           uint32_t eos_token,
                           uint32_t attention_chunks) {
  return session->cached_graph_exec != nullptr &&
         session->cached_context_steps == context_steps &&
         session->cached_prompt_token_count == prompt_token_count &&
         session->cached_has_eos_token == has_eos_token &&
         session->cached_eos_token == eos_token &&
         session->cached_attention_chunks == attention_chunks;
}

bool use_cublas_layer_path(const NervaCudaHfDecodeSequenceSession *session) {
  const uint32_t attention_hidden = session->heads * session->head_dim;
  return session->hidden >= 128 && attention_hidden == session->hidden &&
         session->host_layouts.size() == session->layer_count &&
         session->device_projection_input != nullptr &&
         session->device_qkv_packed != nullptr &&
         session->device_gate_up_packed != nullptr &&
         session->cublas != nullptr && session->cublas_lt != nullptr;
}

cudaError_t autotune_session_lt_gemv_plans(
    NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr || !use_cublas_layer_path(session) ||
      session->layer_count == 0) {
    return cudaErrorInvalidValue;
  }
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const PackedProjectionShape packed_shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  const SequenceLayerLayout layout = session->host_layouts[0];
  LayerScratch scratch = layer_scratch_ptrs(
      session->device_scratch, session->hidden, attention_hidden, kv_hidden,
      session->intermediate);
  cudaError_t err = cudaMemsetAsync(
      session->device_projection_input, 0, session->projection_input_bytes,
      session->stream);
  if (err == cudaSuccess)
    err = create_lt_gemv_plan(
        &session->qkv_plan, static_cast<uint32_t>(packed_shape.qkv_rows),
        session->hidden, session->dtype);
  if (err == cudaSuccess)
    err = create_lt_gemv_plan(&session->attention_output_plan,
                              session->hidden, attention_hidden,
                              session->dtype);
  if (err == cudaSuccess)
    err = create_lt_gemv_plan(
        &session->gate_up_plan,
        static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden,
        session->dtype);
  if (err == cudaSuccess)
    err = create_lt_gemv_plan(&session->down_plan, session->hidden,
                              session->intermediate, session->dtype);
  if (err == cudaSuccess)
    err = create_lt_gemv_plan(&session->lm_head_plan, session->vocab_size,
                              session->hidden, session->dtype);
  if (err == cudaSuccess)
    err = autotune_lt_gemv_plan(
        session->cublas, session->cublas_lt, session->stream,
        session->cublas_workspace, kCublasWorkspaceBytes, &session->qkv_plan,
        session->device_qkv_packed, session->device_projection_input,
        scratch.q);
  if (err == cudaSuccess)
    err = autotune_lt_gemv_plan(
        session->cublas, session->cublas_lt, session->stream,
        session->cublas_workspace, kCublasWorkspaceBytes,
        &session->attention_output_plan,
        session->device_arena + layout.w_o, session->device_projection_input,
        scratch.residual);
  if (err == cudaSuccess)
    err = autotune_lt_gemv_plan(
        session->cublas, session->cublas_lt, session->stream,
        session->cublas_workspace, kCublasWorkspaceBytes,
        &session->gate_up_plan,
        session->device_gate_up_packed, session->device_projection_input,
        scratch.gate);
  if (err == cudaSuccess)
    err = autotune_lt_gemv_plan(
        session->cublas, session->cublas_lt, session->stream,
        session->cublas_workspace, kCublasWorkspaceBytes, &session->down_plan,
        session->device_arena + layout.w_down, session->device_projection_input,
        scratch.down);
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = autotune_lt_gemv_plan(
        session->cublas, session->cublas_lt, session->stream,
        session->cublas_workspace, kCublasWorkspaceBytes,
        &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, device_logits);
  }
  return err;
}

void copy_cached_profile(const NervaCudaHfDecodeSequenceSession *session,
                         NervaCudaHfDecodeSequenceResult *out) {
  out->projection_ns = session->cached_projection_ns;
  out->qkv_projection_ns = session->cached_qkv_projection_ns;
  out->attention_output_projection_ns =
      session->cached_attention_output_projection_ns;
  out->gate_up_projection_ns = session->cached_gate_up_projection_ns;
  out->down_projection_ns = session->cached_down_projection_ns;
  out->lm_head_projection_ns = session->cached_lm_head_projection_ns;
  out->attention_ns = session->cached_attention_ns;
  out->mlp_ns = session->cached_mlp_ns;
  out->norm_ns = session->cached_norm_ns;
  out->sampling_ns = session->cached_sampling_ns;
}

cudaError_t encoded_row_major_gemm_tokens_cached(
    NervaCudaHfDecodeSequenceSession *session, const uint16_t *matrix,
    const uint16_t *input, uint32_t rows, uint32_t cols, uint32_t tokens,
    uint32_t dtype, float beta, float *output);

cudaError_t project_encoded_rows(NervaCudaHfDecodeSequenceSession *session,
                                 const LtGemvPlan *single_token_plan,
                                 const uint16_t *matrix,
                                 const uint16_t *input, uint32_t rows,
                                 uint32_t cols, uint32_t tokens,
                                 uint32_t dtype, float beta,
                                 float *output) {
  if (session == nullptr || matrix == nullptr || input == nullptr ||
      output == nullptr || rows == 0 || cols == 0 || tokens == 0) {
    return cudaErrorInvalidValue;
  }
  if (tokens == 1) {
    if (single_token_plan != nullptr && single_token_plan->ready &&
        single_token_plan->rows == rows && single_token_plan->cols == cols &&
        single_token_plan->dtype == dtype) {
      return encoded_row_major_gemv_planned(
          session->cublas, session->cublas_lt, session->stream,
          session->cublas_workspace, kCublasWorkspaceBytes, single_token_plan,
          matrix, input, beta, output);
    }
    return encoded_row_major_gemv_beta(session->cublas, matrix, input, rows,
                                       cols, dtype, beta, output);
  }
  return encoded_row_major_gemm_tokens_cached(session, matrix, input, rows, cols,
                                             tokens, dtype, beta, output);
}

#if NERVA_HAVE_CUDNN_FRONTEND
bool cudnn_decode_debug_enabled() {
  static int enabled = []() {
    const char *value = getenv("NERVA_CUDNN_DECODE_DEBUG");
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
  }();
  return enabled != 0;
}

bool cudnn_decode_runtime_enabled() {
  static int enabled = []() {
    const char *value = getenv("NERVA_CUDNN_DECODE");
    if (value == nullptr || value[0] == '\0') {
      return 1;
    }
    const bool is_disabled =
        strcmp(value, "0") == 0 || strcmp(value, "false") == 0 ||
        strcmp(value, "False") == 0 || strcmp(value, "FALSE") == 0;
    return is_disabled ? 0 : 1;
  }();
  return enabled != 0;
}

void log_cudnn_decode_status(const char *phase,
                             cudnn_frontend::error_object status) {
  if (!cudnn_decode_debug_enabled()) {
    return;
  }
  fprintf(stderr, "[nerva-cudnn-decode] %s failed code=%d message=%s\n",
          phase, static_cast<int>(status.get_code()),
          status.get_message().c_str());
}

void log_cudnn_decode_cuda_error(const char *phase, cudaError_t err) {
  if (!cudnn_decode_debug_enabled()) {
    return;
  }
  fprintf(stderr, "[nerva-cudnn-decode] %s failed cuda=%s: %s\n", phase,
          cudaGetErrorName(err), cudaGetErrorString(err));
}

cudaError_t ensure_cudnn_prefill_sdpa_plan(
    NervaCudaHfDecodeSequenceSession *session, uint32_t seq_tokens) {
  if (session == nullptr || session->cudnn == nullptr || seq_tokens == 0 ||
      session->dtype != kDTypeBF16 || session->head_dim == 0 ||
      session->heads == 0 || session->kv_heads == 0 ||
      session->heads % session->kv_heads != 0) {
    return cudaErrorNotSupported;
  }
  if (session->cudnn_prefill_sdpa != nullptr &&
      session->cudnn_prefill_sdpa->seq_tokens == seq_tokens &&
      session->cudnn_prefill_sdpa->heads == session->heads &&
      session->cudnn_prefill_sdpa->kv_heads == session->kv_heads &&
      session->cudnn_prefill_sdpa->head_dim == session->head_dim) {
    return cudaSuccess;
  }

  auto *plan = new (std::nothrow) CudnnPrefillSdpaPlan();
  if (plan == nullptr) {
    return cudaErrorMemoryAllocation;
  }
  plan->seq_tokens = seq_tokens;
  plan->heads = session->heads;
  plan->kv_heads = session->kv_heads;
  plan->head_dim = session->head_dim;
  const uint64_t attention_hidden =
      static_cast<uint64_t>(session->heads) * session->head_dim;
  const uint64_t kv_hidden =
      static_cast<uint64_t>(session->kv_heads) * session->head_dim;
  plan->rows = attention_hidden + kv_hidden * 2;
  plan->graph = std::make_unique<cudnn_frontend::graph::Graph>();
  plan->graph->set_io_data_type(cudnn_frontend::DataType_t::BFLOAT16)
      .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

  constexpr int64_t kTensorQ = 9001;
  constexpr int64_t kTensorK = 9002;
  constexpr int64_t kTensorV = 9003;
  constexpr int64_t kTensorO = 9004;
  const int64_t batch = 1;
  const int64_t heads = static_cast<int64_t>(session->heads);
  const int64_t kv_heads = static_cast<int64_t>(session->kv_heads);
  const int64_t seq = static_cast<int64_t>(seq_tokens);
  const int64_t dim = static_cast<int64_t>(session->head_dim);
  const int64_t rows = static_cast<int64_t>(plan->rows);
  const std::vector<int64_t> q_stride = {seq * rows, dim, rows, 1};
  const std::vector<int64_t> k_stride = {seq * rows, dim, rows, 1};
  const std::vector<int64_t> v_stride = {seq * rows, dim, rows, 1};
  const std::vector<int64_t> o_stride = {
      seq * heads * dim, dim, heads * dim, 1};

  auto q_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_q")
                    .set_uid(kTensorQ)
                    .set_dim({batch, heads, seq, dim})
                    .set_stride(q_stride);
  auto k_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_k")
                    .set_uid(kTensorK)
                    .set_dim({batch, kv_heads, seq, dim})
                    .set_stride(k_stride);
  auto v_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_v")
                    .set_uid(kTensorV)
                    .set_dim({batch, kv_heads, seq, dim})
                    .set_stride(v_stride);
  auto o_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_o")
                    .set_uid(kTensorO)
                    .set_dim({batch, heads, seq, dim})
                    .set_stride(o_stride);

  auto sdpa = cudnn_frontend::graph::SDPA_attributes()
                  .set_name("nerva_prefill_sdpa")
                  .set_generate_stats(false)
                  .set_causal_mask(true)
                  .set_attn_scale(rsqrtf(static_cast<float>(session->head_dim)));
  auto q = plan->graph->tensor(q_desc);
  auto k = plan->graph->tensor(k_desc);
  auto v = plan->graph->tensor(v_desc);
  auto outputs = plan->graph->sdpa(q, k, v, sdpa);
  outputs[0]->set_output(true)
      .set_dim({batch, heads, seq, dim})
      .set_stride(o_stride)
      .set_uid(kTensorO);

  cudnnStatus_t stream_status = cudnnSetStream(session->cudnn, session->stream);
  if (stream_status != CUDNN_STATUS_SUCCESS) {
    delete plan;
    return cudnn_to_cuda(stream_status);
  }
  auto status = plan->graph->build(session->cudnn,
                                   {cudnn_frontend::HeurMode_t::A});
  if (status.is_bad()) {
    delete plan;
    return cudaErrorNotSupported;
  }
  const int64_t workspace = plan->graph->get_workspace_size();
  if (workspace < 0 ||
      static_cast<uint64_t>(workspace) > kCublasWorkspaceBytes) {
    delete plan;
    return cudaErrorMemoryAllocation;
  }
  plan->workspace_bytes = static_cast<size_t>(workspace);
  delete session->cudnn_prefill_sdpa;
  session->cudnn_prefill_sdpa = plan;
  return cudaSuccess;
}

cudaError_t execute_cudnn_prefill_sdpa(
    NervaCudaHfDecodeSequenceSession *session, uint32_t seq_tokens) {
  cudaError_t err = ensure_cudnn_prefill_sdpa_plan(session, seq_tokens);
  if (err != cudaSuccess) {
    return err;
  }
  constexpr int64_t kTensorQ = 9001;
  constexpr int64_t kTensorK = 9002;
  constexpr int64_t kTensorV = 9003;
  constexpr int64_t kTensorO = 9004;
  const uint64_t attention_hidden =
      static_cast<uint64_t>(session->heads) * session->head_dim;
  const uint64_t kv_hidden =
      static_cast<uint64_t>(session->kv_heads) * session->head_dim;
  uint16_t *base = session->device_prefill_qkv_encoded;
  std::unordered_map<int64_t, void *> tensors = {
      {kTensorQ, base},
      {kTensorK, base + attention_hidden},
      {kTensorV, base + attention_hidden + kv_hidden},
      {kTensorO, session->device_prefill_attn},
  };
  cudnnStatus_t stream_status = cudnnSetStream(session->cudnn, session->stream);
  if (stream_status != CUDNN_STATUS_SUCCESS) {
    return cudnn_to_cuda(stream_status);
  }
  auto status = session->cudnn_prefill_sdpa->graph->execute(
      session->cudnn, tensors, session->cublas_workspace);
  return status.is_good() ? cudaSuccess : cudaErrorLaunchFailure;
}

bool can_use_cudnn_decode_sdpa(const NervaCudaHfDecodeSequenceSession *session,
                               uint32_t attention_chunks) {
  const bool usable =
      session != nullptr && attention_chunks != 0 &&
      cudnn_decode_runtime_enabled() &&
      session->cudnn_decode_sdpa_disabled == 0 &&
      session->cudnn != nullptr && session->dtype == kDTypeBF16 &&
      session->heads != 0 && session->kv_heads != 0 &&
      session->heads % session->kv_heads == 0 && session->head_dim != 0 &&
      session->device_decode_q != nullptr &&
      session->device_decode_seq_len_q != nullptr &&
      session->device_decode_seq_len_kv != nullptr;
  if (!usable && cudnn_decode_debug_enabled()) {
    fprintf(stderr,
            "[nerva-cudnn-decode] gate failed session=%d chunks=%u disabled=%u "
            "cudnn=%d dtype=%u heads=%u kv_heads=%u head_dim=%u q=%d "
            "seq_q=%d seq_kv=%d\n",
            session != nullptr, attention_chunks,
            session == nullptr ? 0 : session->cudnn_decode_sdpa_disabled,
            session != nullptr && session->cudnn != nullptr,
            session == nullptr ? 0 : session->dtype,
            session == nullptr ? 0 : session->heads,
            session == nullptr ? 0 : session->kv_heads,
            session == nullptr ? 0 : session->head_dim,
            session != nullptr && session->device_decode_q != nullptr,
            session != nullptr && session->device_decode_seq_len_q != nullptr,
            session != nullptr && session->device_decode_seq_len_kv != nullptr);
  }
  return usable;
}

cudaError_t ensure_cudnn_decode_sdpa_plan(
    NervaCudaHfDecodeSequenceSession *session) {
  if (!can_use_cudnn_decode_sdpa(session, 1)) {
    return cudaErrorNotSupported;
  }
  if (session->cudnn_decode_sdpa != nullptr &&
      session->cudnn_decode_sdpa->max_context_tokens ==
          session->max_context_tokens &&
      session->cudnn_decode_sdpa->kv_token_capacity ==
          session->kv_token_capacity &&
      session->cudnn_decode_sdpa->heads == session->heads &&
      session->cudnn_decode_sdpa->kv_heads == session->kv_heads &&
      session->cudnn_decode_sdpa->head_dim == session->head_dim) {
    return cudaSuccess;
  }

  auto *plan = new (std::nothrow) CudnnDecodeSdpaPlan();
  if (plan == nullptr) {
    return cudaErrorMemoryAllocation;
  }
  plan->max_context_tokens = session->max_context_tokens;
  plan->kv_token_capacity = session->kv_token_capacity;
  plan->heads = session->heads;
  plan->kv_heads = session->kv_heads;
  plan->head_dim = session->head_dim;
  plan->graph = std::make_unique<cudnn_frontend::graph::Graph>();
  plan->graph->set_io_data_type(cudnn_frontend::DataType_t::BFLOAT16)
      .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

  constexpr int64_t kTensorQ = 9101;
  constexpr int64_t kTensorK = 9102;
  constexpr int64_t kTensorV = 9103;
  constexpr int64_t kTensorO = 9104;
  constexpr int64_t kTensorSeqLenQ = 9105;
  constexpr int64_t kTensorSeqLenKv = 9106;
  const int64_t batch = 1;
  const int64_t heads = static_cast<int64_t>(session->heads);
  const int64_t kv_heads = static_cast<int64_t>(session->kv_heads);
  const int64_t seq = static_cast<int64_t>(session->kv_token_capacity);
  const int64_t dim = static_cast<int64_t>(session->head_dim);
  const int64_t attention_hidden = heads * dim;
  const int64_t kv_hidden = kv_heads * dim;
  const std::vector<int64_t> q_stride = {attention_hidden, dim,
                                         attention_hidden, 1};
  const std::vector<int64_t> kv_stride = {seq * kv_hidden, dim, kv_hidden, 1};

  auto q_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_decode_q")
                    .set_uid(kTensorQ)
                    .set_data_type(cudnn_frontend::DataType_t::BFLOAT16)
                    .set_dim({batch, heads, 1, dim})
                    .set_stride(q_stride);
  auto k_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_decode_k_cache")
                    .set_uid(kTensorK)
                    .set_data_type(cudnn_frontend::DataType_t::BFLOAT16)
                    .set_dim({batch, kv_heads, seq, dim})
                    .set_stride(kv_stride);
  auto v_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_decode_v_cache")
                    .set_uid(kTensorV)
                    .set_data_type(cudnn_frontend::DataType_t::BFLOAT16)
                    .set_dim({batch, kv_heads, seq, dim})
                    .set_stride(kv_stride);
  auto seq_len_q_desc = cudnn_frontend::graph::Tensor_attributes()
                            .set_name("nerva_decode_seq_len_q")
                            .set_uid(kTensorSeqLenQ)
                            .set_data_type(cudnn_frontend::DataType_t::INT32)
                            .set_dim({batch, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(false);
  auto seq_len_kv_desc = cudnn_frontend::graph::Tensor_attributes()
                             .set_name("nerva_decode_seq_len_kv")
                             .set_uid(kTensorSeqLenKv)
                             .set_data_type(cudnn_frontend::DataType_t::INT32)
                             .set_dim({batch, 1, 1, 1})
                             .set_stride({1, 1, 1, 1})
                             .set_is_pass_by_value(false);

  auto q = plan->graph->tensor(q_desc);
  auto k = plan->graph->tensor(k_desc);
  auto v = plan->graph->tensor(v_desc);
  auto seq_len_q = plan->graph->tensor(seq_len_q_desc);
  auto seq_len_kv = plan->graph->tensor(seq_len_kv_desc);
  auto sdpa = cudnn_frontend::graph::SDPA_attributes()
                  .set_name("nerva_decode_sdpa")
                  .set_generate_stats(false)
                  .set_padding_mask(true)
                  .set_seq_len_q(seq_len_q)
                  .set_seq_len_kv(seq_len_kv)
                  .set_attn_scale(rsqrtf(static_cast<float>(session->head_dim)));
  auto outputs = plan->graph->sdpa(q, k, v, sdpa);
  outputs[0]->set_output(true)
      .set_uid(kTensorO)
      .set_data_type(cudnn_frontend::DataType_t::BFLOAT16)
      .set_dim({batch, heads, 1, dim})
      .set_stride(q_stride);

  cudnnStatus_t stream_status = cudnnSetStream(session->cudnn, session->stream);
  if (stream_status != CUDNN_STATUS_SUCCESS) {
    delete plan;
    return cudnn_to_cuda(stream_status);
  }
  auto status = plan->graph->build(session->cudnn,
                                   {cudnn_frontend::HeurMode_t::A});
  if (status.is_bad()) {
    log_cudnn_decode_status("build", status);
    delete plan;
    return cudaErrorNotSupported;
  }
  const int64_t workspace = plan->graph->get_workspace_size();
  if (workspace < 0 ||
      static_cast<uint64_t>(workspace) > kCublasWorkspaceBytes) {
    delete plan;
    return cudaErrorMemoryAllocation;
  }
  plan->workspace_bytes = static_cast<size_t>(workspace);
  if (cudnn_decode_debug_enabled()) {
    fprintf(stderr,
            "[nerva-cudnn-decode] build ok max_context=%u kv_capacity=%u "
            "heads=%u kv_heads=%u head_dim=%u workspace=%zu\n",
            session->max_context_tokens, session->kv_token_capacity,
            session->heads, session->kv_heads, session->head_dim,
            plan->workspace_bytes);
  }
  delete session->cudnn_decode_sdpa;
  session->cudnn_decode_sdpa = plan;
  return cudaSuccess;
}

cudaError_t execute_cudnn_decode_sdpa(
    NervaCudaHfDecodeSequenceSession *session, uint32_t layer_index) {
  cudaError_t err = ensure_cudnn_decode_sdpa_plan(session);
  if (err != cudaSuccess) {
    return err;
  }
  constexpr int64_t kTensorQ = 9101;
  constexpr int64_t kTensorK = 9102;
  constexpr int64_t kTensorV = 9103;
  constexpr int64_t kTensorO = 9104;
  constexpr int64_t kTensorSeqLenQ = 9105;
  constexpr int64_t kTensorSeqLenKv = 9106;
  const uint64_t kv_hidden =
      static_cast<uint64_t>(session->kv_heads) * session->head_dim;
  const uint64_t layer_kv_elements =
      static_cast<uint64_t>(session->kv_token_capacity) * kv_hidden;
  uint16_t *layer_keys =
      session->device_kv_keys + layer_kv_elements * layer_index;
  uint16_t *layer_values =
      session->device_kv_values + layer_kv_elements * layer_index;
  std::unordered_map<int64_t, void *> tensors = {
      {kTensorQ, session->device_decode_q},
      {kTensorK, layer_keys},
      {kTensorV, layer_values},
      {kTensorO, session->device_projection_input},
      {kTensorSeqLenQ, session->device_decode_seq_len_q},
      {kTensorSeqLenKv, session->device_decode_seq_len_kv},
  };
  auto status = session->cudnn_decode_sdpa->graph->execute(
      session->cudnn, tensors, session->cublas_workspace);
  if (status.is_bad()) {
    log_cudnn_decode_status("execute", status);
    return cudaErrorLaunchFailure;
  }
  return cudaSuccess;
}
#endif

void stash_prefill_metrics(NervaCudaHfDecodeSequenceSession *session,
                           const NervaCudaHfDecodeSequenceResult *out) {
  session->pending_prefill_kernel_launches = out->kernel_launches;
  session->pending_prefill_device_elapsed_ns = out->device_elapsed_ns;
  session->pending_prefill_sync_calls = out->sync_calls;
  session->pending_prefill_graph_replays = out->graph_replays;
  session->pending_prefill_graph_launches = out->graph_launches;
  session->pending_prefill_graph_nodes = out->graph_nodes;
  session->pending_prefill_available = 1;
}

void drain_prefill_metrics(NervaCudaHfDecodeSequenceSession *session,
                           NervaCudaHfDecodeSequenceResult *out) {
  if (session->pending_prefill_available == 0) {
    return;
  }
  out->kernel_launches += session->pending_prefill_kernel_launches;
  out->device_elapsed_ns += session->pending_prefill_device_elapsed_ns;
  out->sync_calls += session->pending_prefill_sync_calls;
  out->graph_replays += session->pending_prefill_graph_replays;
  out->graph_launches += session->pending_prefill_graph_launches;
  if (out->graph_nodes == 0) {
    out->graph_nodes = session->pending_prefill_graph_nodes;
  }
  session->pending_prefill_available = 0;
  session->pending_prefill_kernel_launches = 0;
  session->pending_prefill_device_elapsed_ns = 0;
  session->pending_prefill_sync_calls = 0;
  session->pending_prefill_graph_replays = 0;
  session->pending_prefill_graph_launches = 0;
  session->pending_prefill_graph_nodes = 0;
}

cudaError_t profile_begin(NervaCudaHfDecodeSequenceSession *session) {
  return cudaEventRecord(session->profile_start, session->stream);
}

cudaError_t profile_end(NervaCudaHfDecodeSequenceSession *session,
                        uint64_t *bucket) {
  cudaError_t err = cudaEventRecord(session->profile_stop, session->stream);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaEventSynchronize(session->profile_stop);
  if (err != cudaSuccess) {
    return err;
  }
  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, session->profile_start,
                             session->profile_stop);
  if (err == cudaSuccess && elapsed_ms > 0.0f) {
    uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_ms * 1000000.0f);
    *bucket += elapsed_ns == 0 ? 1 : elapsed_ns;
  }
  return err;
}

cudaError_t pack_session_weight_replicas(
    NervaCudaHfDecodeSequenceSession *session) {
  if (!use_cublas_layer_path(session)) {
    return cudaSuccess;
  }
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const PackedProjectionShape shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  hf_pack_qkv_weights_kernel<<<
      static_cast<uint32_t>(shape.qkv_rows * session->layer_count),
      kDecodeThreads, 0, session->stream>>>(
      session->device_qkv_packed, session->device_arena,
      session->device_layouts, session->layer_count, session->hidden,
      attention_hidden, kv_hidden);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  hf_pack_gate_up_weights_kernel<<<
      static_cast<uint32_t>(shape.gate_up_rows * session->layer_count),
      kDecodeThreads, 0, session->stream>>>(
      session->device_gate_up_packed, session->device_arena,
      session->device_layouts, session->layer_count, session->hidden,
      session->intermediate);
  return cudaGetLastError();
}

cudaError_t launch_monolithic_session_step(
    NervaCudaHfDecodeSequenceSession *session, uint32_t max_steps,
    uint32_t prompt_token_count, uint32_t has_eos_token, uint32_t eos_token) {
  hf_decode_sequence_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
      session->device_arena, session->arena_layout, session->device_layouts,
      session->layer_count, session->dtype, session->hidden, session->heads,
      session->kv_heads, session->head_dim, session->intermediate, 0,
      session->device_step, max_steps, session->device_prompt_tokens,
      prompt_token_count, session->rms_eps, session->rope_theta,
      session->device_scratch, session->device_kv_keys,
      session->device_kv_values, session->kv_block_count,
      session->device_kv_block_table,
      session->device_slots);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = final_head_gemv(session->cublas, session->device_arena,
                          session->arena_layout, session->dtype,
                          session->hidden, session->vocab_size, device_logits);
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    hf_decode_final_head_reduce_kernel<<<1, kDecodeSampleThreads, 0,
                                         session->stream>>>(
        session->device_step, max_steps, has_eos_token, eos_token,
        device_logits, session->vocab_size, session->device_slots);
    err = cudaGetLastError();
  }
  return err;
}

cudaError_t launch_cublas_layer_session_step(
    NervaCudaHfDecodeSequenceSession *session, uint32_t max_steps,
    uint32_t prompt_token_count, uint32_t has_eos_token, uint32_t eos_token,
    uint32_t attention_chunks) {
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const uint32_t decode_head_threads = decode_head_threads_for_session(session);
  cudaError_t err = cudaSuccess;
  uint64_t input_offset = session->arena_layout.input;
  uint64_t output_offset = session->arena_layout.scratch;
  LayerScratch scratch = layer_scratch_ptrs(
      session->device_scratch, session->hidden, attention_hidden, kv_hidden,
      session->intermediate);
  const PackedProjectionShape packed_shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  if (err == cudaSuccess && session->layer_count > 0) {
    const SequenceLayerLayout first_layout = session->host_layouts[0];
    hf_decode_prepare_first_attn_norm_encode_kernel<<<
        1, kDecodeNormThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, first_layout,
        session->dtype, session->hidden, attention_hidden, kv_hidden,
        session->intermediate, session->device_step, max_steps,
        session->device_prompt_tokens, prompt_token_count, session->device_slots,
        session->rms_eps, session->device_scratch,
        session->device_projection_input);
    err = cudaGetLastError();
  } else if (err == cudaSuccess) {
    hf_decode_prepare_input_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, session->hidden,
        session->device_step, max_steps, session->device_prompt_tokens,
        prompt_token_count, session->device_slots);
    err = cudaGetLastError();
  }
  for (uint32_t layer_index = 0;
       err == cudaSuccess && layer_index < session->layer_count;
       ++layer_index) {
    const SequenceLayerLayout layout = session->host_layouts[layer_index];
    err = project_encoded_rows(
        session, &session->qkv_plan,
        session->device_qkv_packed +
            packed_shape.qkv_elements_per_layer * layer_index,
        session->device_projection_input,
        static_cast<uint32_t>(packed_shape.qkv_rows), session->hidden, 1,
        session->dtype, 0.0f, scratch.q);
    if (err == cudaSuccess && attention_chunks == 0) {
      hf_layer_qkv_attention_encode_kernel<<<session->heads, decode_head_threads, 0,
                                             session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
#if NERVA_HAVE_CUDNN_FRONTEND
      const bool use_cudnn_decode_sdpa =
          can_use_cudnn_decode_sdpa(session, attention_chunks);
#else
      const bool use_cudnn_decode_sdpa = false;
#endif
      hf_layer_qkv_prepare_kernel<<<session->heads, decode_head_threads, 0,
                                    session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          use_cudnn_decode_sdpa ? session->device_decode_q : nullptr,
          use_cudnn_decode_sdpa ? session->device_decode_seq_len_q : nullptr,
          use_cudnn_decode_sdpa ? session->device_decode_seq_len_kv : nullptr);
      err = cudaGetLastError();
      bool ran_cudnn_decode_sdpa = false;
#if NERVA_HAVE_CUDNN_FRONTEND
      if (err == cudaSuccess && use_cudnn_decode_sdpa) {
        err = execute_cudnn_decode_sdpa(session, layer_index);
        if (err == cudaSuccess) {
          ran_cudnn_decode_sdpa = true;
        }
      }
#endif
      if (err == cudaSuccess && !ran_cudnn_decode_sdpa) {
        const uint32_t query_group = session->heads / session->kv_heads;
        const bool use_shared_warp_gqa =
            query_group == kGroupedGqaHeads &&
            session->heads % session->kv_heads == 0 &&
            session->head_dim <= kSharedWarpGqaHeadDimMax;
        const bool use_grouped_gqa =
            query_group == kGroupedGqaHeads &&
            session->heads % session->kv_heads == 0 &&
            session->head_dim <= kGroupedGqaHeadDimMax;
        const dim3 grid((use_shared_warp_gqa || use_grouped_gqa)
                            ? session->kv_heads
                            : session->heads,
                        attention_chunks);
        if (session->dtype == kDTypeBF16 && use_shared_warp_gqa) {
          hf_layer_shared_warp_gqa_attention_chunk_kernel<kDTypeBF16>
              <<<grid, kSharedWarpGqaThreads, 0, session->stream>>>(
                  layer_index, session->hidden, session->heads,
                  session->kv_heads, session->head_dim, session->intermediate,
                  session->device_step, max_steps, attention_chunks,
                  session->device_scratch, session->device_kv_keys,
                  session->device_kv_values,
                  session->device_decode_attention_values,
                  session->device_decode_attention_m,
                  session->device_decode_attention_l, session->kv_block_count,
                  session->device_kv_block_table);
        } else if (session->dtype == kDTypeBF16 && use_grouped_gqa) {
          hf_layer_grouped_gqa_attention_chunk_kernel<kDTypeBF16>
              <<<grid, kGroupedGqaThreads, 0, session->stream>>>(
                  layer_index, session->hidden, session->heads,
                  session->kv_heads, session->head_dim, session->intermediate,
                  session->device_step, max_steps, attention_chunks,
                  session->device_scratch, session->device_kv_keys,
                  session->device_kv_values,
                  session->device_decode_attention_values,
                  session->device_decode_attention_m,
                  session->device_decode_attention_l, session->kv_block_count,
                  session->device_kv_block_table);
        } else if (session->dtype == kDTypeBF16) {
          hf_layer_attention_chunk_kernel<kDTypeBF16>
              <<<grid, decode_head_threads, 0, session->stream>>>(
                  layer_index, session->hidden, session->heads,
                  session->kv_heads, session->head_dim, session->intermediate,
                  session->device_step, max_steps, attention_chunks,
                  session->device_scratch, session->device_kv_keys,
                  session->device_kv_values,
                  session->device_decode_attention_values,
                  session->device_decode_attention_m,
                  session->device_decode_attention_l, session->kv_block_count,
                  session->device_kv_block_table);
        } else if (use_shared_warp_gqa) {
          hf_layer_shared_warp_gqa_attention_chunk_kernel<kDTypeF16>
              <<<grid, kSharedWarpGqaThreads, 0, session->stream>>>(
                  layer_index, session->hidden, session->heads,
                  session->kv_heads, session->head_dim, session->intermediate,
                  session->device_step, max_steps, attention_chunks,
                  session->device_scratch, session->device_kv_keys,
                  session->device_kv_values,
                  session->device_decode_attention_values,
                  session->device_decode_attention_m,
                  session->device_decode_attention_l, session->kv_block_count,
                  session->device_kv_block_table);
        } else if (use_grouped_gqa) {
          hf_layer_grouped_gqa_attention_chunk_kernel<kDTypeF16>
              <<<grid, kGroupedGqaThreads, 0, session->stream>>>(
                  layer_index, session->hidden, session->heads,
                  session->kv_heads, session->head_dim, session->intermediate,
                  session->device_step, max_steps, attention_chunks,
                  session->device_scratch, session->device_kv_keys,
                  session->device_kv_values,
                  session->device_decode_attention_values,
                  session->device_decode_attention_m,
                  session->device_decode_attention_l, session->kv_block_count,
                  session->device_kv_block_table);
        } else {
          hf_layer_attention_chunk_kernel<kDTypeF16>
              <<<grid, decode_head_threads, 0, session->stream>>>(
                  layer_index, session->hidden, session->heads,
                  session->kv_heads, session->head_dim, session->intermediate,
                  session->device_step, max_steps, attention_chunks,
                  session->device_scratch, session->device_kv_keys,
                  session->device_kv_values,
                  session->device_decode_attention_values,
                  session->device_decode_attention_m,
                  session->device_decode_attention_l, session->kv_block_count,
                  session->device_kv_block_table);
        }
        err = cudaGetLastError();
      }
      if (err == cudaSuccess && !ran_cudnn_decode_sdpa) {
        const size_t reduce_shared_bytes =
            static_cast<size_t>(attention_chunks) * sizeof(float);
        hf_layer_attention_reduce_kernel<<<session->heads, decode_head_threads,
                                           reduce_shared_bytes,
                                           session->stream>>>(
            session->dtype, session->hidden, session->heads, session->kv_heads,
            session->head_dim, session->intermediate, session->device_step,
            max_steps, attention_chunks, session->device_scratch,
            session->device_decode_attention_values,
            session->device_decode_attention_m,
            session->device_decode_attention_l,
            session->device_projection_input);
        err = cudaGetLastError();
      }
    }
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->attention_output_plan,
          session->device_arena + layout.w_o, session->device_projection_input,
          session->hidden, attention_hidden, 1, session->dtype, 0.0f,
          scratch.residual);
    if (err == cudaSuccess) {
      hf_layer_mlp_norm_encode_kernel<<<1, kDecodeNormThreads, 0,
                                        session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->gate_up_plan,
          session->device_gate_up_packed +
              packed_shape.gate_up_elements_per_layer * layer_index,
          session->device_projection_input,
          static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden, 1,
          session->dtype, 0.0f, scratch.gate);
    if (err == cudaSuccess) {
      const uint32_t ff_blocks =
          (session->intermediate + kDecodeThreads - 1) / kDecodeThreads;
      hf_layer_ff_encode_kernel<<<ff_blocks, kDecodeThreads, 0,
                                  session->stream>>>(
          session->dtype, session->hidden, attention_hidden, kv_hidden,
          session->intermediate, session->device_step, max_steps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->down_plan,
          session->device_arena + layout.w_down, session->device_projection_input,
          session->hidden, session->intermediate, 1, session->dtype, 0.0f,
          scratch.down);
    if (err == cudaSuccess && layer_index + 1 < session->layer_count) {
      const SequenceLayerLayout next_layout =
          session->host_layouts[layer_index + 1];
      hf_layer_finish_next_attn_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, output_offset, next_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
      hf_layer_finish_final_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, session->arena_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    const uint64_t next_input = output_offset;
    output_offset = input_offset;
    input_offset = next_input;
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = project_encoded_rows(
        session, &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        1, session->dtype, 0.0f, device_logits);
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    hf_decode_final_head_reduce_kernel<<<1, kDecodeSampleThreads, 0,
                                         session->stream>>>(
        session->device_step, max_steps, has_eos_token, eos_token,
        device_logits, session->vocab_size, session->device_slots);
    err = cudaGetLastError();
  }
  return err;
}

cudaError_t profile_cublas_layer_session_step(
    NervaCudaHfDecodeSequenceSession *session, uint32_t max_steps,
    uint32_t prompt_token_count, uint32_t has_eos_token, uint32_t eos_token,
    uint32_t attention_chunks, uint32_t cursor) {
  uint64_t projection_ns = 0;
  uint64_t qkv_projection_ns = 0;
  uint64_t attention_output_projection_ns = 0;
  uint64_t gate_up_projection_ns = 0;
  uint64_t down_projection_ns = 0;
  uint64_t lm_head_projection_ns = 0;
  uint64_t attention_ns = 0;
  uint64_t mlp_ns = 0;
  uint64_t norm_ns = 0;
  uint64_t sampling_ns = 0;
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const uint32_t decode_head_threads = decode_head_threads_for_session(session);
  const PackedProjectionShape packed_shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  LayerScratch scratch = layer_scratch_ptrs(
      session->device_scratch, session->hidden, attention_hidden, kv_hidden,
      session->intermediate);

  hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(session->device_step,
                                                          cursor);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) err = profile_begin(session);
  if (err == cudaSuccess && session->layer_count > 0) {
    const SequenceLayerLayout first_layout = session->host_layouts[0];
    hf_decode_prepare_first_attn_norm_encode_kernel<<<
        1, kDecodeNormThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, first_layout,
        session->dtype, session->hidden, attention_hidden, kv_hidden,
        session->intermediate, session->device_step, max_steps,
        session->device_prompt_tokens, prompt_token_count, session->device_slots,
        session->rms_eps, session->device_scratch,
        session->device_projection_input);
    err = cudaGetLastError();
  } else if (err == cudaSuccess) {
    hf_decode_prepare_input_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, session->hidden,
        session->device_step, max_steps, session->device_prompt_tokens,
        prompt_token_count, session->device_slots);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = profile_end(session, &norm_ns);

  uint64_t input_offset = session->arena_layout.input;
  uint64_t output_offset = session->arena_layout.scratch;
  for (uint32_t layer_index = 0;
       err == cudaSuccess && layer_index < session->layer_count;
       ++layer_index) {
    const SequenceLayerLayout layout = session->host_layouts[layer_index];
    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->qkv_plan,
          session->device_qkv_packed +
              packed_shape.qkv_elements_per_layer * layer_index,
          session->device_projection_input,
          static_cast<uint32_t>(packed_shape.qkv_rows), session->hidden, 1,
          session->dtype, 0.0f, scratch.q);
    if (err == cudaSuccess) err = profile_end(session, &qkv_projection_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess && attention_chunks == 0) {
      hf_layer_qkv_attention_encode_kernel<<<session->heads, decode_head_threads, 0,
                                             session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
#if NERVA_HAVE_CUDNN_FRONTEND
      const bool use_cudnn_decode_sdpa =
          can_use_cudnn_decode_sdpa(session, attention_chunks);
#else
      const bool use_cudnn_decode_sdpa = false;
#endif
      hf_layer_qkv_prepare_kernel<<<session->heads, decode_head_threads, 0,
                                    session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          use_cudnn_decode_sdpa ? session->device_decode_q : nullptr,
          use_cudnn_decode_sdpa ? session->device_decode_seq_len_q : nullptr,
          use_cudnn_decode_sdpa ? session->device_decode_seq_len_kv : nullptr);
      err = cudaGetLastError();
      bool ran_cudnn_decode_sdpa = false;
#if NERVA_HAVE_CUDNN_FRONTEND
      if (err == cudaSuccess && use_cudnn_decode_sdpa) {
        err = execute_cudnn_decode_sdpa(session, layer_index);
        if (err == cudaSuccess) {
          ran_cudnn_decode_sdpa = true;
        }
      }
#endif
      if (err == cudaSuccess && !ran_cudnn_decode_sdpa) {
        const uint32_t query_group = session->heads / session->kv_heads;
        const bool use_shared_warp_gqa =
            query_group == kGroupedGqaHeads &&
            session->heads % session->kv_heads == 0 &&
            session->head_dim <= kSharedWarpGqaHeadDimMax;
        const bool use_grouped_gqa =
            query_group == kGroupedGqaHeads &&
            session->heads % session->kv_heads == 0 &&
            session->head_dim <= kGroupedGqaHeadDimMax;
        const dim3 grid((use_shared_warp_gqa || use_grouped_gqa)
                            ? session->kv_heads
                            : session->heads,
                        attention_chunks);
        if (session->dtype == kDTypeBF16 && use_shared_warp_gqa) {
          hf_layer_shared_warp_gqa_attention_chunk_kernel<kDTypeBF16>
              <<<grid, kSharedWarpGqaThreads, 0, session->stream>>>(
                  layer_index, session->hidden, session->heads,
                  session->kv_heads, session->head_dim, session->intermediate,
                  session->device_step, max_steps, attention_chunks,
                  session->device_scratch, session->device_kv_keys,
                  session->device_kv_values,
                  session->device_decode_attention_values,
                  session->device_decode_attention_m,
                  session->device_decode_attention_l, session->kv_block_count,
                  session->device_kv_block_table);
        } else if (session->dtype == kDTypeBF16 && use_grouped_gqa) {
          hf_layer_grouped_gqa_attention_chunk_kernel<kDTypeBF16>
              <<<grid, kGroupedGqaThreads, 0, session->stream>>>(
                  layer_index, session->hidden, session->heads,
                  session->kv_heads, session->head_dim, session->intermediate,
                  session->device_step, max_steps, attention_chunks,
                  session->device_scratch, session->device_kv_keys,
                  session->device_kv_values,
                  session->device_decode_attention_values,
                  session->device_decode_attention_m,
                  session->device_decode_attention_l, session->kv_block_count,
                  session->device_kv_block_table);
        } else if (session->dtype == kDTypeBF16) {
          hf_layer_attention_chunk_kernel<kDTypeBF16>
              <<<grid, decode_head_threads, 0, session->stream>>>(
                  layer_index, session->hidden, session->heads,
                  session->kv_heads, session->head_dim, session->intermediate,
                  session->device_step, max_steps, attention_chunks,
                  session->device_scratch, session->device_kv_keys,
                  session->device_kv_values,
                  session->device_decode_attention_values,
                  session->device_decode_attention_m,
                  session->device_decode_attention_l, session->kv_block_count,
                  session->device_kv_block_table);
        } else if (use_shared_warp_gqa) {
          hf_layer_shared_warp_gqa_attention_chunk_kernel<kDTypeF16>
              <<<grid, kSharedWarpGqaThreads, 0, session->stream>>>(
                  layer_index, session->hidden, session->heads,
                  session->kv_heads, session->head_dim, session->intermediate,
                  session->device_step, max_steps, attention_chunks,
                  session->device_scratch, session->device_kv_keys,
                  session->device_kv_values,
                  session->device_decode_attention_values,
                  session->device_decode_attention_m,
                  session->device_decode_attention_l, session->kv_block_count,
                  session->device_kv_block_table);
        } else if (use_grouped_gqa) {
          hf_layer_grouped_gqa_attention_chunk_kernel<kDTypeF16>
              <<<grid, kGroupedGqaThreads, 0, session->stream>>>(
                  layer_index, session->hidden, session->heads,
                  session->kv_heads, session->head_dim, session->intermediate,
                  session->device_step, max_steps, attention_chunks,
                  session->device_scratch, session->device_kv_keys,
                  session->device_kv_values,
                  session->device_decode_attention_values,
                  session->device_decode_attention_m,
                  session->device_decode_attention_l, session->kv_block_count,
                  session->device_kv_block_table);
        } else {
          hf_layer_attention_chunk_kernel<kDTypeF16>
              <<<grid, decode_head_threads, 0, session->stream>>>(
                  layer_index, session->hidden, session->heads,
                  session->kv_heads, session->head_dim, session->intermediate,
                  session->device_step, max_steps, attention_chunks,
                  session->device_scratch, session->device_kv_keys,
                  session->device_kv_values,
                  session->device_decode_attention_values,
                  session->device_decode_attention_m,
                  session->device_decode_attention_l, session->kv_block_count,
                  session->device_kv_block_table);
        }
        err = cudaGetLastError();
      }
      if (err == cudaSuccess && !ran_cudnn_decode_sdpa) {
        const size_t reduce_shared_bytes =
            static_cast<size_t>(attention_chunks) * sizeof(float);
        hf_layer_attention_reduce_kernel<<<session->heads, decode_head_threads,
                                           reduce_shared_bytes,
                                           session->stream>>>(
            session->dtype, session->hidden, session->heads, session->kv_heads,
            session->head_dim, session->intermediate, session->device_step,
            max_steps, attention_chunks, session->device_scratch,
            session->device_decode_attention_values,
            session->device_decode_attention_m,
            session->device_decode_attention_l,
            session->device_projection_input);
        err = cudaGetLastError();
      }
    }
    if (err == cudaSuccess) err = profile_end(session, &attention_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->attention_output_plan,
          session->device_arena + layout.w_o, session->device_projection_input,
          session->hidden, attention_hidden, 1, session->dtype, 0.0f,
          scratch.residual);
    if (err == cudaSuccess) {
      err = profile_end(session, &attention_output_projection_ns);
    }

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess) {
      hf_layer_mlp_norm_encode_kernel<<<1, kDecodeNormThreads, 0,
                                        session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = profile_end(session, &norm_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->gate_up_plan,
          session->device_gate_up_packed +
              packed_shape.gate_up_elements_per_layer * layer_index,
          session->device_projection_input,
          static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden, 1,
          session->dtype, 0.0f, scratch.gate);
    if (err == cudaSuccess) err = profile_end(session, &gate_up_projection_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess) {
      const uint32_t ff_blocks =
          (session->intermediate + kDecodeThreads - 1) / kDecodeThreads;
      hf_layer_ff_encode_kernel<<<ff_blocks, kDecodeThreads, 0,
                                  session->stream>>>(
          session->dtype, session->hidden, attention_hidden, kv_hidden,
          session->intermediate, session->device_step, max_steps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = profile_end(session, &mlp_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->down_plan,
          session->device_arena + layout.w_down, session->device_projection_input,
          session->hidden, session->intermediate, 1, session->dtype, 0.0f,
          scratch.down);
    if (err == cudaSuccess) err = profile_end(session, &down_projection_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess && layer_index + 1 < session->layer_count) {
      const SequenceLayerLayout next_layout =
          session->host_layouts[layer_index + 1];
      hf_layer_finish_next_attn_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, output_offset, next_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
      hf_layer_finish_final_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, session->arena_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = profile_end(session, &norm_ns);
    const uint64_t next_input = output_offset;
    output_offset = input_offset;
    input_offset = next_input;
  }

  if (err == cudaSuccess) err = profile_begin(session);
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = project_encoded_rows(
        session, &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        1, session->dtype, 0.0f, device_logits);
  }
  if (err == cudaSuccess) err = profile_end(session, &lm_head_projection_ns);

  if (err == cudaSuccess) err = profile_begin(session);
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    hf_decode_final_head_reduce_kernel<<<1, kDecodeSampleThreads, 0,
                                         session->stream>>>(
        session->device_step, max_steps, has_eos_token, eos_token,
        device_logits, session->vocab_size, session->device_slots);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = profile_end(session, &sampling_ns);

  if (err == cudaSuccess) {
    hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(
        session->device_step, cursor);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(session->stream);
  if (err == cudaSuccess) {
    projection_ns = qkv_projection_ns + attention_output_projection_ns +
                    gate_up_projection_ns + down_projection_ns +
                    lm_head_projection_ns;
    session->cached_projection_ns = projection_ns;
    session->cached_qkv_projection_ns = qkv_projection_ns;
    session->cached_attention_output_projection_ns =
        attention_output_projection_ns;
    session->cached_gate_up_projection_ns = gate_up_projection_ns;
    session->cached_down_projection_ns = down_projection_ns;
    session->cached_lm_head_projection_ns = lm_head_projection_ns;
    session->cached_attention_ns = attention_ns;
    session->cached_mlp_ns = mlp_ns;
    session->cached_norm_ns = norm_ns;
    session->cached_sampling_ns = sampling_ns;
  }
  return err;
}

cudaError_t ensure_session_graph(NervaCudaHfDecodeSequenceSession *session,
                                 uint32_t max_steps,
                                 uint32_t prompt_token_count,
                                 uint32_t has_eos_token,
                                 uint32_t eos_token,
                                 uint32_t attention_chunks,
                                 uint32_t profile_cursor,
                                 NervaCudaHfDecodeSequenceResult *out);

cudaError_t encoded_row_major_gemm_tokens_cached(
    NervaCudaHfDecodeSequenceSession *session, const uint16_t *matrix,
    const uint16_t *input, uint32_t rows, uint32_t cols, uint32_t tokens,
    uint32_t dtype, float beta, float *output) {
  if (session == nullptr) {
    return cudaErrorInvalidValue;
  }
  for (LtGemmTokensPlan &plan : session->projection_block_plans) {
    if (plan.ready && plan.rows == rows && plan.cols == cols &&
        plan.tokens == tokens && plan.dtype == dtype) {
      cudaError_t err = launch_lt_gemm_tokens_plan(
          session->cublas_lt, session->stream, session->cublas_workspace,
          kCublasWorkspaceBytes, &plan, matrix, input, beta, output);
      if (err == cudaSuccess) {
        return err;
      }
      return encoded_row_major_gemm_tokens(session->cublas, matrix, input, rows,
                                           cols, tokens, dtype, beta, output);
    }
  }

  session->projection_block_plans.emplace_back();
  LtGemmTokensPlan &plan = session->projection_block_plans.back();
  cudaError_t err = create_lt_gemm_tokens_plan(&plan, rows, cols, tokens, dtype);
  if (err == cudaSuccess) {
    err = launch_lt_gemm_tokens_plan(
        session->cublas_lt, session->stream, session->cublas_workspace,
        kCublasWorkspaceBytes, &plan, matrix, input, beta, output);
  }
  if (err == cudaSuccess) {
    return err;
  }
  destroy_lt_gemm_tokens_plan(&plan);
  session->projection_block_plans.pop_back();
  return encoded_row_major_gemm_tokens(session->cublas, matrix, input, rows,
                                       cols, tokens, dtype, beta, output);
}

cudaError_t launch_cublas_session_prefill(
    NervaCudaHfDecodeSequenceSession *session, uint32_t prompt_token_count,
    uint32_t has_eos_token, uint32_t eos_token,
    NervaCudaHfDecodeSequenceResult *out) {
  if (prompt_token_count == 0 || prompt_token_count > session->max_context_tokens ||
      !use_cublas_layer_path(session)) {
    return cudaErrorInvalidValue;
  }
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const PackedProjectionShape packed_shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  const bool collect_profile = session->detailed_profile != 0;
  uint64_t qkv_projection_ns = 0;
  uint64_t attention_output_projection_ns = 0;
  uint64_t gate_up_projection_ns = 0;
  uint64_t down_projection_ns = 0;
  uint64_t lm_head_projection_ns = 0;
  uint64_t attention_ns = 0;
  uint64_t mlp_ns = 0;
  uint64_t norm_ns = 0;
  uint64_t sampling_ns = 0;
  auto profile_stage_begin = [&]() -> cudaError_t {
    return collect_profile ? profile_begin(session) : cudaSuccess;
  };
  auto profile_stage_end = [&](uint64_t *bucket) -> cudaError_t {
    return collect_profile ? profile_end(session, bucket) : cudaSuccess;
  };
  cudaError_t err = cudaEventRecord(session->device_start, session->stream);
  if (err == cudaSuccess) {
    hf_prefill_embed_kernel<<<prompt_token_count, kDecodeThreads, 0,
                              session->stream>>>(
        session->device_arena, session->arena_layout, session->hidden,
        session->device_prompt_tokens, prompt_token_count,
        session->device_prefill_hidden_a);
    err = cudaGetLastError();
    out->kernel_launches += 1;
  }
  uint16_t *hidden_in = session->device_prefill_hidden_a;
  uint16_t *hidden_out = session->device_prefill_hidden_b;
  for (uint32_t layer_index = 0;
       err == cudaSuccess && layer_index < session->layer_count;
       ++layer_index) {
    const SequenceLayerLayout layout = session->host_layouts[layer_index];
    for (uint32_t chunk_start = 0;
         err == cudaSuccess && chunk_start < prompt_token_count;
         chunk_start += session->prefill_chunk_tokens) {
      const uint32_t chunk_tokens =
          std::min(session->prefill_chunk_tokens, prompt_token_count - chunk_start);
      if (err == cudaSuccess) err = profile_stage_begin();
      if (err == cudaSuccess) {
        hf_prefill_attn_norm_kernel<<<chunk_tokens, kDecodeThreads, 0,
                                      session->stream>>>(
            session->device_arena, layout, session->dtype, session->hidden,
            chunk_start, chunk_tokens, session->rms_eps, hidden_in,
            session->device_prefill_norm);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&norm_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        err = project_encoded_rows(
            session, &session->qkv_plan,
            session->device_qkv_packed +
                packed_shape.qkv_elements_per_layer * layer_index,
            session->device_prefill_norm,
            static_cast<uint32_t>(packed_shape.qkv_rows), session->hidden,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_qkv);
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&qkv_projection_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      const uint32_t query_group =
          session->kv_heads == 0 ? 0 : session->heads / session->kv_heads;
      const bool use_grouped_gqa =
          query_group == kGroupedGqaHeads &&
          session->heads % session->kv_heads == 0 &&
          session->head_dim <= kSharedWarpGqaHeadDimMax;
#if NERVA_HAVE_CUDNN_FRONTEND
      const bool use_cudnn_sdpa =
          session->cudnn_prefill_sdpa_disabled == 0 &&
          session->cudnn != nullptr &&
          session->device_prefill_qkv_encoded != nullptr &&
          session->dtype == kDTypeBF16 && use_grouped_gqa &&
          chunk_start == 0 && chunk_tokens == prompt_token_count &&
          session->head_dim <= 128;
#endif
      if (err == cudaSuccess) {
        const dim3 grid(chunk_tokens, std::max(session->heads, session->kv_heads));
        hf_prefill_qkv_publish_kernel<<<grid, session->head_threads, 0,
                                      session->stream>>>(
            session->device_arena, layout, layer_index, session->dtype,
            session->heads, session->kv_heads, session->head_dim,
            session->max_context_tokens, chunk_start, chunk_tokens,
            session->rms_eps, session->rope_theta, session->device_prefill_qkv,
            session->device_kv_keys, session->device_kv_values,
#if NERVA_HAVE_CUDNN_FRONTEND
            use_cudnn_sdpa ? session->device_prefill_qkv_encoded : nullptr,
#else
            nullptr,
#endif
            session->kv_block_count, session->device_kv_block_table);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&norm_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
#if NERVA_HAVE_CUDNN_FRONTEND
        bool ran_cudnn_sdpa = false;
        if (use_cudnn_sdpa) {
          err = execute_cudnn_prefill_sdpa(session, chunk_tokens);
          if (err == cudaSuccess) {
            out->kernel_launches += 1;
            ran_cudnn_sdpa = true;
          } else if (err == cudaErrorNotSupported ||
                     err == cudaErrorMemoryAllocation) {
            session->cudnn_prefill_sdpa_disabled = 1;
            err = cudaSuccess;
          }
        }
        if (!ran_cudnn_sdpa) {
#endif
        if (session->dtype == kDTypeBF16 && use_grouped_gqa) {
          const dim3 grid(chunk_tokens, session->kv_heads);
          hf_prefill_grouped_gqa_attention_direct_kernel<kDTypeBF16>
              <<<grid, kSharedWarpGqaThreads, 0, session->stream>>>(
                  layer_index, session->heads, session->kv_heads,
                  session->head_dim, session->max_context_tokens, chunk_start,
                  chunk_tokens, session->device_prefill_qkv,
                  session->device_kv_keys, session->device_kv_values,
                  session->kv_block_count, session->device_kv_block_table,
                  session->device_prefill_attn);
        } else if (use_grouped_gqa) {
          const dim3 grid(chunk_tokens, session->kv_heads);
          hf_prefill_grouped_gqa_attention_direct_kernel<kDTypeF16>
              <<<grid, kSharedWarpGqaThreads, 0, session->stream>>>(
                  layer_index, session->heads, session->kv_heads,
                  session->head_dim, session->max_context_tokens, chunk_start,
                  chunk_tokens, session->device_prefill_qkv,
                  session->device_kv_keys, session->device_kv_values,
                  session->kv_block_count, session->device_kv_block_table,
                  session->device_prefill_attn);
        } else {
          const dim3 grid(chunk_tokens, session->heads);
          hf_prefill_attention_kernel<<<grid, session->head_threads,
                                        session->head_dim * sizeof(float),
                                        session->stream>>>(
              layer_index, session->dtype, session->heads, session->kv_heads,
              session->head_dim, session->max_context_tokens, chunk_start,
              chunk_tokens, session->device_prefill_qkv, session->device_kv_keys,
              session->device_kv_values, session->kv_block_count,
              session->device_kv_block_table, session->device_prefill_attn);
        }
          err = cudaGetLastError();
          out->kernel_launches += 1;
#if NERVA_HAVE_CUDNN_FRONTEND
        }
#endif
      }
      if (err == cudaSuccess) err = profile_stage_end(&attention_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        err = project_encoded_rows(
            session, &session->attention_output_plan,
            session->device_arena + layout.w_o,
            session->device_prefill_attn, session->hidden, attention_hidden,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_o);
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) {
        err = profile_stage_end(&attention_output_projection_ns);
      }
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        hf_prefill_mlp_norm_kernel<<<chunk_tokens, kDecodeThreads, 0,
                                     session->stream>>>(
            session->device_arena, layout, session->dtype, session->hidden,
            chunk_start, chunk_tokens, session->rms_eps, hidden_in,
            session->device_prefill_o, session->device_prefill_norm);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&norm_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        err = project_encoded_rows(
            session, &session->gate_up_plan,
            session->device_gate_up_packed +
                packed_shape.gate_up_elements_per_layer * layer_index,
            session->device_prefill_norm,
            static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_gate_up);
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&gate_up_projection_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        const uint32_t blocks =
            static_cast<uint32_t>(
                (static_cast<uint64_t>(chunk_tokens) * session->intermediate +
                 kDecodeThreads - 1) /
                kDecodeThreads);
        hf_prefill_ff_kernel<<<blocks, kDecodeThreads, 0, session->stream>>>(
            session->dtype, session->intermediate, chunk_tokens,
            session->device_prefill_gate_up, session->device_prefill_ff);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&mlp_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        err = project_encoded_rows(
            session, &session->down_plan,
            session->device_arena + layout.w_down,
            session->device_prefill_ff, session->hidden, session->intermediate,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_down);
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&down_projection_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        const uint32_t blocks =
            static_cast<uint32_t>(
                (static_cast<uint64_t>(chunk_tokens) * session->hidden +
                 kDecodeThreads - 1) /
                kDecodeThreads);
        hf_prefill_finish_kernel<<<blocks, kDecodeThreads, 0, session->stream>>>(
            session->dtype, session->hidden, chunk_start, chunk_tokens,
            session->device_prefill_o, session->device_prefill_down, hidden_out);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&mlp_ns);
    }
    std::swap(hidden_in, hidden_out);
  }
  if (err == cudaSuccess) {
    err = profile_stage_begin();
  }
  if (err == cudaSuccess) {
    hf_prefill_final_norm_last_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, session->dtype,
        session->hidden, prompt_token_count, session->rms_eps, hidden_in,
        session->device_projection_input);
    err = cudaGetLastError();
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) err = profile_stage_end(&norm_ns);
  if (err == cudaSuccess) {
    hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(
        session->device_step, prompt_token_count - 1u);
    err = cudaGetLastError();
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) {
    err = profile_stage_begin();
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = project_encoded_rows(
        session, &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        1, session->dtype, 0.0f, device_logits);
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) err = profile_stage_end(&lm_head_projection_ns);
  if (err == cudaSuccess) {
    err = profile_stage_begin();
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    hf_decode_final_head_reduce_kernel<<<1, kDecodeSampleThreads, 0,
                                         session->stream>>>(
        session->device_step, session->max_context_tokens, has_eos_token,
        eos_token, device_logits, session->vocab_size, session->device_slots);
    err = cudaGetLastError();
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) err = profile_stage_end(&sampling_ns);
  if (err == cudaSuccess) {
    err = cudaEventRecord(session->device_stop, session->stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(session->stream);
    out->sync_calls += 1;
  }
  if (err == cudaSuccess) {
    float device_ms = 0.0f;
    err = cudaEventElapsedTime(&device_ms, session->device_start,
                               session->device_stop);
    if (err == cudaSuccess && device_ms > 0.0f) {
      out->device_elapsed_ns = static_cast<uint64_t>(device_ms * 1000000.0f);
      if (out->device_elapsed_ns == 0) out->device_elapsed_ns = 1;
    }
  }
  if (err == cudaSuccess && collect_profile) {
    out->qkv_projection_ns = qkv_projection_ns;
    out->attention_output_projection_ns = attention_output_projection_ns;
    out->gate_up_projection_ns = gate_up_projection_ns;
    out->down_projection_ns = down_projection_ns;
    out->lm_head_projection_ns = lm_head_projection_ns;
    out->projection_ns = qkv_projection_ns + attention_output_projection_ns +
                         gate_up_projection_ns + down_projection_ns +
                         lm_head_projection_ns;
    out->attention_ns = attention_ns;
    out->mlp_ns = mlp_ns;
    out->norm_ns = norm_ns;
    out->sampling_ns = sampling_ns;
  }
  return err;
}

cudaError_t launch_serial_session_prefill(
    NervaCudaHfDecodeSequenceSession *session, uint32_t prompt_token_count,
    uint32_t has_eos_token, uint32_t eos_token,
    NervaCudaHfDecodeSequenceResult *out) {
  cudaError_t err =
      ensure_session_graph(session, session->max_context_tokens, prompt_token_count,
                           has_eos_token, eos_token, 0, 0, out);
  if (err == cudaSuccess) {
    err = cudaEventRecord(session->device_start, session->stream);
  }
  for (uint32_t step = 0; err == cudaSuccess && step < prompt_token_count; ++step) {
    err = cudaGraphLaunch(session->cached_graph_exec, session->stream);
    if (err == cudaSuccess) {
      out->graph_replays += 1;
      out->graph_launches += 1;
      out->kernel_launches += out->graph_nodes == 0 ? 1 : out->graph_nodes;
    }
  }
  if (err == cudaSuccess) {
    err = cudaEventRecord(session->device_stop, session->stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(session->stream);
    out->sync_calls += 1;
  }
  if (err == cudaSuccess) {
    float device_ms = 0.0f;
    err = cudaEventElapsedTime(&device_ms, session->device_start,
                               session->device_stop);
    if (err == cudaSuccess && device_ms > 0.0f) {
      out->device_elapsed_ns = static_cast<uint64_t>(device_ms * 1000000.0f);
      if (out->device_elapsed_ns == 0) out->device_elapsed_ns = 1;
    }
  }
  return err;
}

void fill_session_result_header(const NervaCudaHfDecodeSequenceSession *session,
                                NervaCudaHfDecodeSequenceResult *out,
                                uint32_t steps, uint32_t seed_token) {
  out->device_count = 1;
  out->dtype = session->dtype;
  out->hidden = session->hidden;
  out->heads = session->heads;
  out->kv_heads = session->kv_heads;
  out->head_dim = session->head_dim;
  out->intermediate = session->intermediate;
  out->vocab_size = session->vocab_size;
  out->layer_count = session->layer_count;
  out->steps = steps;
  out->seed_token = seed_token;
  out->resident_weight_bytes = session->resident_weight_bytes;
  out->planned_weight_blocks = session->planned_weight_blocks;
  out->planned_gpu_resident_blocks = session->planned_gpu_resident_blocks;
  out->planned_gpu_staged_blocks = session->planned_gpu_staged_blocks;
  out->planned_weight_bytes = session->planned_weight_bytes;
  out->planned_gpu_resident_weight_bytes =
      session->planned_gpu_resident_weight_bytes;
  out->planned_gpu_staged_weight_bytes =
      session->planned_gpu_staged_weight_bytes;
  out->planned_weight_descriptor_count =
      session->planned_weight_descriptor_count;
  out->planned_weight_descriptor_hash =
      session->planned_weight_descriptor_hash;
}

uint32_t observed_from_slot_range(uint32_t steps, uint32_t has_eos_token,
                                  uint32_t eos_token,
                                  const NervaCudaSyntheticTokenSlot *slots) {
  uint32_t count = steps;
  for (uint32_t index = 0; index < steps; ++index) {
    if (slots[index].completion != kCompletionDeviceComplete) {
      count = index;
      break;
    }
    if (has_eos_token != 0 && slots[index].token == eos_token) {
      count = index + 1;
      break;
    }
  }
  return count;
}

cudaError_t ensure_session_graph(NervaCudaHfDecodeSequenceSession *session,
                                 uint32_t max_steps,
                                 uint32_t prompt_token_count,
                                 uint32_t has_eos_token,
                                 uint32_t eos_token,
                                 uint32_t attention_chunks,
                                 uint32_t profile_cursor,
                                 NervaCudaHfDecodeSequenceResult *out) {
  uint32_t cache_attention_chunks = attention_chunks;
#if NERVA_HAVE_CUDNN_FRONTEND
  if (session->cudnn_decode_sdpa != nullptr &&
      can_use_cudnn_decode_sdpa(session, attention_chunks)) {
    cache_attention_chunks = 1;
  }
#endif
  if (session_graph_matches(session, max_steps, prompt_token_count,
                            has_eos_token, eos_token, cache_attention_chunks)) {
    out->graph_nodes = session->cached_graph_nodes;
    out->graph_cache_hits = 1;
    copy_cached_profile(session, out);
    return cudaSuccess;
  }
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  cudaError_t err = cudaSuccess;
  for (uint32_t attempt = 0; attempt < 2; ++attempt) {
    reset_session_graph(session);
    bool tried_cudnn_decode_sdpa = false;
    bool captured_cudnn_decode_sdpa = false;
#if NERVA_HAVE_CUDNN_FRONTEND
    if (can_use_cudnn_decode_sdpa(session, attention_chunks)) {
      tried_cudnn_decode_sdpa = true;
      err = ensure_cudnn_decode_sdpa_plan(session);
      if (err != cudaSuccess) {
        session->cudnn_decode_sdpa_disabled = 1;
        err = cudaSuccess;
        tried_cudnn_decode_sdpa = false;
      } else {
        captured_cudnn_decode_sdpa = true;
      }
    }
#endif
    bool capture_started = false;
    err = cudaStreamBeginCapture(session->stream, cudaStreamCaptureModeGlobal);
    capture_started = err == cudaSuccess;
    if (err == cudaSuccess) {
      err = use_cublas_layer_path(session)
                ? launch_cublas_layer_session_step(
                      session, max_steps, prompt_token_count, has_eos_token,
                      eos_token, attention_chunks)
                : launch_monolithic_session_step(
                      session, max_steps, prompt_token_count, has_eos_token,
                      eos_token);
    }
    if (capture_started) {
      cudaError_t end_err = cudaStreamEndCapture(session->stream, &graph);
      if (err == cudaSuccess) {
        err = end_err;
      } else if (graph != nullptr) {
        cudaGraphDestroy(graph);
        graph = nullptr;
      }
    }
    if (err == cudaSuccess) {
      size_t graph_nodes = 0;
      err = cudaGraphGetNodes(graph, nullptr, &graph_nodes);
      out->graph_nodes = static_cast<uint64_t>(graph_nodes);
    }
    if (err == cudaSuccess) {
      err = cudaGraphInstantiate(&graph_exec, graph, 0);
    }
    if (err == cudaSuccess) {
      session->cached_graph = graph;
      session->cached_graph_exec = graph_exec;
      session->cached_context_steps = max_steps;
      session->cached_prompt_token_count = prompt_token_count;
      session->cached_has_eos_token = has_eos_token;
      session->cached_eos_token = eos_token;
      session->cached_attention_chunks = captured_cudnn_decode_sdpa
                                             ? 1
                                             : attention_chunks;
      session->cached_graph_nodes = out->graph_nodes;
      out->graph_captures = 1;
      graph = nullptr;
      graph_exec = nullptr;
      break;
    }
#if NERVA_HAVE_CUDNN_FRONTEND
    if (tried_cudnn_decode_sdpa) {
      log_cudnn_decode_cuda_error("graph capture", err);
      session->cudnn_decode_sdpa_disabled = 1;
      if (graph_exec != nullptr) {
        cudaGraphExecDestroy(graph_exec);
        graph_exec = nullptr;
      }
      if (graph != nullptr) {
        cudaGraphDestroy(graph);
        graph = nullptr;
      }
      continue;
    }
#endif
    break;
  }
  if (err == cudaSuccess && use_cublas_layer_path(session) &&
      session->detailed_profile != 0) {
    err = profile_cublas_layer_session_step(
        session, max_steps, prompt_token_count, has_eos_token, eos_token,
        attention_chunks, profile_cursor);
    if (err == cudaSuccess) {
      copy_cached_profile(session, out);
    }
  }
  if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  return err;
}

void fill_create_result(const NervaCudaHfDecodeSequenceSession *session,
                        NervaCudaHfDecodeSequenceSessionCreateResult *out) {
  out->status = 0;
  out->dtype = session->dtype;
  out->hidden = session->hidden;
  out->heads = session->heads;
  out->kv_heads = session->kv_heads;
  out->head_dim = session->head_dim;
  out->intermediate = session->intermediate;
  out->vocab_size = session->vocab_size;
  out->layer_count = session->layer_count;
  out->max_context_tokens = session->max_context_tokens;
  out->prefill_chunk_tokens = session->prefill_chunk_tokens;
  out->head_threads = session->head_threads;
  out->resident_weight_bytes = session->resident_weight_bytes;
  out->planned_weight_blocks = session->planned_weight_blocks;
  out->planned_gpu_resident_blocks = session->planned_gpu_resident_blocks;
  out->planned_gpu_staged_blocks = session->planned_gpu_staged_blocks;
  out->planned_weight_bytes = session->planned_weight_bytes;
  out->planned_gpu_resident_weight_bytes = session->planned_gpu_resident_weight_bytes;
  out->planned_gpu_staged_weight_bytes = session->planned_gpu_staged_weight_bytes;
  out->planned_weight_descriptor_count = session->planned_weight_descriptor_count;
  out->planned_weight_descriptor_hash = session->planned_weight_descriptor_hash;
  out->descriptor_gpu_resident_h2d_bytes = session->descriptor_gpu_resident_h2d_bytes;
  out->descriptor_gpu_staged_h2d_bytes = session->descriptor_gpu_staged_h2d_bytes;
  out->resident_kv_bytes = session->kv_bytes;
  out->device_arena_bytes = session_device_footprint(session);
  out->pinned_host_bytes = session->slots_bytes + session->load_staging_bytes;
  out->h2d_bytes = session->h2d_bytes;
  out->sync_calls = session->setup_sync_calls + 1;
}

int fail(NervaCudaHfDecodeSequenceSessionCreateResult *out, cudaError_t err,
         int32_t failure_stage) {
  out->cuda_error = static_cast<int32_t>(err);
  out->failure_stage = failure_stage;
  return -1;
}

}  // namespace

extern "C" int nerva_cuda_hf_decode_sequence_u16(
    const NervaCudaHfDecodeSequenceRequest *request,
    NervaCudaHfDecodeSequenceResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(request, out);
  if (!valid_request(request)) {
    return -1;
  }
  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorNoDevice);
    return -1;
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  const uint64_t hidden = request->hidden;
  const uint64_t attention_hidden = request->heads * request->head_dim;
  const uint64_t kv_hidden = request->kv_heads * request->head_dim;
  const uint64_t intermediate = request->intermediate;
  const uint64_t vocab_size = request->vocab_size;
  const uint32_t context_steps = request->prompt_token_count + request->steps - 1u;
  SequenceArenaLayout arena_layout{};
  std::vector<SequenceLayerLayout> layouts(request->layer_count);
  uint64_t elements = 0;
  arena_layout.embeddings = push(elements, vocab_size * hidden);
  arena_layout.input = push(elements, hidden);
  arena_layout.scratch = push(elements, hidden);
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    pack_layer(layouts[index], elements, request->layers[index], hidden,
               attention_hidden, kv_hidden, request->head_dim, intermediate);
  }
  arena_layout.final_norm = push(elements, hidden);
  arena_layout.lm_head = push(elements, vocab_size * hidden);
  const uint64_t arena_bytes = elements * sizeof(uint16_t);
  const uint64_t resident_weight_bytes = arena_bytes - (hidden * 2 * sizeof(uint16_t));
  if (request->planned_weight_blocks != 0 &&
      request->planned_weight_bytes != resident_weight_bytes) {
    out->status = -1;
    return -1;
  }
  if (!validate_weight_descriptors(request, resident_weight_bytes, out)) {
    out->status = -1;
    return -1;
  }
  const uint64_t layout_bytes = layouts.size() * sizeof(SequenceLayerLayout);
  const uint64_t block_scratch =
      hidden * 5 + attention_hidden * 2 + kv_hidden * 2 + intermediate * 3;
  const uint64_t final_scratch = hidden * 2 + vocab_size;
  const uint64_t scratch_elements =
      block_scratch > final_scratch ? block_scratch : final_scratch;
  const uint64_t scratch_bytes = scratch_elements * sizeof(float);
  const uint32_t kv_block_count =
      ceil_div_u32(context_steps, kKvCacheBlockTokens);
  const uint32_t kv_token_capacity = kv_block_count * kKvCacheBlockTokens;
  const uint64_t kv_bytes =
      request->layer_count * static_cast<uint64_t>(kv_token_capacity) * kv_hidden *
      sizeof(uint16_t) * 2;
  const uint64_t kv_block_table_bytes =
      static_cast<uint64_t>(kv_block_count) * sizeof(uint32_t);
  const uint64_t slots_bytes =
      static_cast<uint64_t>(context_steps) * sizeof(NervaCudaSyntheticTokenSlot);
  const uint64_t prompt_bytes =
      static_cast<uint64_t>(request->prompt_token_count) * sizeof(uint32_t);
  const bool descriptor_mode = request->planned_weight_blocks != 0;
  const uint64_t host_weight_bytes =
      descriptor_mode ? pinned_weight_staging_bytes(request, resident_weight_bytes)
                      : arena_bytes;
  uint64_t setup_sync_calls = 0;

  uint16_t *host_arena = nullptr;
  uint16_t *device_arena = nullptr;
  SequenceLayerLayout *device_layouts = nullptr;
  float *device_scratch = nullptr;
  uint16_t *device_kv_keys = nullptr;
  uint16_t *device_kv_values = nullptr;
  uint32_t *device_kv_block_table = nullptr;
  uint32_t *device_prompt_tokens = nullptr;
  NervaCudaSyntheticTokenSlot *host_slots = nullptr;
  NervaCudaSyntheticTokenSlot *device_slots = nullptr;
  uint32_t *device_step = nullptr;
  void *cublas_workspace = nullptr;
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas = nullptr;
  cudaEvent_t device_start = nullptr;
  cudaEvent_t device_stop = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  err = cudaHostAlloc(reinterpret_cast<void **>(&host_arena), host_weight_bytes,
                      cudaHostAllocDefault);
  if (err == cudaSuccess)
    err = cudaHostAlloc(reinterpret_cast<void **>(&host_slots), slots_bytes,
                        cudaHostAllocDefault);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_arena), arena_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_layouts), layout_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_scratch), scratch_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_kv_keys), kv_bytes / 2);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_kv_values), kv_bytes / 2);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_kv_block_table), kv_block_table_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_prompt_tokens), prompt_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_slots), slots_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_step), sizeof(uint32_t));
  if (err == cudaSuccess) err = cudaMalloc(&cublas_workspace, kCublasWorkspaceBytes);
  if (err == cudaSuccess) err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err == cudaSuccess) err = cublas_to_cuda(cublasCreate(&cublas));
  if (err == cudaSuccess) {
    err = configure_cublas(cublas, stream, cublas_workspace,
                           kCublasWorkspaceBytes);
  }
  if (err == cudaSuccess) err = cudaEventCreate(&device_start);
  if (err == cudaSuccess) err = cudaEventCreate(&device_stop);
  if (err != cudaSuccess) {
    fail(out, err);
    if (device_stop != nullptr) cudaEventDestroy(device_stop);
    if (device_start != nullptr) cudaEventDestroy(device_start);
    if (cublas != nullptr) cublasDestroy(cublas);
    if (stream != nullptr) cudaStreamDestroy(stream);
    cudaFree(cublas_workspace);
    cudaFree(device_step);
    cudaFree(device_slots);
    cudaFree(device_prompt_tokens);
    cudaFree(device_kv_block_table);
    cudaFree(device_kv_values);
    cudaFree(device_kv_keys);
    cudaFree(device_scratch);
    cudaFree(device_layouts);
    cudaFree(device_arena);
    cudaFreeHost(host_slots);
    cudaFreeHost(host_arena);
    return -1;
  }

  memset(host_arena, 0, host_weight_bytes);
  memset(host_slots, 0, slots_bytes);
  const uint64_t embedding_bytes = vocab_size * hidden * sizeof(uint16_t);
  const uint64_t scratch_gap_bytes = hidden * 2 * sizeof(uint16_t);
  if (!descriptor_mode) {
    memcpy(host_arena + arena_layout.embeddings, request->embeddings,
           vocab_size * hidden * sizeof(uint16_t));
    for (uint32_t index = 0; index < request->layer_count; ++index) {
      copy_layer(host_arena, layouts[index], request->layers[index], hidden,
                 attention_hidden, kv_hidden, request->head_dim, intermediate);
    }
    memcpy(host_arena + arena_layout.final_norm, request->final_norm_weight,
           hidden * sizeof(uint16_t));
    memcpy(host_arena + arena_layout.lm_head, request->lm_head,
           vocab_size * hidden * sizeof(uint16_t));
  }

  if (err == cudaSuccess && descriptor_mode) {
    err = copy_weight_descriptors_to_device(
        device_arena, host_arena, host_weight_bytes, request, arena_bytes,
        embedding_bytes, scratch_gap_bytes, stream, out, &setup_sync_calls);
  } else if (err == cudaSuccess) {
    err = cudaMemcpyAsync(device_arena, host_arena, arena_bytes,
                          cudaMemcpyHostToDevice, stream);
    out->h2d_bytes = arena_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(device_layouts, layouts.data(), layout_bytes,
                          cudaMemcpyHostToDevice, stream);
    out->h2d_bytes += layout_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(device_prompt_tokens, request->prompt_tokens, prompt_bytes,
                          cudaMemcpyHostToDevice, stream);
    out->h2d_bytes += prompt_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(device_slots, 0, slots_bytes, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(device_kv_keys, 0, kv_bytes / 2, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(device_kv_values, 0, kv_bytes / 2, stream);
  }
  if (err == cudaSuccess) {
    const uint32_t blocks = (kv_block_count + kDecodeThreads - 1u) / kDecodeThreads;
    hf_init_identity_kv_block_table_kernel<<<blocks, kDecodeThreads, 0, stream>>>(
        device_kv_block_table, kv_block_count);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(device_step, 0, sizeof(uint32_t), stream);
  }
  if (err == cudaSuccess) {
    err = warm_cublas_gemv(cublas, device_arena, arena_layout, request->dtype,
                           device_scratch, stream);
  }
  bool capture_started = false;
  if (err == cudaSuccess) {
    err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    capture_started = err == cudaSuccess;
  }
  if (err == cudaSuccess) {
    hf_decode_sequence_kernel<<<1, kDecodeThreads, 0, stream>>>(
        device_arena, arena_layout, device_layouts, request->layer_count, request->dtype,
        request->hidden, request->heads, request->kv_heads, request->head_dim,
        request->intermediate, 0, device_step, context_steps, device_prompt_tokens,
        request->prompt_token_count, request->rms_eps, request->rope_theta,
        device_scratch, device_kv_keys, device_kv_values, kv_block_count,
        device_kv_block_table, device_slots);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    float *device_logits = device_scratch + hidden * 2;
    err = final_head_gemv(cublas, device_arena, arena_layout, request->dtype,
                          request->hidden, request->vocab_size, device_logits);
  }
  if (err == cudaSuccess) {
    float *device_logits = device_scratch + hidden * 2;
    hf_decode_final_head_reduce_kernel<<<1, kDecodeSampleThreads, 0, stream>>>(
        device_step, context_steps, request->has_eos_token, request->eos_token,
        device_logits, request->vocab_size, device_slots);
    err = cudaGetLastError();
  }
  if (capture_started) {
    cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
    if (err == cudaSuccess) {
      err = end_err;
    } else if (graph != nullptr) {
      cudaGraphDestroy(graph);
      graph = nullptr;
    }
  }
  if (err == cudaSuccess) {
    size_t graph_nodes = 0;
    err = cudaGraphGetNodes(graph, nullptr, &graph_nodes);
    out->graph_nodes = static_cast<uint64_t>(graph_nodes);
    out->graph_captures = 1;
  }
  if (err == cudaSuccess) {
    err = cudaGraphInstantiate(&graph_exec, graph, 0);
  }
  if (err == cudaSuccess) {
    err = cudaEventRecord(device_start, stream);
  }
  for (uint32_t step = 0; err == cudaSuccess && step < context_steps; ++step) {
    err = cudaGraphLaunch(graph_exec, stream);
    if (err == cudaSuccess) {
      out->graph_replays += 1;
      out->graph_launches += 1;
      out->kernel_launches += out->graph_nodes == 0 ? 1 : out->graph_nodes;
    }
  }
  if (err == cudaSuccess) {
    err = cudaEventRecord(device_stop, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_slots, device_slots, slots_bytes,
                          cudaMemcpyDeviceToHost, stream);
    out->d2h_bytes = slots_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
    out->sync_calls = setup_sync_calls + 1;
  }
  if (err == cudaSuccess) {
    float device_ms = 0.0f;
    err = cudaEventElapsedTime(&device_ms, device_start, device_stop);
    if (err == cudaSuccess && device_ms > 0.0f) {
      out->device_elapsed_ns = static_cast<uint64_t>(device_ms * 1000000.0f);
      if (out->device_elapsed_ns == 0) {
        out->device_elapsed_ns = 1;
      }
    }
  }

  if (err == cudaSuccess) {
    out->observed_tokens = observed_count(request, host_slots);
    const uint32_t output_start = request->prompt_token_count - 1u;
    for (uint32_t index = 0; index < out->observed_tokens; ++index) {
      request->output_tokens[index] = host_slots[output_start + index].token;
    }
    out->last_token = out->observed_tokens == 0
                          ? 0
                          : request->output_tokens[out->observed_tokens - 1];
    out->observed_token_hash = hash_tokens(request->output_tokens, out->observed_tokens);
    out->resident_weight_bytes = resident_weight_bytes;
    out->resident_kv_bytes = kv_bytes;
    out->kv_tokens = output_start + out->observed_tokens;
    out->device_arena_bytes =
        arena_bytes + layout_bytes + scratch_bytes + kv_bytes +
        kv_block_table_bytes + prompt_bytes + slots_bytes + sizeof(uint32_t) +
        kCublasWorkspaceBytes;
    out->pinned_host_bytes = host_weight_bytes + slots_bytes;
    out->host_causality_edges = 0;
    out->status = out->observed_tokens > 0 ? 0 : -1;
    for (uint32_t index = 0; out->status == 0 && index < out->observed_tokens; ++index) {
      const NervaCudaSyntheticTokenSlot &slot = host_slots[output_start + index];
      if (slot.request_id != kRequestId || slot.sequence_id != kSequenceId ||
          slot.completion != kCompletionDeviceComplete ||
          slot.token_index != output_start + index) {
        out->status = -1;
      }
    }
  } else {
    fail(out, err);
  }

  if (graph_exec != nullptr) {
    cudaGraphExecDestroy(graph_exec);
  }
  if (graph != nullptr) {
    cudaGraphDestroy(graph);
  }
  if (device_stop != nullptr) {
    cudaEventDestroy(device_stop);
  }
  if (device_start != nullptr) {
    cudaEventDestroy(device_start);
  }
  if (cublas != nullptr) {
    cublasDestroy(cublas);
  }
  cudaStreamDestroy(stream);
  cudaFree(cublas_workspace);
  cudaFree(device_step);
  cudaFree(device_slots);
  cudaFree(device_prompt_tokens);
  cudaFree(device_kv_block_table);
  cudaFree(device_kv_values);
  cudaFree(device_kv_keys);
  cudaFree(device_scratch);
  cudaFree(device_layouts);
  cudaFree(device_arena);
  cudaFreeHost(host_slots);
  cudaFreeHost(host_arena);
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_hf_decode_sequence_session_create(
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request,
    NervaCudaHfDecodeSequenceSessionCreateResult *out,
    NervaCudaHfDecodeSequenceSession **session_out) {
  if (out == nullptr || session_out == nullptr) {
    return -1;
  }
  *session_out = nullptr;
  clear_session_create_result(request, out);
  if (request == nullptr) {
    out->failure_stage = kCreateStageInvalidRequest;
    return -1;
  }
  const bool descriptor_mode = has_declared_weight_plan(request);
  if (request->layers == nullptr ||
      (!descriptor_mode &&
       (request->embeddings == nullptr || request->final_norm_weight == nullptr ||
        request->lm_head == nullptr)) ||
      request->layer_count == 0 || request->max_context_tokens == 0 ||
      request->hidden == 0 || request->heads == 0 || request->kv_heads == 0 ||
      request->head_dim == 0 || request->intermediate == 0 ||
      request->vocab_size == 0 || request->dtype > kDTypeBF16 ||
      request->kv_heads > request->heads || request->heads % request->kv_heads != 0 ||
      (request->rope_theta > 0.0f && request->head_dim % 2 != 0)) {
    out->failure_stage = kCreateStageInvalidRequest;
    return -1;
  }
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    if (!valid_layer(request->layers[index], !descriptor_mode)) {
      out->failure_stage = kCreateStageInvalidRequest;
      return -1;
    }
  }
  if (descriptor_mode &&
      (request->planned_weight_blocks == 0 || request->planned_weight_bytes == 0 ||
       request->planned_weight_descriptors == nullptr ||
       request->planned_weight_descriptor_count != request->planned_weight_blocks ||
       request->planned_weight_descriptor_hash == 0)) {
    out->failure_stage = kCreateStageInvalidRequest;
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err, kCreateStageGetDeviceCount);
  }
  if (out->device_count <= 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorNoDevice);
    out->failure_stage = kCreateStageGetDeviceCount;
    return -1;
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err, kCreateStageSetDevice);
  }
  cudaDeviceProp device_props{};
  cudaError_t props_err = cudaGetDeviceProperties(&device_props, 0);
  if (props_err != cudaSuccess) {
    device_props.warpSize = 32;
    device_props.major = 0;
    cudaGetLastError();
  }
  size_t device_free_before_alloc = 0;
  size_t device_total_before_alloc = 0;
  cudaError_t mem_info_err =
      cudaMemGetInfo(&device_free_before_alloc, &device_total_before_alloc);
  if (mem_info_err != cudaSuccess) {
    device_free_before_alloc = 0;
    device_total_before_alloc = 0;
  }

  auto *session = new (std::nothrow) NervaCudaHfDecodeSequenceSession();
  if (session == nullptr) {
    out->cuda_error = static_cast<int32_t>(cudaErrorMemoryAllocation);
    out->failure_stage = kCreateStageSessionAlloc;
    return -1;
  }
  const uint64_t hidden = request->hidden;
  const uint64_t attention_hidden = request->heads * request->head_dim;
  const uint64_t kv_hidden = request->kv_heads * request->head_dim;
  const uint64_t intermediate = request->intermediate;
  const uint64_t vocab_size = request->vocab_size;
  std::vector<SequenceLayerLayout> layouts(request->layer_count);
  uint64_t elements = 0;
  session->arena_layout.embeddings = push(elements, vocab_size * hidden);
  session->arena_layout.input = push(elements, hidden);
  session->arena_layout.scratch = push(elements, hidden);
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    pack_layer(layouts[index], elements, request->layers[index], hidden,
               attention_hidden, kv_hidden, request->head_dim, intermediate);
  }
  session->arena_layout.final_norm = push(elements, hidden);
  session->arena_layout.lm_head = push(elements, vocab_size * hidden);
  session->arena_bytes = elements * sizeof(uint16_t);
  session->resident_weight_bytes = session->arena_bytes - hidden * 2 * sizeof(uint16_t);
  if (request->planned_weight_blocks != 0 &&
      request->planned_weight_bytes != session->resident_weight_bytes) {
    out->failure_stage = kCreateStageInvalidRequest;
    delete session;
    return -1;
  }
  if (!validate_weight_descriptors(request, session->resident_weight_bytes, out)) {
    out->failure_stage = kCreateStageInvalidRequest;
    delete session;
    return -1;
  }

  const uint64_t block_scratch =
      hidden * 5 + attention_hidden * 2 + kv_hidden * 2 + intermediate * 3;
  const uint64_t final_scratch = hidden * 2 + vocab_size;
  const uint64_t scratch_elements =
      block_scratch > final_scratch ? block_scratch : final_scratch;
  const uint64_t projection_input_elements =
      intermediate > attention_hidden
          ? (intermediate > hidden ? intermediate : hidden)
          : (attention_hidden > hidden ? attention_hidden : hidden);
  const uint64_t prefill_qkv_rows = attention_hidden + kv_hidden * 2;
  const uint64_t prefill_gate_up_rows = intermediate * 2;
  const bool pack_cublas =
      should_pack_cublas_weights(request->hidden, attention_hidden);
  const PackedProjectionShape packed_shape = packed_projection_shape(
      hidden, attention_hidden, kv_hidden, intermediate);
  session->dtype = request->dtype;
  session->hidden = request->hidden;
  session->heads = request->heads;
  session->kv_heads = request->kv_heads;
  session->head_dim = request->head_dim;
  session->head_threads = tuned_head_threads(request->head_dim, device_props);
  session->intermediate = request->intermediate;
  session->vocab_size = request->vocab_size;
  session->layer_count = request->layer_count;
  session->max_context_tokens = request->max_context_tokens;
  session->kv_block_count =
      ceil_div_u32(request->max_context_tokens, kKvCacheBlockTokens);
  session->kv_token_capacity = session->kv_block_count * kKvCacheBlockTokens;
  session->detailed_profile = request->detailed_profile == 0 ? 0u : 1u;
  session->rms_eps = request->rms_eps;
  session->rope_theta = request->rope_theta;
  session->layout_bytes = layouts.size() * sizeof(SequenceLayerLayout);
  session->scratch_bytes = scratch_elements * sizeof(float);
  session->projection_input_bytes = projection_input_elements * sizeof(uint16_t);
  session->prefill_hidden_bytes =
      static_cast<uint64_t>(request->max_context_tokens) * hidden *
      sizeof(uint16_t);
  session->decode_attention_max_chunks =
      ceil_div_u32(request->max_context_tokens, kDecodeAttentionChunkTokens);
  session->decode_attention_values_bytes =
      static_cast<uint64_t>(request->heads) *
      session->decode_attention_max_chunks * request->head_dim * sizeof(float);
  session->decode_attention_stats_bytes =
      static_cast<uint64_t>(request->heads) *
      session->decode_attention_max_chunks * sizeof(float);
  session->decode_q_bytes =
      static_cast<uint64_t>(attention_hidden) * sizeof(uint16_t);
  session->decode_seq_len_bytes = sizeof(int32_t) * 2u;
  if (pack_cublas) {
    session->packed_qkv_bytes =
        packed_shape.qkv_elements_per_layer * request->layer_count *
        sizeof(uint16_t);
    session->packed_gate_up_bytes =
        packed_shape.gate_up_elements_per_layer * request->layer_count *
        sizeof(uint16_t);
  }
  session->kv_bytes =
      request->layer_count * static_cast<uint64_t>(session->kv_token_capacity) *
      kv_hidden * sizeof(uint16_t) * 2;
  session->kv_block_table_bytes =
      static_cast<uint64_t>(session->kv_block_count) * sizeof(uint32_t);
  session->slots_bytes =
      static_cast<uint64_t>(request->max_context_tokens) *
      sizeof(NervaCudaSyntheticTokenSlot);
  session->prompt_bytes =
      static_cast<uint64_t>(request->max_context_tokens) * sizeof(uint32_t);
  const uint64_t fixed_device_bytes =
      session_fixed_footprint_without_prefill_chunk(session);
  const uint32_t prefill_chunk = tune_prefill_chunk_tokens(
      request->max_context_tokens, fixed_device_bytes, projection_input_elements,
      prefill_qkv_rows, attention_hidden, hidden, prefill_gate_up_rows,
      intermediate, static_cast<uint64_t>(device_free_before_alloc));
  session->prefill_chunk_tokens = prefill_chunk;
  session->prefill_norm_bytes =
      projection_input_elements * static_cast<uint64_t>(prefill_chunk) *
      sizeof(uint16_t);
  session->prefill_qkv_bytes =
      prefill_qkv_rows * static_cast<uint64_t>(prefill_chunk) * sizeof(float);
  session->prefill_qkv_encoded_bytes =
      prefill_qkv_rows * static_cast<uint64_t>(prefill_chunk) *
      sizeof(uint16_t);
  session->prefill_attn_bytes =
      attention_hidden * static_cast<uint64_t>(prefill_chunk) * sizeof(uint16_t);
  session->prefill_o_bytes =
      hidden * static_cast<uint64_t>(prefill_chunk) * sizeof(float);
  session->prefill_gate_up_bytes =
      prefill_gate_up_rows * static_cast<uint64_t>(prefill_chunk) * sizeof(float);
  session->prefill_ff_bytes =
      intermediate * static_cast<uint64_t>(prefill_chunk) * sizeof(uint16_t);
  session->prefill_down_bytes =
      hidden * static_cast<uint64_t>(prefill_chunk) * sizeof(float);
  session->planned_weight_blocks = request->planned_weight_blocks;
  session->planned_gpu_resident_blocks = request->planned_gpu_resident_blocks;
  session->planned_gpu_staged_blocks = request->planned_gpu_staged_blocks;
  session->planned_weight_bytes = request->planned_weight_bytes;
  session->planned_gpu_resident_weight_bytes =
      request->planned_gpu_resident_weight_bytes;
  session->planned_gpu_staged_weight_bytes =
      request->planned_gpu_staged_weight_bytes;
  session->planned_weight_descriptor_count =
      request->planned_weight_descriptor_count;
  session->planned_weight_descriptor_hash = request->planned_weight_descriptor_hash;
  session->host_layouts = layouts;

  uint16_t *host_arena = nullptr;
  const uint64_t host_weight_bytes =
      descriptor_mode
          ? pinned_weight_staging_bytes(request, session->resident_weight_bytes)
          : session->arena_bytes;
  uint64_t setup_sync_calls = 0;
  int32_t failure_stage = kCreateStageHostWeightAlloc;
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_arena), host_weight_bytes,
                      cudaHostAllocDefault);
  if (err == cudaSuccess) {
    failure_stage = kCreateStageHostSlotsAlloc;
    err = cudaHostAlloc(reinterpret_cast<void **>(&session->host_slots),
                        session->slots_bytes, cudaHostAllocDefault);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDeviceArenaAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_arena),
                     session->arena_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDeviceLayoutsAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_layouts),
                     session->layout_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDeviceScratchAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_scratch),
                     session->scratch_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageProjectionInputAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_projection_input),
                     session->projection_input_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStagePrefillHiddenAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_hidden_a),
                     session->prefill_hidden_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_hidden_b),
                     session->prefill_hidden_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStagePrefillChunkAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_norm),
                     session->prefill_norm_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_qkv),
                     session->prefill_qkv_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_prefill_qkv_encoded),
        session->prefill_qkv_encoded_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_attn),
                     session->prefill_attn_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_o),
                     session->prefill_o_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_gate_up),
                     session->prefill_gate_up_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_ff),
                     session->prefill_ff_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_down),
                     session->prefill_down_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDecodeAttentionAlloc;
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_decode_attention_values),
        session->decode_attention_values_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_decode_attention_m),
                     session->decode_attention_stats_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_decode_attention_l),
                     session->decode_attention_stats_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDecodeSdpaAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_decode_q),
                     session->decode_q_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_decode_seq_len_q),
        sizeof(int32_t));
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_decode_seq_len_kv),
        sizeof(int32_t));
  }
  if (err == cudaSuccess && session->packed_qkv_bytes != 0) {
    failure_stage = kCreateStagePackedQkvAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_qkv_packed),
                     session->packed_qkv_bytes);
  }
  if (err == cudaSuccess && session->packed_gate_up_bytes != 0) {
    failure_stage = kCreateStagePackedGateUpAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_gate_up_packed),
                     session->packed_gate_up_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageKvKeysAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_kv_keys),
                     session->kv_bytes / 2);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageKvValuesAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_kv_values),
                     session->kv_bytes / 2);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_kv_block_table),
                     session->kv_block_table_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStagePromptTokensAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prompt_tokens),
                     session->prompt_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDeviceSlotsAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_slots),
                     session->slots_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDeviceStepAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_step),
                     sizeof(uint32_t));
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageCublasWorkspaceAlloc;
    err = cudaMalloc(&session->cublas_workspace, kCublasWorkspaceBytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageStreamCreate;
    err = cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageCublasCreate;
    err = cublas_to_cuda(cublasCreate(&session->cublas));
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageCublasLtCreate;
    err = cublas_to_cuda(cublasLtCreate(&session->cublas_lt));
  }
#if NERVA_HAVE_CUDNN_FRONTEND
  if (err == cudaSuccess) {
    failure_stage = kCreateStageCublasConfigure;
    err = cudnn_to_cuda(cudnnCreate(&session->cudnn));
  }
#endif
  if (err == cudaSuccess) {
    failure_stage = kCreateStageCublasConfigure;
    err = configure_cublas(session->cublas, session->stream,
                           session->cublas_workspace,
                           kCublasWorkspaceBytes);
  }
#if NERVA_HAVE_CUDNN_FRONTEND
  if (err == cudaSuccess) {
    err = cudnn_to_cuda(cudnnSetStream(session->cudnn, session->stream));
  }
#endif
  if (err == cudaSuccess) {
    failure_stage = kCreateStageStartEventCreate;
    err = cudaEventCreate(&session->device_start);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageStopEventCreate;
    err = cudaEventCreate(&session->device_stop);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageStartEventCreate;
    err = cudaEventCreate(&session->profile_start);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageStopEventCreate;
    err = cudaEventCreate(&session->profile_stop);
  }
  if (err != cudaSuccess) {
    fail(out, err, failure_stage);
    cudaFreeHost(host_arena);
    free_session_fields(session);
    delete session;
    return -1;
  }

  memset(host_arena, 0, host_weight_bytes);
  const uint64_t embedding_bytes = vocab_size * hidden * sizeof(uint16_t);
  const uint64_t scratch_gap_bytes = hidden * 2 * sizeof(uint16_t);
  if (!descriptor_mode) {
    memcpy(host_arena + session->arena_layout.embeddings, request->embeddings,
           vocab_size * hidden * sizeof(uint16_t));
    for (uint32_t index = 0; index < request->layer_count; ++index) {
      copy_layer(host_arena, layouts[index], request->layers[index], hidden,
                 attention_hidden, kv_hidden, request->head_dim, intermediate);
    }
    memcpy(host_arena + session->arena_layout.final_norm,
           request->final_norm_weight, hidden * sizeof(uint16_t));
    memcpy(host_arena + session->arena_layout.lm_head, request->lm_head,
           vocab_size * hidden * sizeof(uint16_t));
  }
  if (err == cudaSuccess && descriptor_mode) {
    failure_stage = kCreateStageDescriptorCopy;
    err = copy_weight_descriptors_to_device(
        session->device_arena, host_arena, host_weight_bytes, request,
        session->arena_bytes, embedding_bytes, scratch_gap_bytes,
        session->stream, out, &setup_sync_calls);
  } else if (err == cudaSuccess) {
    failure_stage = kCreateStageDescriptorCopy;
    err = cudaMemcpyAsync(session->device_arena, host_arena, session->arena_bytes,
                          cudaMemcpyHostToDevice, session->stream);
    out->h2d_bytes = session->arena_bytes;
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageLayoutCopy;
    err = cudaMemcpyAsync(session->device_layouts, layouts.data(),
                          session->layout_bytes, cudaMemcpyHostToDevice,
                          session->stream);
    out->h2d_bytes += session->layout_bytes;
  }
  if (err == cudaSuccess) {
    const uint32_t blocks =
        (session->kv_block_count + kDecodeThreads - 1u) / kDecodeThreads;
    hf_init_identity_kv_block_table_kernel<<<blocks, kDecodeThreads, 0,
                                             session->stream>>>(
        session->device_kv_block_table, session->kv_block_count);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStagePackReplicas;
    err = pack_session_weight_replicas(session);
  }
  if (err == cudaSuccess && use_cublas_layer_path(session)) {
    failure_stage = kCreateStageProjectionPlanAutotune;
    err = autotune_session_lt_gemv_plans(session);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageWarmCublas;
    err = warm_cublas_gemv(session->cublas, session->device_arena,
                           session->arena_layout, session->dtype,
                           session->device_scratch, session->stream);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageSetupSynchronize;
    err = cudaStreamSynchronize(session->stream);
  }
  cudaFreeHost(host_arena);
  if (err != cudaSuccess) {
    fail(out, err, failure_stage);
    free_session_fields(session);
    delete session;
    return -1;
  }
  session->h2d_bytes = out->h2d_bytes;
  session->load_staging_bytes = host_weight_bytes;
  session->setup_sync_calls = setup_sync_calls;
  session->descriptor_gpu_resident_h2d_bytes =
      out->descriptor_gpu_resident_h2d_bytes;
  session->descriptor_gpu_staged_h2d_bytes =
      out->descriptor_gpu_staged_h2d_bytes;
  fill_create_result(session, out);
  *session_out = session;
  return 0;
}

extern "C" int nerva_cuda_hf_decode_sequence_session_run(
    const NervaCudaHfDecodeSequenceSessionRunRequest *request,
    NervaCudaHfDecodeSequenceResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->prompt_tokens == nullptr || request->output_tokens == nullptr ||
      request->steps == 0 || request->prompt_token_count == 0 ||
      request->output_token_capacity < request->steps ||
      request->prompt_tokens[request->prompt_token_count - 1u] !=
          request->seed_token ||
      request->prompt_token_count > UINT32_MAX - request->steps + 1u) {
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  const uint32_t context_steps =
      request->prompt_token_count + request->steps - 1u;
  if (context_steps > session->max_context_tokens) {
    return -1;
  }
  for (uint32_t index = 0; index < request->prompt_token_count; ++index) {
    if (request->prompt_tokens[index] >= session->vocab_size) {
      return -1;
    }
  }
  out->device_count = 1;
  out->dtype = session->dtype;
  out->hidden = session->hidden;
  out->heads = session->heads;
  out->kv_heads = session->kv_heads;
  out->head_dim = session->head_dim;
  out->intermediate = session->intermediate;
  out->vocab_size = session->vocab_size;
  out->layer_count = session->layer_count;
  out->steps = request->steps;
  out->seed_token = request->seed_token;
  out->resident_weight_bytes = session->resident_weight_bytes;
  out->planned_weight_blocks = session->planned_weight_blocks;
  out->planned_gpu_resident_blocks = session->planned_gpu_resident_blocks;
  out->planned_gpu_staged_blocks = session->planned_gpu_staged_blocks;
  out->planned_weight_bytes = session->planned_weight_bytes;
  out->planned_gpu_resident_weight_bytes =
      session->planned_gpu_resident_weight_bytes;
  out->planned_gpu_staged_weight_bytes =
      session->planned_gpu_staged_weight_bytes;
  out->planned_weight_descriptor_count =
      session->planned_weight_descriptor_count;
  out->planned_weight_descriptor_hash =
      session->planned_weight_descriptor_hash;

  cudaError_t err = cudaMemsetAsync(session->device_slots, 0,
                                    session->slots_bytes, session->stream);
  if (err == cudaSuccess)
    err = cudaMemsetAsync(session->device_step, 0, sizeof(uint32_t),
                          session->stream);
  const uint64_t prompt_bytes =
      static_cast<uint64_t>(request->prompt_token_count) * sizeof(uint32_t);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(session->device_prompt_tokens, request->prompt_tokens,
                          prompt_bytes, cudaMemcpyHostToDevice, session->stream);
    out->h2d_bytes = prompt_bytes;
  }

  const bool graph_hit = err == cudaSuccess &&
                         session_graph_matches(session, context_steps,
                                               request->prompt_token_count,
                                               request->has_eos_token,
                                               request->eos_token, 0);
  if (graph_hit) {
    out->graph_nodes = session->cached_graph_nodes;
    out->graph_cache_hits = 1;
  }
  if (err == cudaSuccess && !graph_hit) {
    reset_session_graph(session);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool capture_started = false;
    err = cudaStreamBeginCapture(session->stream, cudaStreamCaptureModeGlobal);
    capture_started = err == cudaSuccess;
    if (err == cudaSuccess) {
      hf_decode_sequence_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
          session->device_arena, session->arena_layout, session->device_layouts,
          session->layer_count, session->dtype, session->hidden, session->heads,
          session->kv_heads, session->head_dim, session->intermediate, 0,
          session->device_step, context_steps, session->device_prompt_tokens,
          request->prompt_token_count, session->rms_eps, session->rope_theta,
          session->device_scratch, session->device_kv_keys,
          session->device_kv_values, session->kv_block_count,
          session->device_kv_block_table,
          session->device_slots);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
      float *device_logits = session->device_scratch + session->hidden * 2;
      err = final_head_gemv(session->cublas, session->device_arena,
                            session->arena_layout, session->dtype,
                            session->hidden, session->vocab_size,
                            device_logits);
    }
    if (err == cudaSuccess) {
      float *device_logits = session->device_scratch + session->hidden * 2;
      hf_decode_final_head_reduce_kernel<<<1, kDecodeSampleThreads, 0,
                                           session->stream>>>(
          session->device_step, context_steps, request->has_eos_token,
          request->eos_token, device_logits, session->vocab_size,
          session->device_slots);
      err = cudaGetLastError();
    }
    if (capture_started) {
      cudaError_t end_err = cudaStreamEndCapture(session->stream, &graph);
      if (err == cudaSuccess) {
        err = end_err;
      } else if (graph != nullptr) {
        cudaGraphDestroy(graph);
        graph = nullptr;
      }
    }
    if (err == cudaSuccess) {
      size_t graph_nodes = 0;
      err = cudaGraphGetNodes(graph, nullptr, &graph_nodes);
      out->graph_nodes = static_cast<uint64_t>(graph_nodes);
    }
    if (err == cudaSuccess) err = cudaGraphInstantiate(&graph_exec, graph, 0);
    if (err == cudaSuccess) {
      session->cached_graph = graph;
      session->cached_graph_exec = graph_exec;
      session->cached_context_steps = context_steps;
      session->cached_prompt_token_count = request->prompt_token_count;
      session->cached_has_eos_token = request->has_eos_token;
      session->cached_eos_token = request->eos_token;
      session->cached_attention_chunks = 0;
      session->cached_graph_nodes = out->graph_nodes;
      out->graph_captures = 1;
      graph = nullptr;
      graph_exec = nullptr;
    }
    if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
    if (graph != nullptr) cudaGraphDestroy(graph);
  }
  if (err == cudaSuccess) err = cudaEventRecord(session->device_start, session->stream);
  for (uint32_t step = 0; err == cudaSuccess && step < context_steps; ++step) {
    err = cudaGraphLaunch(session->cached_graph_exec, session->stream);
    if (err == cudaSuccess) {
      out->graph_replays += 1;
      out->graph_launches += 1;
      out->kernel_launches += out->graph_nodes == 0 ? 1 : out->graph_nodes;
    }
  }
  if (err == cudaSuccess) err = cudaEventRecord(session->device_stop, session->stream);
  const uint64_t slots_bytes =
      static_cast<uint64_t>(context_steps) * sizeof(NervaCudaSyntheticTokenSlot);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(session->host_slots, session->device_slots, slots_bytes,
                          cudaMemcpyDeviceToHost, session->stream);
    out->d2h_bytes = slots_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(session->stream);
    out->sync_calls += 1;
  }
  if (err == cudaSuccess) {
    float device_ms = 0.0f;
    err = cudaEventElapsedTime(&device_ms, session->device_start,
                               session->device_stop);
    if (err == cudaSuccess && device_ms > 0.0f) {
      out->device_elapsed_ns = static_cast<uint64_t>(device_ms * 1000000.0f);
      if (out->device_elapsed_ns == 0) out->device_elapsed_ns = 1;
    }
  }
  if (err == cudaSuccess) {
    out->observed_tokens =
        observed_count_for(request->steps, request->prompt_token_count,
                           request->has_eos_token, request->eos_token,
                           session->host_slots);
    const uint32_t output_start = request->prompt_token_count - 1u;
    for (uint32_t index = 0; index < out->observed_tokens; ++index) {
      request->output_tokens[index] =
          session->host_slots[output_start + index].token;
    }
    out->last_token = out->observed_tokens == 0
                          ? 0
                          : request->output_tokens[out->observed_tokens - 1];
    out->observed_token_hash =
        hash_tokens(request->output_tokens, out->observed_tokens);
    out->resident_kv_bytes = session->kv_bytes;
    out->kv_tokens = output_start + out->observed_tokens;
    out->device_arena_bytes = session_device_footprint(session);
    out->pinned_host_bytes = session->slots_bytes;
    out->host_causality_edges = 0;
    out->status = out->observed_tokens > 0 ? 0 : -1;
    for (uint32_t index = 0; out->status == 0 && index < out->observed_tokens;
         ++index) {
      const NervaCudaSyntheticTokenSlot &slot =
          session->host_slots[output_start + index];
      if (slot.request_id != kRequestId || slot.sequence_id != kSequenceId ||
          slot.completion != kCompletionDeviceComplete ||
          slot.token_index != output_start + index) {
        out->status = -1;
      }
    }
  } else {
    fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_hf_decode_sequence_session_start(
    const NervaCudaHfDecodeSequenceSessionStartRequest *request,
    NervaCudaHfDecodeSequenceResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->prompt_tokens == nullptr || request->prompt_token_count == 0) {
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  if (request->prompt_token_count > session->max_context_tokens) {
    return -1;
  }
  for (uint32_t index = 0; index < request->prompt_token_count; ++index) {
    if (request->prompt_tokens[index] >= session->vocab_size) {
      return -1;
    }
  }

  fill_session_result_header(
      session, out, 0, request->prompt_tokens[request->prompt_token_count - 1u]);
  session->pending_prefill_available = 0;
  cudaError_t err = cudaMemsetAsync(session->device_slots, 0,
                                    session->slots_bytes, session->stream);
  if (err == cudaSuccess)
    err = cudaMemsetAsync(session->device_step, 0, sizeof(uint32_t),
                          session->stream);
  const uint64_t prompt_bytes =
      static_cast<uint64_t>(request->prompt_token_count) * sizeof(uint32_t);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(session->device_prompt_tokens, request->prompt_tokens,
                          prompt_bytes, cudaMemcpyHostToDevice, session->stream);
    out->h2d_bytes = prompt_bytes;
  }
  if (err == cudaSuccess) {
    err = use_cublas_layer_path(session)
              ? launch_cublas_session_prefill(
                    session, request->prompt_token_count,
                    request->has_eos_token, request->eos_token, out)
              : launch_serial_session_prefill(
                    session, request->prompt_token_count,
                    request->has_eos_token, request->eos_token, out);
  }
  if (err == cudaSuccess) {
    stash_prefill_metrics(session, out);
    session->active_prompt_token_count = request->prompt_token_count;
    session->active_has_eos_token = request->has_eos_token;
    session->active_eos_token = request->eos_token;
    session->active_seed_token = request->prompt_tokens[request->prompt_token_count - 1u];
    session->active_observed_tokens = 0;
    session->active_cursor = request->prompt_token_count;
    session->active_started = true;
    session->active_finished = false;
    out->resident_kv_bytes = session->kv_bytes;
    out->kv_tokens = request->prompt_token_count;
    out->device_arena_bytes = session_device_footprint(session);
    out->pinned_host_bytes = session->slots_bytes;
    out->status = 0;
  } else {
    fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_hf_decode_sequence_session_advance(
    const NervaCudaHfDecodeSequenceSessionAdvanceRequest *request,
    NervaCudaHfDecodeSequenceResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->output_tokens == nullptr || request->steps == 0 ||
      request->output_token_capacity < request->steps) {
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  if (!session->active_started || session->active_finished ||
      session->active_prompt_token_count == 0) {
    return -1;
  }
  const uint32_t prompt_count = session->active_prompt_token_count;
  const uint32_t slot_start = prompt_count - 1u + session->active_observed_tokens;
  const uint32_t target_cursor =
      prompt_count + session->active_observed_tokens + request->steps - 1u;
  if (target_cursor > session->max_context_tokens ||
      target_cursor < session->active_cursor) {
    return -1;
  }
  const uint32_t run_count = target_cursor - session->active_cursor;
  const uint32_t seed_token =
      session->active_observed_tokens == 0
          ? session->active_seed_token
          : session->host_slots[slot_start - 1u].token;
  fill_session_result_header(session, out, request->steps, seed_token);

  cudaError_t err = cudaSuccess;
  if (run_count != 0) {
    const uint32_t attention_chunks =
        decode_attention_chunks_for_cursor(session, target_cursor);
    err = ensure_session_graph(session, session->max_context_tokens, prompt_count,
                               session->active_has_eos_token,
                               session->active_eos_token, attention_chunks,
                               session->active_cursor, out);
  }
  if (err == cudaSuccess && run_count != 0)
    err = cudaEventRecord(session->device_start, session->stream);
  for (uint32_t step = 0; err == cudaSuccess && step < run_count; ++step) {
    err = cudaGraphLaunch(session->cached_graph_exec, session->stream);
    if (err == cudaSuccess) {
      out->graph_replays += 1;
      out->graph_launches += 1;
      out->kernel_launches += out->graph_nodes == 0 ? 1 : out->graph_nodes;
    }
  }
  if (err == cudaSuccess && run_count != 0)
    err = cudaEventRecord(session->device_stop, session->stream);
  const uint64_t slots_bytes =
      static_cast<uint64_t>(request->steps) * sizeof(NervaCudaSyntheticTokenSlot);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(session->host_slots + slot_start,
                          session->device_slots + slot_start, slots_bytes,
                          cudaMemcpyDeviceToHost, session->stream);
    out->d2h_bytes = slots_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(session->stream);
    out->sync_calls += 1;
  }
  if (err == cudaSuccess && run_count != 0) {
    float device_ms = 0.0f;
    err = cudaEventElapsedTime(&device_ms, session->device_start,
                               session->device_stop);
    if (err == cudaSuccess && device_ms > 0.0f) {
      out->device_elapsed_ns += static_cast<uint64_t>(device_ms * 1000000.0f);
      if (out->device_elapsed_ns == 0) out->device_elapsed_ns = 1;
    }
  }
  if (err == cudaSuccess) {
    NervaCudaSyntheticTokenSlot *observed_slots = session->host_slots + slot_start;
    out->observed_tokens = observed_from_slot_range(
        request->steps, session->active_has_eos_token, session->active_eos_token,
        observed_slots);
    for (uint32_t index = 0; index < out->observed_tokens; ++index) {
      request->output_tokens[index] = observed_slots[index].token;
    }
    out->last_token = out->observed_tokens == 0
                          ? 0
                          : request->output_tokens[out->observed_tokens - 1];
    out->observed_token_hash =
        hash_tokens(request->output_tokens, out->observed_tokens);
    out->resident_kv_bytes = session->kv_bytes;
    out->kv_tokens = slot_start + out->observed_tokens;
    out->device_arena_bytes = session_device_footprint(session);
    out->pinned_host_bytes = session->slots_bytes;
    out->host_causality_edges = 0;
    out->status = out->observed_tokens > 0 ? 0 : -1;
    for (uint32_t index = 0; out->status == 0 && index < out->observed_tokens;
         ++index) {
      const NervaCudaSyntheticTokenSlot &slot = observed_slots[index];
      if (slot.request_id != kRequestId || slot.sequence_id != kSequenceId ||
          slot.completion != kCompletionDeviceComplete ||
          slot.token_index != slot_start + index) {
        out->status = -1;
      }
    }
    if (out->status == 0) {
      scale_profile_counters(out, out->observed_tokens);
      session->active_observed_tokens += out->observed_tokens;
      session->active_cursor =
          out->observed_tokens < request->steps ? session->max_context_tokens
                                                : target_cursor;
      session->active_finished = out->observed_tokens < request->steps ||
                                 out->kv_tokens >= session->max_context_tokens;
    }
  } else {
    fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_hf_decode_sequence_projection_batch_plan(
    const NervaCudaHfDecodeSequenceProjectionBatchPlanRequest *request,
    NervaCudaHfDecodeSequenceProjectionBatchPlanResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->reason = kProjectionBatchPlanInvalidRequest;
  if (request == nullptr) {
    return -1;
  }
  out->requested_session_count = request->session_count;
  out->target_block_tokens =
      request->target_block_tokens == 0 ? 1u : request->target_block_tokens;
  out->min_block_tokens =
      request->min_block_tokens == 0 ? 1u : request->min_block_tokens;
  if (request->session_count == 0) {
    out->reason = kProjectionBatchPlanNoSessions;
    out->status = 0;
    return 0;
  }
  if (request->sessions == nullptr) {
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  std::vector<NervaCudaHfDecodeSequenceSession *> ready;
  ready.reserve(request->session_count);
  for (uint32_t index = 0; index < request->session_count; ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (session == nullptr || !session->active_started ||
        session->active_finished || session->active_prompt_token_count == 0 ||
        session->active_cursor >= session->max_context_tokens) {
      continue;
    }
    ready.push_back(session);
  }

  if (ready.empty()) {
    out->reason = kProjectionBatchPlanNoReadySessions;
    out->status = 0;
    return 0;
  }

  auto same_projection_model =
      [](const NervaCudaHfDecodeSequenceSession *lhs,
         const NervaCudaHfDecodeSequenceSession *rhs) {
        return lhs->planned_weight_descriptor_hash != 0 &&
               rhs->planned_weight_descriptor_hash != 0 &&
               lhs->planned_weight_descriptor_hash ==
                   rhs->planned_weight_descriptor_hash &&
               lhs->dtype == rhs->dtype && lhs->hidden == rhs->hidden &&
               lhs->heads == rhs->heads && lhs->kv_heads == rhs->kv_heads &&
               lhs->head_dim == rhs->head_dim &&
               lhs->intermediate == rhs->intermediate &&
               lhs->vocab_size == rhs->vocab_size &&
               lhs->layer_count == rhs->layer_count &&
               lhs->resident_weight_bytes == rhs->resident_weight_bytes;
      };

  bool any_hash = false;
  for (const NervaCudaHfDecodeSequenceSession *session : ready) {
    any_hash = any_hash || session->planned_weight_descriptor_hash != 0;
  }
  if (!any_hash) {
    out->reason = kProjectionBatchPlanSharedWeightsUnproven;
    out->status = 0;
    return 0;
  }

  NervaCudaHfDecodeSequenceSession *best = nullptr;
  uint32_t best_count = 0;
  for (NervaCudaHfDecodeSequenceSession *candidate : ready) {
    if (candidate->planned_weight_descriptor_hash == 0) {
      continue;
    }
    uint32_t compatible = 0;
    for (NervaCudaHfDecodeSequenceSession *other : ready) {
      if (same_projection_model(candidate, other)) {
        compatible += 1;
      }
    }
    if (compatible > best_count) {
      best = candidate;
      best_count = compatible;
    }
  }

  out->eligible_session_count = best_count;
  if (best == nullptr || best_count < out->min_block_tokens) {
    out->reason = kProjectionBatchPlanInsufficientCompatibleReady;
    out->status = 0;
    return 0;
  }

  const uint32_t block_tokens =
      std::min(best_count, out->target_block_tokens);
  const uint64_t attention_hidden =
      static_cast<uint64_t>(best->heads) * best->head_dim;
  const uint64_t kv_hidden =
      static_cast<uint64_t>(best->kv_heads) * best->head_dim;
  const PackedProjectionShape packed_shape = packed_projection_shape(
      best->hidden, attention_hidden, kv_hidden, best->intermediate);
  const uint64_t hidden = best->hidden;
  const uint64_t intermediate = best->intermediate;
  const uint64_t vocab_size = best->vocab_size;
  const uint64_t token_u16 = static_cast<uint64_t>(block_tokens) * sizeof(uint16_t);
  const uint64_t token_f32 = static_cast<uint64_t>(block_tokens) * sizeof(float);
  const uint64_t max_input_cols =
      std::max<uint64_t>(hidden, std::max<uint64_t>(attention_hidden, intermediate));
  const uint64_t max_output_rows =
      std::max<uint64_t>(
          vocab_size,
          std::max<uint64_t>(packed_shape.qkv_rows,
                             std::max<uint64_t>(packed_shape.gate_up_rows,
                                                hidden)));

  out->reason = kProjectionBatchPlanReady;
  out->exact = 1;
  out->block_tokens = block_tokens;
  out->dtype = best->dtype;
  out->hidden = best->hidden;
  out->heads = best->heads;
  out->kv_heads = best->kv_heads;
  out->head_dim = best->head_dim;
  out->intermediate = best->intermediate;
  out->vocab_size = best->vocab_size;
  out->layer_count = best->layer_count;
  out->max_context_tokens = best->max_context_tokens;
  out->planned_weight_descriptor_hash = best->planned_weight_descriptor_hash;
  out->resident_weight_bytes = best->resident_weight_bytes;
  out->qkv_rows = packed_shape.qkv_rows;
  out->gate_up_rows = packed_shape.gate_up_rows;
  out->qkv_input_bytes = hidden * token_u16;
  out->qkv_output_bytes = packed_shape.qkv_rows * token_f32;
  out->attention_output_input_bytes = attention_hidden * token_u16;
  out->attention_output_output_bytes = hidden * token_f32;
  out->gate_up_input_bytes = hidden * token_u16;
  out->gate_up_output_bytes = packed_shape.gate_up_rows * token_f32;
  out->down_input_bytes = intermediate * token_u16;
  out->down_output_bytes = hidden * token_f32;
  out->lm_head_input_bytes = hidden * token_u16;
  out->lm_head_output_bytes = vocab_size * token_f32;
  out->pack_input_bytes = max_input_cols * token_u16;
  out->max_projection_output_bytes = max_output_rows * token_f32;
  out->hot_path_allocations = 0;
  out->status = 0;
  return 0;
}

extern "C" int nerva_cuda_hf_decode_sequence_projection_batch_execute(
    const NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest *request,
    NervaCudaHfDecodeSequenceProjectionBatchExecuteResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->reason = kProjectionBatchPlanInvalidRequest;
  if (request == nullptr) {
    return -1;
  }
  out->requested_session_count = request->session_count;
  out->target_block_tokens =
      request->target_block_tokens == 0 ? 1u : request->target_block_tokens;
  out->min_block_tokens =
      request->min_block_tokens == 0 ? 1u : request->min_block_tokens;
  out->projection_kind = request->projection_kind;
  out->layer_index = request->layer_index;
  if (request->session_count == 0) {
    out->reason = kProjectionBatchPlanNoSessions;
    out->status = 0;
    return 0;
  }
  if (request->sessions == nullptr) {
    return -1;
  }
  const bool layer_projection =
      request->projection_kind == kProjectionBatchKindQkv ||
      request->projection_kind == kProjectionBatchKindAttentionOutput ||
      request->projection_kind == kProjectionBatchKindGateUp ||
      request->projection_kind == kProjectionBatchKindDown;
  const bool lm_head_projection =
      request->projection_kind == kProjectionBatchKindLmHead;
  if (!layer_projection && !lm_head_projection) {
    out->reason = kProjectionBatchPlanUnsupportedProjection;
    out->status = 0;
    return 0;
  }
  if (lm_head_projection && request->layer_index != 0) {
    out->reason = kProjectionBatchPlanInvalidLayer;
    out->status = 0;
    return 0;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  auto ready_session = [](const NervaCudaHfDecodeSequenceSession *session) {
    return session != nullptr && session->active_started &&
           !session->active_finished && session->active_prompt_token_count != 0 &&
           session->active_cursor < session->max_context_tokens &&
           use_cublas_layer_path(session);
  };
  auto same_projection_model =
      [](const NervaCudaHfDecodeSequenceSession *lhs,
         const NervaCudaHfDecodeSequenceSession *rhs) {
        return lhs->planned_weight_descriptor_hash != 0 &&
               rhs->planned_weight_descriptor_hash != 0 &&
               lhs->planned_weight_descriptor_hash ==
                   rhs->planned_weight_descriptor_hash &&
               lhs->dtype == rhs->dtype && lhs->hidden == rhs->hidden &&
               lhs->heads == rhs->heads && lhs->kv_heads == rhs->kv_heads &&
               lhs->head_dim == rhs->head_dim &&
               lhs->intermediate == rhs->intermediate &&
               lhs->vocab_size == rhs->vocab_size &&
               lhs->layer_count == rhs->layer_count &&
               lhs->resident_weight_bytes == rhs->resident_weight_bytes;
      };

  uint32_t ready_count = 0;
  bool any_hash = false;
  for (uint32_t index = 0; index < request->session_count; ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (!ready_session(session)) {
      continue;
    }
    ready_count += 1;
    any_hash = any_hash || session->planned_weight_descriptor_hash != 0;
  }
  if (ready_count == 0) {
    out->reason = kProjectionBatchPlanNoReadySessions;
    out->status = 0;
    return 0;
  }
  if (!any_hash) {
    out->reason = kProjectionBatchPlanSharedWeightsUnproven;
    out->status = 0;
    return 0;
  }

  NervaCudaHfDecodeSequenceSession *best = nullptr;
  uint32_t best_count = 0;
  for (uint32_t candidate_index = 0; candidate_index < request->session_count;
       ++candidate_index) {
    NervaCudaHfDecodeSequenceSession *candidate =
        request->sessions[candidate_index];
    if (!ready_session(candidate) ||
        candidate->planned_weight_descriptor_hash == 0 ||
        (layer_projection && request->layer_index >= candidate->layer_count)) {
      continue;
    }
    uint32_t compatible = 0;
    for (uint32_t other_index = 0; other_index < request->session_count;
         ++other_index) {
      NervaCudaHfDecodeSequenceSession *other = request->sessions[other_index];
      if (ready_session(other) && same_projection_model(candidate, other)) {
        compatible += 1;
      }
    }
    if (compatible > best_count) {
      best = candidate;
      best_count = compatible;
    }
  }

  out->eligible_session_count = best_count;
  if (best == nullptr || best_count < out->min_block_tokens) {
    out->reason = kProjectionBatchPlanInsufficientCompatibleReady;
    out->status = 0;
    return 0;
  }
  if (layer_projection && request->layer_index >= best->layer_count) {
    out->reason = kProjectionBatchPlanInvalidLayer;
    out->status = 0;
    return 0;
  }

  uint32_t block_tokens = 0;
  for (uint32_t index = 0; index < request->session_count; ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (ready_session(session) && same_projection_model(best, session)) {
      block_tokens += 1;
      if (block_tokens >= out->target_block_tokens) {
        break;
      }
    }
  }
  if (block_tokens < out->min_block_tokens) {
    out->reason = kProjectionBatchPlanInsufficientCompatibleReady;
    out->status = 0;
    return 0;
  }

  const uint32_t attention_hidden = best->heads * best->head_dim;
  const uint32_t kv_hidden = best->kv_heads * best->head_dim;
  const PackedProjectionShape packed_shape = packed_projection_shape(
      best->hidden, attention_hidden, kv_hidden, best->intermediate);
  uint32_t rows = 0;
  uint32_t cols = 0;
  const uint16_t *matrix = nullptr;
  float *batch_output = nullptr;
  uint64_t batch_output_capacity = 0;
  const SequenceLayerLayout layout =
      layer_projection ? best->host_layouts[request->layer_index]
                       : SequenceLayerLayout{};
  switch (request->projection_kind) {
    case kProjectionBatchKindQkv:
      rows = static_cast<uint32_t>(packed_shape.qkv_rows);
      cols = best->hidden;
      matrix = best->device_qkv_packed +
               packed_shape.qkv_elements_per_layer * request->layer_index;
      batch_output = best->device_prefill_qkv;
      batch_output_capacity = best->prefill_qkv_bytes;
      break;
    case kProjectionBatchKindAttentionOutput:
      rows = best->hidden;
      cols = attention_hidden;
      matrix = best->device_arena + layout.w_o;
      batch_output = best->device_prefill_o;
      batch_output_capacity = best->prefill_o_bytes;
      break;
    case kProjectionBatchKindGateUp:
      rows = static_cast<uint32_t>(packed_shape.gate_up_rows);
      cols = best->hidden;
      matrix = best->device_gate_up_packed +
               packed_shape.gate_up_elements_per_layer * request->layer_index;
      batch_output = best->device_prefill_gate_up;
      batch_output_capacity = best->prefill_gate_up_bytes;
      break;
    case kProjectionBatchKindDown:
      rows = best->hidden;
      cols = best->intermediate;
      matrix = best->device_arena + layout.w_down;
      batch_output = best->device_prefill_down;
      batch_output_capacity = best->prefill_down_bytes;
      break;
    case kProjectionBatchKindLmHead:
      rows = best->vocab_size;
      cols = best->hidden;
      matrix = best->device_arena + best->arena_layout.lm_head;
      batch_output = best->device_prefill_gate_up;
      batch_output_capacity = best->prefill_gate_up_bytes;
      break;
    default:
      break;
  }
  const uint64_t input_bytes =
      static_cast<uint64_t>(cols) * block_tokens * sizeof(uint16_t);
  const uint64_t output_bytes =
      static_cast<uint64_t>(rows) * block_tokens * sizeof(float);
  if (rows == 0 || cols == 0 || matrix == nullptr || batch_output == nullptr ||
      best->device_prefill_norm == nullptr ||
      best->prefill_norm_bytes < input_bytes ||
      batch_output_capacity < output_bytes) {
    out->reason = kProjectionBatchPlanInsufficientScratch;
    out->status = 0;
    return 0;
  }

  uint16_t *batch_input = best->device_prefill_norm;
  uint32_t selected_index = 0;
  for (uint32_t index = 0; index < request->session_count &&
                           selected_index < block_tokens;
       ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (!ready_session(session) || !same_projection_model(best, session)) {
      continue;
    }
    selected_index += 1;
    if (session != best) {
      err = cudaStreamSynchronize(session->stream);
      out->sync_calls += 1;
      if (err != cudaSuccess) {
        out->cuda_error = static_cast<int32_t>(err);
        return -1;
      }
    }
  }

  err = cudaEventRecord(best->device_start, best->stream);
  const uint32_t pack_blocks = ceil_div_u32(cols, kDecodeThreads);
  selected_index = 0;
  for (uint32_t index = 0; err == cudaSuccess && index < request->session_count &&
                           selected_index < block_tokens;
       ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (!ready_session(session) || !same_projection_model(best, session)) {
      continue;
    }
    hf_projection_batch_pack_u16_kernel<<<pack_blocks, kDecodeThreads, 0,
                                          best->stream>>>(
        session->device_projection_input, batch_input, cols, selected_index);
    err = cudaGetLastError();
    out->pack_kernel_launches += 1;
    selected_index += 1;
  }

  if (err == cudaSuccess) {
    err = project_encoded_rows(best, nullptr, matrix, batch_input, rows, cols,
                               block_tokens, best->dtype, 0.0f, batch_output);
    out->projection_kernel_launches += 1;
  }

  const uint32_t scatter_blocks = ceil_div_u32(rows, kDecodeThreads);
  selected_index = 0;
  for (uint32_t index = 0; err == cudaSuccess && index < request->session_count &&
                           selected_index < block_tokens;
       ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (!ready_session(session) || !same_projection_model(best, session)) {
      continue;
    }
    LayerScratch scratch = layer_scratch_ptrs(
        session->device_scratch, session->hidden, attention_hidden, kv_hidden,
        session->intermediate);
    float *scatter_dst = nullptr;
    switch (request->projection_kind) {
      case kProjectionBatchKindQkv:
        scatter_dst = scratch.q;
        break;
      case kProjectionBatchKindAttentionOutput:
        scatter_dst = scratch.residual;
        break;
      case kProjectionBatchKindGateUp:
        scatter_dst = scratch.gate;
        break;
      case kProjectionBatchKindDown:
        scatter_dst = scratch.down;
        break;
      case kProjectionBatchKindLmHead:
        scatter_dst = session->device_scratch + session->hidden * 2;
        break;
      default:
        break;
    }
    if (scatter_dst == nullptr) {
      err = cudaErrorInvalidValue;
      break;
    }
    hf_projection_batch_scatter_f32_kernel<<<scatter_blocks, kDecodeThreads, 0,
                                             best->stream>>>(
        batch_output, scatter_dst, rows, selected_index);
    err = cudaGetLastError();
    out->scatter_kernel_launches += 1;
    selected_index += 1;
  }
  if (err == cudaSuccess) {
    err = cudaEventRecord(best->device_stop, best->stream);
  }
  if (err == cudaSuccess) {
    err = cudaEventSynchronize(best->device_stop);
    out->sync_calls += 1;
  }
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, best->device_start,
                             best->device_stop);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }
  out->reason = kProjectionBatchPlanReady;
  out->exact = 1;
  out->block_tokens = block_tokens;
  out->dtype = best->dtype;
  out->rows = rows;
  out->cols = cols;
  out->input_bytes = input_bytes;
  out->output_bytes = output_bytes;
  out->elapsed_ns = elapsed_ms <= 0.0f
                        ? 1
                        : static_cast<uint64_t>(elapsed_ms * 1000000.0f);
  out->hot_path_allocations = 0;
  out->status = 0;
  return 0;
}

extern "C" int nerva_cuda_hf_decode_sequence_layer_projection_batch_execute(
    const NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteRequest *request,
    NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->reason = kProjectionBatchPlanInvalidRequest;
  if (request == nullptr) {
    return -1;
  }

  out->requested_session_count = request->session_count;
  out->target_block_tokens =
      request->target_block_tokens == 0 ? 1u : request->target_block_tokens;
  out->min_block_tokens =
      request->min_block_tokens == 0 ? 1u : request->min_block_tokens;
  out->layer_index = request->layer_index;
  if (request->session_count == 0) {
    out->reason = kProjectionBatchPlanNoSessions;
    out->status = 0;
    return 0;
  }
  if (request->sessions == nullptr) {
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  auto ready_session = [](const NervaCudaHfDecodeSequenceSession *session) {
    return session != nullptr && session->active_started &&
           !session->active_finished && session->active_prompt_token_count != 0 &&
           session->active_cursor < session->max_context_tokens &&
           use_cublas_layer_path(session);
  };
  auto same_projection_model =
      [](const NervaCudaHfDecodeSequenceSession *lhs,
         const NervaCudaHfDecodeSequenceSession *rhs) {
        return lhs->planned_weight_descriptor_hash != 0 &&
               rhs->planned_weight_descriptor_hash != 0 &&
               lhs->planned_weight_descriptor_hash ==
                   rhs->planned_weight_descriptor_hash &&
               lhs->dtype == rhs->dtype && lhs->hidden == rhs->hidden &&
               lhs->heads == rhs->heads && lhs->kv_heads == rhs->kv_heads &&
               lhs->head_dim == rhs->head_dim &&
               lhs->intermediate == rhs->intermediate &&
               lhs->vocab_size == rhs->vocab_size &&
               lhs->layer_count == rhs->layer_count &&
               lhs->resident_weight_bytes == rhs->resident_weight_bytes;
      };

  uint32_t ready_count = 0;
  bool any_hash = false;
  for (uint32_t index = 0; index < request->session_count; ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (!ready_session(session)) {
      continue;
    }
    ready_count += 1;
    any_hash = any_hash || session->planned_weight_descriptor_hash != 0;
  }
  if (ready_count == 0) {
    out->reason = kProjectionBatchPlanNoReadySessions;
    out->status = 0;
    return 0;
  }
  if (!any_hash) {
    out->reason = kProjectionBatchPlanSharedWeightsUnproven;
    out->status = 0;
    return 0;
  }

  NervaCudaHfDecodeSequenceSession *best = nullptr;
  uint32_t best_count = 0;
  for (uint32_t candidate_index = 0; candidate_index < request->session_count;
       ++candidate_index) {
    NervaCudaHfDecodeSequenceSession *candidate =
        request->sessions[candidate_index];
    if (!ready_session(candidate) ||
        candidate->planned_weight_descriptor_hash == 0 ||
        request->layer_index >= candidate->layer_count) {
      continue;
    }
    uint32_t compatible = 0;
    for (uint32_t other_index = 0; other_index < request->session_count;
         ++other_index) {
      NervaCudaHfDecodeSequenceSession *other = request->sessions[other_index];
      if (ready_session(other) && same_projection_model(candidate, other)) {
        compatible += 1;
      }
    }
    if (compatible > best_count) {
      best = candidate;
      best_count = compatible;
    }
  }
  out->eligible_session_count = best_count;
  if (best == nullptr || best_count < out->min_block_tokens) {
    out->reason = kProjectionBatchPlanInsufficientCompatibleReady;
    out->status = 0;
    return 0;
  }
  if (request->layer_index >= best->layer_count) {
    out->reason = kProjectionBatchPlanInvalidLayer;
    out->status = 0;
    return 0;
  }

  std::vector<NervaCudaHfDecodeSequenceSession *> selected;
  selected.reserve(out->target_block_tokens);
  for (uint32_t index = 0; index < request->session_count &&
                           selected.size() < out->target_block_tokens;
       ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (ready_session(session) && same_projection_model(best, session)) {
      selected.push_back(session);
    }
  }
  if (selected.size() < out->min_block_tokens) {
    out->reason = kProjectionBatchPlanInsufficientCompatibleReady;
    out->status = 0;
    return 0;
  }
  out->block_tokens = static_cast<uint32_t>(selected.size());
  out->dtype = best->dtype;

  for (NervaCudaHfDecodeSequenceSession *session : selected) {
    if (session == best) {
      continue;
    }
    err = cudaStreamSynchronize(session->stream);
    out->sync_calls += 1;
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  auto run_stage =
      [&](uint32_t projection_kind,
          NervaCudaHfDecodeSequenceProjectionBatchExecuteResult *stage_out)
          -> int {
    NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest stage_request{};
    stage_request.sessions = request->sessions;
    stage_request.session_count = request->session_count;
    stage_request.target_block_tokens = request->target_block_tokens;
    stage_request.min_block_tokens = request->min_block_tokens;
    stage_request.projection_kind = projection_kind;
    stage_request.layer_index = request->layer_index;
    const int rc = nerva_cuda_hf_decode_sequence_projection_batch_execute(
        &stage_request, stage_out);
    out->cuda_error = stage_out->cuda_error;
    out->device_count = stage_out->device_count;
    out->reason = stage_out->reason;
    out->eligible_session_count = stage_out->eligible_session_count;
    out->block_tokens = stage_out->block_tokens;
    out->target_block_tokens = stage_out->target_block_tokens;
    out->min_block_tokens = stage_out->min_block_tokens;
    out->dtype = stage_out->dtype;
    return rc;
  };

  auto launch_attention_encode =
      [&](NervaCudaHfDecodeSequenceSession *session) -> cudaError_t {
    const uint32_t attention_hidden = session->heads * session->head_dim;
    const uint32_t kv_hidden = session->kv_heads * session->head_dim;
    const uint32_t decode_head_threads = decode_head_threads_for_session(session);
    const uint32_t attention_chunks =
        decode_attention_chunks_for_cursor(session, session->active_cursor);
    const SequenceLayerLayout layout =
        session->host_layouts[request->layer_index];
    const uint32_t max_steps = session->max_context_tokens;
    if (attention_chunks == 0) {
      hf_layer_qkv_attention_encode_kernel<<<session->heads,
                                             decode_head_threads, 0,
                                             best->stream>>>(
          session->device_arena, layout, request->layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          session->device_projection_input);
      out->dependency_kernel_launches += 1;
      return cudaGetLastError();
    }

    hf_layer_qkv_prepare_kernel<<<session->heads, decode_head_threads, 0,
                                  best->stream>>>(
        session->device_arena, layout, request->layer_index, session->dtype,
        session->hidden, session->heads, session->kv_heads, session->head_dim,
        session->intermediate, session->device_step, max_steps,
        session->rms_eps, session->rope_theta, session->device_scratch,
        session->device_kv_keys, session->device_kv_values,
        session->kv_block_count, session->device_kv_block_table, nullptr,
        nullptr, nullptr);
    out->dependency_kernel_launches += 1;
    cudaError_t local_err = cudaGetLastError();
    if (local_err != cudaSuccess) {
      return local_err;
    }

    const uint32_t query_group = session->heads / session->kv_heads;
    const bool use_shared_warp_gqa =
        query_group == kGroupedGqaHeads &&
        session->heads % session->kv_heads == 0 &&
        session->head_dim <= kSharedWarpGqaHeadDimMax;
    const bool use_grouped_gqa =
        query_group == kGroupedGqaHeads &&
        session->heads % session->kv_heads == 0 &&
        session->head_dim <= kGroupedGqaHeadDimMax;
    const dim3 grid((use_shared_warp_gqa || use_grouped_gqa) ? session->kv_heads
                                                             : session->heads,
                    attention_chunks);
    if (session->dtype == kDTypeBF16 && use_shared_warp_gqa) {
      hf_layer_shared_warp_gqa_attention_chunk_kernel<kDTypeBF16>
          <<<grid, kSharedWarpGqaThreads, 0, best->stream>>>(
              request->layer_index, session->hidden, session->heads,
              session->kv_heads, session->head_dim, session->intermediate,
              session->device_step, max_steps, attention_chunks,
              session->device_scratch, session->device_kv_keys,
              session->device_kv_values, session->device_decode_attention_values,
              session->device_decode_attention_m, session->device_decode_attention_l,
              session->kv_block_count, session->device_kv_block_table);
    } else if (session->dtype == kDTypeBF16 && use_grouped_gqa) {
      hf_layer_grouped_gqa_attention_chunk_kernel<kDTypeBF16>
          <<<grid, kGroupedGqaThreads, 0, best->stream>>>(
              request->layer_index, session->hidden, session->heads,
              session->kv_heads, session->head_dim, session->intermediate,
              session->device_step, max_steps, attention_chunks,
              session->device_scratch, session->device_kv_keys,
              session->device_kv_values, session->device_decode_attention_values,
              session->device_decode_attention_m, session->device_decode_attention_l,
              session->kv_block_count, session->device_kv_block_table);
    } else if (session->dtype == kDTypeBF16) {
      hf_layer_attention_chunk_kernel<kDTypeBF16>
          <<<grid, decode_head_threads, 0, best->stream>>>(
              request->layer_index, session->hidden, session->heads,
              session->kv_heads, session->head_dim, session->intermediate,
              session->device_step, max_steps, attention_chunks,
              session->device_scratch, session->device_kv_keys,
              session->device_kv_values, session->device_decode_attention_values,
              session->device_decode_attention_m, session->device_decode_attention_l,
              session->kv_block_count, session->device_kv_block_table);
    } else if (use_shared_warp_gqa) {
      hf_layer_shared_warp_gqa_attention_chunk_kernel<kDTypeF16>
          <<<grid, kSharedWarpGqaThreads, 0, best->stream>>>(
              request->layer_index, session->hidden, session->heads,
              session->kv_heads, session->head_dim, session->intermediate,
              session->device_step, max_steps, attention_chunks,
              session->device_scratch, session->device_kv_keys,
              session->device_kv_values, session->device_decode_attention_values,
              session->device_decode_attention_m, session->device_decode_attention_l,
              session->kv_block_count, session->device_kv_block_table);
    } else if (use_grouped_gqa) {
      hf_layer_grouped_gqa_attention_chunk_kernel<kDTypeF16>
          <<<grid, kGroupedGqaThreads, 0, best->stream>>>(
              request->layer_index, session->hidden, session->heads,
              session->kv_heads, session->head_dim, session->intermediate,
              session->device_step, max_steps, attention_chunks,
              session->device_scratch, session->device_kv_keys,
              session->device_kv_values, session->device_decode_attention_values,
              session->device_decode_attention_m, session->device_decode_attention_l,
              session->kv_block_count, session->device_kv_block_table);
    } else {
      hf_layer_attention_chunk_kernel<kDTypeF16>
          <<<grid, decode_head_threads, 0, best->stream>>>(
              request->layer_index, session->hidden, session->heads,
              session->kv_heads, session->head_dim, session->intermediate,
              session->device_step, max_steps, attention_chunks,
              session->device_scratch, session->device_kv_keys,
              session->device_kv_values, session->device_decode_attention_values,
              session->device_decode_attention_m, session->device_decode_attention_l,
              session->kv_block_count, session->device_kv_block_table);
    }
    out->dependency_kernel_launches += 1;
    local_err = cudaGetLastError();
    if (local_err != cudaSuccess) {
      return local_err;
    }
    const size_t reduce_shared_bytes =
        static_cast<size_t>(attention_chunks) * sizeof(float);
    hf_layer_attention_reduce_kernel<<<session->heads, decode_head_threads,
                                       reduce_shared_bytes, best->stream>>>(
        session->dtype, session->hidden, session->heads, session->kv_heads,
        session->head_dim, session->intermediate, session->device_step,
        max_steps, attention_chunks, session->device_scratch,
        session->device_decode_attention_values, session->device_decode_attention_m,
        session->device_decode_attention_l, session->device_projection_input);
    out->dependency_kernel_launches += 1;
    return cudaGetLastError();
  };

  if (request->layer_index == 0) {
    for (NervaCudaHfDecodeSequenceSession *session : selected) {
      hf_decode_set_step_kernel<<<1, 1, 0, best->stream>>>(
          session->device_step, session->active_cursor);
      out->dependency_kernel_launches += 1;
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        out->cuda_error = static_cast<int32_t>(err);
        return -1;
      }
      const uint32_t attention_hidden = session->heads * session->head_dim;
      const uint32_t kv_hidden = session->kv_heads * session->head_dim;
      const SequenceLayerLayout first_layout = session->host_layouts[0];
      hf_decode_prepare_first_attn_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, best->stream>>>(
          session->device_arena, session->arena_layout, first_layout,
          session->dtype, session->hidden, attention_hidden, kv_hidden,
          session->intermediate, session->device_step, session->max_context_tokens,
          session->device_prompt_tokens, session->active_prompt_token_count,
          session->device_slots, session->rms_eps, session->device_scratch,
          session->device_projection_input);
      out->dependency_kernel_launches += 1;
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        out->cuda_error = static_cast<int32_t>(err);
        return -1;
      }
    }
  }

  constexpr uint32_t kLayerProjectionKinds[] = {
      kProjectionBatchKindQkv,
      kProjectionBatchKindAttentionOutput,
      kProjectionBatchKindGateUp,
      kProjectionBatchKindDown,
  };
  NervaCudaHfDecodeSequenceProjectionBatchExecuteResult stages[4];

  int rc = run_stage(kLayerProjectionKinds[0], &stages[0]);
  if (rc != 0 || stages[0].status != 0 || stages[0].exact == 0) {
    out->exact = 0;
    out->status = stages[0].status;
    return rc;
  }

  for (NervaCudaHfDecodeSequenceSession *session : selected) {
    err = launch_attention_encode(session);
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  rc = run_stage(kLayerProjectionKinds[1], &stages[1]);
  if (rc != 0 || stages[1].status != 0 || stages[1].exact == 0) {
    out->exact = 0;
    out->status = stages[1].status;
    return rc;
  }

  for (NervaCudaHfDecodeSequenceSession *session : selected) {
    const uint32_t attention_hidden = session->heads * session->head_dim;
    const uint32_t kv_hidden = session->kv_heads * session->head_dim;
    const SequenceLayerLayout layout =
        session->host_layouts[request->layer_index];
    hf_layer_mlp_norm_encode_kernel<<<1, kDecodeNormThreads, 0, best->stream>>>(
        session->device_arena, layout, session->dtype, session->hidden,
        attention_hidden, kv_hidden, session->intermediate, session->device_step,
        session->max_context_tokens, session->rms_eps, session->device_scratch,
        session->device_projection_input);
    out->dependency_kernel_launches += 1;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  rc = run_stage(kLayerProjectionKinds[2], &stages[2]);
  if (rc != 0 || stages[2].status != 0 || stages[2].exact == 0) {
    out->exact = 0;
    out->status = stages[2].status;
    return rc;
  }

  for (NervaCudaHfDecodeSequenceSession *session : selected) {
    const uint32_t attention_hidden = session->heads * session->head_dim;
    const uint32_t kv_hidden = session->kv_heads * session->head_dim;
    const uint32_t ff_blocks =
        (session->intermediate + kDecodeThreads - 1) / kDecodeThreads;
    hf_layer_ff_encode_kernel<<<ff_blocks, kDecodeThreads, 0, best->stream>>>(
        session->dtype, session->hidden, attention_hidden, kv_hidden,
        session->intermediate, session->device_step, session->max_context_tokens,
        session->device_scratch, session->device_projection_input);
    out->dependency_kernel_launches += 1;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  rc = run_stage(kLayerProjectionKinds[3], &stages[3]);
  if (rc != 0 || stages[3].status != 0 || stages[3].exact == 0) {
    out->exact = 0;
    out->status = stages[3].status;
    return rc;
  }

  for (NervaCudaHfDecodeSequenceSession *session : selected) {
    const uint32_t attention_hidden = session->heads * session->head_dim;
    const uint32_t kv_hidden = session->kv_heads * session->head_dim;
    if (request->layer_index + 1 < session->layer_count) {
      const SequenceLayerLayout next_layout =
          session->host_layouts[request->layer_index + 1];
      const uint64_t output_offset =
          (request->layer_index % 2 == 0) ? session->arena_layout.scratch
                                          : session->arena_layout.input;
      hf_layer_finish_next_attn_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, best->stream>>>(
          session->device_arena, output_offset, next_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, session->max_context_tokens, session->rms_eps,
          session->device_scratch, session->device_projection_input);
    } else {
      hf_layer_finish_final_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, best->stream>>>(
          session->device_arena, session->arena_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, session->max_context_tokens, session->rms_eps,
          session->device_scratch, session->device_projection_input);
    }
    out->dependency_kernel_launches += 1;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  err = cudaStreamSynchronize(best->stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  const auto &qkv = stages[0];
  const auto &attention_output = stages[1];
  const auto &gate_up = stages[2];
  const auto &down = stages[3];
  out->reason = kProjectionBatchPlanReady;
  out->exact = 1;
  out->status = 0;
  out->qkv_rows = qkv.rows;
  out->attention_output_rows = attention_output.rows;
  out->gate_up_rows = gate_up.rows;
  out->down_rows = down.rows;
  out->hidden_cols = qkv.cols;
  out->attention_output_cols = attention_output.cols;
  out->down_cols = down.cols;
  out->input_bytes = qkv.input_bytes + attention_output.input_bytes +
                     gate_up.input_bytes + down.input_bytes;
  out->output_bytes = qkv.output_bytes + attention_output.output_bytes +
                      gate_up.output_bytes + down.output_bytes;
  out->qkv_elapsed_ns = qkv.elapsed_ns;
  out->attention_output_elapsed_ns = attention_output.elapsed_ns;
  out->gate_up_elapsed_ns = gate_up.elapsed_ns;
  out->down_elapsed_ns = down.elapsed_ns;
  out->elapsed_ns = qkv.elapsed_ns + attention_output.elapsed_ns +
                    gate_up.elapsed_ns + down.elapsed_ns;
  out->pack_kernel_launches = qkv.pack_kernel_launches +
                              attention_output.pack_kernel_launches +
                              gate_up.pack_kernel_launches +
                              down.pack_kernel_launches;
  out->projection_kernel_launches =
      qkv.projection_kernel_launches +
      attention_output.projection_kernel_launches +
      gate_up.projection_kernel_launches + down.projection_kernel_launches;
  out->scatter_kernel_launches = qkv.scatter_kernel_launches +
                                 attention_output.scatter_kernel_launches +
                                 gate_up.scatter_kernel_launches +
                                 down.scatter_kernel_launches;
  out->sync_calls += qkv.sync_calls + attention_output.sync_calls +
                     gate_up.sync_calls + down.sync_calls;
  out->hot_path_allocations = qkv.hot_path_allocations +
                              attention_output.hot_path_allocations +
                              gate_up.hot_path_allocations +
                              down.hot_path_allocations;
  return 0;
}

extern "C" int nerva_cuda_hf_decode_sequence_batch_advance_one(
    const NervaCudaHfDecodeSequenceBatchAdvanceRequest *request,
    NervaCudaHfDecodeSequenceBatchAdvanceResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->reason = kProjectionBatchPlanInvalidRequest;
  if (request == nullptr) {
    return -1;
  }
  out->requested_session_count = request->session_count;
  out->target_block_tokens =
      request->target_block_tokens == 0 ? 1u : request->target_block_tokens;
  out->min_block_tokens =
      request->min_block_tokens == 0 ? 1u : request->min_block_tokens;
  if (request->session_count == 0) {
    out->reason = kProjectionBatchPlanNoSessions;
    out->status = 0;
    return 0;
  }
  if (request->sessions == nullptr || request->output_tokens == nullptr ||
      request->output_token_capacity < request->session_count) {
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  auto ready_session = [](const NervaCudaHfDecodeSequenceSession *session) {
    if (session == nullptr || !session->active_started ||
        session->active_finished || session->active_prompt_token_count == 0 ||
        session->active_cursor >= session->max_context_tokens ||
        !use_cublas_layer_path(session)) {
      return false;
    }
    const uint32_t prompt_count = session->active_prompt_token_count;
    const uint32_t target_cursor =
        prompt_count + session->active_observed_tokens;
    return target_cursor == session->active_cursor + 1u;
  };
  auto same_projection_model =
      [](const NervaCudaHfDecodeSequenceSession *lhs,
         const NervaCudaHfDecodeSequenceSession *rhs) {
        return lhs->planned_weight_descriptor_hash != 0 &&
               rhs->planned_weight_descriptor_hash != 0 &&
               lhs->planned_weight_descriptor_hash ==
                   rhs->planned_weight_descriptor_hash &&
               lhs->dtype == rhs->dtype && lhs->hidden == rhs->hidden &&
               lhs->heads == rhs->heads && lhs->kv_heads == rhs->kv_heads &&
               lhs->head_dim == rhs->head_dim &&
               lhs->intermediate == rhs->intermediate &&
               lhs->vocab_size == rhs->vocab_size &&
               lhs->layer_count == rhs->layer_count &&
               lhs->resident_weight_bytes == rhs->resident_weight_bytes;
      };

  uint32_t ready_count = 0;
  bool any_hash = false;
  for (uint32_t index = 0; index < request->session_count; ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (!ready_session(session)) {
      continue;
    }
    ready_count += 1;
    any_hash = any_hash || session->planned_weight_descriptor_hash != 0;
  }
  if (ready_count == 0) {
    out->reason = kProjectionBatchPlanNoReadySessions;
    out->status = 0;
    return 0;
  }
  if (!any_hash) {
    out->reason = kProjectionBatchPlanSharedWeightsUnproven;
    out->status = 0;
    return 0;
  }

  NervaCudaHfDecodeSequenceSession *best = nullptr;
  uint32_t best_count = 0;
  for (uint32_t candidate_index = 0; candidate_index < request->session_count;
       ++candidate_index) {
    NervaCudaHfDecodeSequenceSession *candidate =
        request->sessions[candidate_index];
    if (!ready_session(candidate) ||
        candidate->planned_weight_descriptor_hash == 0 ||
        candidate->layer_count == 0) {
      continue;
    }
    uint32_t compatible = 0;
    for (uint32_t other_index = 0; other_index < request->session_count;
         ++other_index) {
      NervaCudaHfDecodeSequenceSession *other = request->sessions[other_index];
      if (ready_session(other) && same_projection_model(candidate, other)) {
        compatible += 1;
      }
    }
    if (compatible > best_count) {
      best = candidate;
      best_count = compatible;
    }
  }
  out->eligible_session_count = best_count;
  if (best == nullptr || best_count < out->min_block_tokens) {
    out->reason = kProjectionBatchPlanInsufficientCompatibleReady;
    out->status = 0;
    return 0;
  }

  std::vector<uint32_t> selected_indices;
  selected_indices.reserve(out->target_block_tokens);
  for (uint32_t index = 0; index < request->session_count &&
                           selected_indices.size() < out->target_block_tokens;
       ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (ready_session(session) && same_projection_model(best, session)) {
      selected_indices.push_back(index);
    }
  }
  if (selected_indices.size() < out->min_block_tokens) {
    out->reason = kProjectionBatchPlanInsufficientCompatibleReady;
    out->status = 0;
    return 0;
  }
  std::vector<NervaCudaHfDecodeSequenceSession *> selected_sessions;
  selected_sessions.reserve(selected_indices.size());
  for (uint32_t request_index : selected_indices) {
    selected_sessions.push_back(request->sessions[request_index]);
  }

  out->reason = kProjectionBatchPlanReady;
  out->exact = 1;
  out->block_tokens = static_cast<uint32_t>(selected_indices.size());
  out->dtype = best->dtype;
  out->layer_count = best->layer_count;

  for (uint32_t layer_index = 0;
       layer_index < best->layer_count; ++layer_index) {
    NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteRequest layer_request{};
    layer_request.sessions = selected_sessions.data();
    layer_request.session_count =
        static_cast<uint32_t>(selected_sessions.size());
    layer_request.target_block_tokens = request->target_block_tokens;
    layer_request.min_block_tokens = request->min_block_tokens;
    layer_request.layer_index = layer_index;
    NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult layer_out{};
    const int rc = nerva_cuda_hf_decode_sequence_layer_projection_batch_execute(
        &layer_request, &layer_out);
    out->cuda_error = layer_out.cuda_error;
    out->device_count = layer_out.device_count;
    out->reason = layer_out.reason;
    out->eligible_session_count = layer_out.eligible_session_count;
    out->block_tokens = layer_out.block_tokens;
    out->target_block_tokens = layer_out.target_block_tokens;
    out->min_block_tokens = layer_out.min_block_tokens;
    out->dtype = layer_out.dtype;
    if (rc != 0 || layer_out.status != 0 || layer_out.exact == 0) {
      out->exact = 0;
      out->status = layer_out.status;
      return rc;
    }
    out->projection_elapsed_ns += layer_out.elapsed_ns;
    out->qkv_elapsed_ns += layer_out.qkv_elapsed_ns;
    out->attention_output_elapsed_ns += layer_out.attention_output_elapsed_ns;
    out->gate_up_elapsed_ns += layer_out.gate_up_elapsed_ns;
    out->down_elapsed_ns += layer_out.down_elapsed_ns;
    out->pack_kernel_launches += layer_out.pack_kernel_launches;
    out->projection_kernel_launches += layer_out.projection_kernel_launches;
    out->scatter_kernel_launches += layer_out.scatter_kernel_launches;
    out->dependency_kernel_launches += layer_out.dependency_kernel_launches;
    out->sync_calls += layer_out.sync_calls;
    out->hot_path_allocations += layer_out.hot_path_allocations;
  }

  NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest lm_head_request{};
  lm_head_request.sessions = selected_sessions.data();
  lm_head_request.session_count =
      static_cast<uint32_t>(selected_sessions.size());
  lm_head_request.target_block_tokens = request->target_block_tokens;
  lm_head_request.min_block_tokens = request->min_block_tokens;
  lm_head_request.projection_kind = kProjectionBatchKindLmHead;
  lm_head_request.layer_index = 0;
  NervaCudaHfDecodeSequenceProjectionBatchExecuteResult lm_head_out{};
  const int lm_rc = nerva_cuda_hf_decode_sequence_projection_batch_execute(
      &lm_head_request, &lm_head_out);
  out->cuda_error = lm_head_out.cuda_error;
  out->device_count = lm_head_out.device_count;
  out->reason = lm_head_out.reason;
  out->eligible_session_count = lm_head_out.eligible_session_count;
  out->block_tokens = lm_head_out.block_tokens;
  out->target_block_tokens = lm_head_out.target_block_tokens;
  out->min_block_tokens = lm_head_out.min_block_tokens;
  out->dtype = lm_head_out.dtype;
  if (lm_rc != 0 || lm_head_out.status != 0 || lm_head_out.exact == 0) {
    out->exact = 0;
    out->status = lm_head_out.status;
    return lm_rc;
  }
  out->projection_elapsed_ns += lm_head_out.elapsed_ns;
  out->lm_head_elapsed_ns = lm_head_out.elapsed_ns;
  out->pack_kernel_launches += lm_head_out.pack_kernel_launches;
  out->projection_kernel_launches += lm_head_out.projection_kernel_launches;
  out->scatter_kernel_launches += lm_head_out.scatter_kernel_launches;
  out->sync_calls += lm_head_out.sync_calls;
  out->hot_path_allocations += lm_head_out.hot_path_allocations;

  for (uint32_t selected = 0; selected < selected_indices.size(); ++selected) {
    const uint32_t request_index = selected_indices[selected];
    NervaCudaHfDecodeSequenceSession *session = request->sessions[request_index];
    float *device_logits = session->device_scratch + session->hidden * 2;
    hf_decode_final_head_reduce_kernel<<<1, kDecodeSampleThreads, 0,
                                         best->stream>>>(
        session->device_step, session->max_context_tokens,
        session->active_has_eos_token, session->active_eos_token,
        device_logits, session->vocab_size, session->device_slots);
    out->sampling_kernel_launches += 1;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  for (uint32_t selected = 0; selected < selected_indices.size(); ++selected) {
    const uint32_t request_index = selected_indices[selected];
    NervaCudaHfDecodeSequenceSession *session = request->sessions[request_index];
    const uint32_t slot_start =
        session->active_prompt_token_count - 1u + session->active_observed_tokens;
    err = cudaMemcpyAsync(session->host_slots + slot_start,
                          session->device_slots + slot_start,
                          sizeof(NervaCudaSyntheticTokenSlot),
                          cudaMemcpyDeviceToHost, best->stream);
    out->d2h_bytes += sizeof(NervaCudaSyntheticTokenSlot);
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }
  err = cudaStreamSynchronize(best->stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  std::vector<uint32_t> observed;
  observed.reserve(selected_indices.size());
  for (uint32_t selected = 0; selected < selected_indices.size(); ++selected) {
    const uint32_t request_index = selected_indices[selected];
    NervaCudaHfDecodeSequenceSession *session = request->sessions[request_index];
    const uint32_t slot_start =
        session->active_prompt_token_count - 1u + session->active_observed_tokens;
    const NervaCudaSyntheticTokenSlot &slot = session->host_slots[slot_start];
    if (slot.request_id != kRequestId || slot.sequence_id != kSequenceId ||
        slot.completion != kCompletionDeviceComplete ||
        slot.token_index != slot_start) {
      out->status = -1;
      return -1;
    }
    request->output_tokens[request_index] = slot.token;
    observed.push_back(slot.token);
    out->last_token = slot.token;
    session->active_observed_tokens += 1;
    session->active_cursor += 1;
    const uint32_t kv_tokens = slot_start + 1u;
    session->active_finished =
        (session->active_has_eos_token != 0 &&
         slot.token == session->active_eos_token) ||
        kv_tokens >= session->max_context_tokens;
  }
  out->observed_tokens = static_cast<uint32_t>(observed.size());
  out->observed_token_hash =
      hash_tokens(observed.data(), static_cast<uint32_t>(observed.size()));
  out->reason = kProjectionBatchPlanReady;
  out->exact = 1;
  out->status = out->observed_tokens == out->block_tokens ? 0 : -1;
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_hf_decode_sequence_session_destroy(
    NervaCudaHfDecodeSequenceSession *session,
    NervaCudaHfDecodeSequenceSessionCreateResult *out) {
  if (out != nullptr) {
    memset(out, 0, sizeof(*out));
    out->status = -1;
  }
  if (session == nullptr) {
    return -1;
  }
  if (out != nullptr) {
    fill_create_result(session, out);
  }
  free_session_fields(session);
  delete session;
  return 0;
}
