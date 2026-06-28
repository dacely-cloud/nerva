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
#include <vector>

namespace {

constexpr uint32_t kDTypeF16 = 0;
constexpr uint32_t kDTypeBF16 = 1;
constexpr uint32_t kRequestId = 1;
constexpr uint32_t kSequenceId = 1;
constexpr uint32_t kCompletionDeviceComplete = 1;
constexpr uint32_t kWeightStrategyGpuResident = 1;
constexpr uint32_t kWeightStrategyGpuStaged = 2;
constexpr uint32_t kDecodeThreads = 256;
constexpr uint32_t kPrefillChunkBaseTokens = 1024;
constexpr uint32_t kPrefillChunkMaxTokens = 8192;
constexpr uint32_t kVerifyMaxDraftTokens = 128;
constexpr uint32_t kDecodeAttentionChunkTokens = 64;
constexpr uint32_t kChunkedDecodeAttentionThreshold = 128;
constexpr uint64_t kMissingOffset = UINT64_MAX;
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x00000100000001b3ull;
constexpr size_t kCublasWorkspaceBytes = 16ull * 1024ull * 1024ull;
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
  kCreateStageVerifyLogitsAlloc = 33,
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

__device__ float encoded_to_f32(uint16_t value, uint32_t dtype) {
  if (dtype == kDTypeBF16) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
  }
  return __half2float(__ushort_as_half(value));
}

__device__ uint16_t f32_to_encoded(float value, uint32_t dtype) {
  if (dtype == kDTypeBF16) {
    uint32_t bits = __float_as_uint(value);
    uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
  }
  return __half_as_ushort(__float2half_rn(value));
}

__device__ float silu(float value) {
  return value / (1.0f + expf(-value));
}

__device__ float block_sum(float value) {
  __shared__ float values[kDecodeThreads];
  const uint32_t tid = threadIdx.x;
  values[tid] = value;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      values[tid] += values[tid + stride];
    }
    __syncthreads();
  }
  return values[0];
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
                          float *kv_keys, float *kv_values) {
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

  const uint64_t kv_base =
      (static_cast<uint64_t>(layer_index) * max_steps + position) * kv_hidden;
  for (uint32_t index = threadIdx.x; index < kv_hidden; index += blockDim.x) {
    kv_keys[kv_base + index] = k[index];
    kv_values[kv_base + index] = v[index];
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
      const uint64_t token_base =
          (static_cast<uint64_t>(layer_index) * max_steps + token) * kv_hidden +
          kv_head * head_dim;
      float score = 0.0f;
      for (uint32_t offset = 0; offset < head_dim; ++offset) {
        score += q[head_start + offset] * kv_keys[token_base + offset];
      }
      score *= scale;
      const float next_m = fmaxf(local_m, score);
      const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
      const float new_scale = expf(score - next_m);
      for (uint32_t offset = 0; offset < head_dim; ++offset) {
        const uint32_t out = head_start + offset;
        attn[out] = attn[out] * old_scale + kv_values[token_base + offset] * new_scale;
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
  __shared__ float best_values[kDecodeThreads];
  __shared__ uint32_t best_indices[kDecodeThreads];
  __shared__ uint32_t current_position_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  if (current_position >= max_steps) {
    return;
  }
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
      *step_cursor = has_eos_token != 0 && best_index == eos_token
                         ? max_steps
                         : current_position + 1;
    }
  }
}

__global__ void hf_decode_sequence_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, SequenceLayerLayout *layers,
    uint32_t layer_count, uint32_t dtype, uint32_t hidden, uint32_t heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t intermediate, uint32_t position,
    uint32_t *step_cursor, uint32_t max_steps, const uint32_t *prompt_tokens,
    uint32_t prompt_token_count, float rms_eps, float rope_theta, float *scratch,
    float *kv_keys, float *kv_values, const NervaCudaSyntheticTokenSlot *slots) {
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
  if (current_position >= max_steps) {
    return;
  }
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
              kv_values);
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
  if (current_position >= max_steps) {
    return;
  }
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
  if (current_position >= max_steps) {
    return;
  }
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
    float *kv_keys, float *kv_values, uint16_t *projection_input) {
  __shared__ uint32_t current_position_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  if (current_position >= max_steps) {
    return;
  }
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
    const uint64_t token_base =
        (static_cast<uint64_t>(layer_index) * max_steps + token) * kv_hidden +
        kv_head * head_dim;
    float partial = 0.0f;
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      partial += s.q[head_start + offset] * kv_keys[token_base + offset];
    }
    const float score = block_sum(partial) * scale;
    const float next_m = fmaxf(local_m, score);
    const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
    const float new_scale = expf(score - next_m);
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      const uint32_t out = head_start + offset;
      s.attn[out] =
          s.attn[out] * old_scale + kv_values[token_base + offset] * new_scale;
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
    float *kv_keys, float *kv_values, uint16_t *projection_input) {
  __shared__ uint32_t current_position_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  if (current_position >= max_steps) {
    return;
  }
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
  const uint64_t kv_base =
      (static_cast<uint64_t>(layer_index) * max_steps + current_position) *
          kv_hidden +
      kv_start;
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
      kv_keys[kv_base + offset] = current_k;
      kv_values[kv_base + offset] = current_v;
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
    float partial = 0.0f;
    const uint64_t token_base =
        (static_cast<uint64_t>(layer_index) * max_steps + token) * kv_hidden +
        kv_start;
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      float key_value = kv_keys[token_base + offset];
      if (token == current_position) {
        key_value = s.residual[head_start + offset];
      }
      partial += s.q[head_start + offset] * key_value;
    }
    const float score = block_sum(partial) * scale;
    const float next_m = fmaxf(local_m, score);
    const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
    const float new_scale = expf(score - next_m);
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      const uint32_t out = head_start + offset;
      float value_value = kv_values[token_base + offset];
      if (token == current_position) {
        value_value = s.mlp_norm[head_start + offset];
      }
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
    float *kv_keys, float *kv_values) {
  __shared__ uint32_t current_position_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  if (current_position >= max_steps) {
    return;
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
  }
  if (head < kv_heads) {
    float *k = s.k + head * head_dim;
    float *v = s.v + head * head_dim;
    add_optional_head_bias(arena, layout.k_bias, head, head_dim, dtype, k);
    add_optional_head_bias(arena, layout.v_bias, head, head_dim, dtype, v);
    per_head_rms_norm_block(arena, layout.k_norm, k, head_dim, dtype, rms_eps);
    apply_rope_head(k, head_dim, current_position, rope_theta);
    const uint64_t kv_base =
        (static_cast<uint64_t>(layer_index) * max_steps + current_position) *
            kv_hidden +
        static_cast<uint64_t>(head) * head_dim;
    for (uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
      kv_keys[kv_base + index] = k[index];
      kv_values[kv_base + index] = v[index];
    }
  }
}

__global__ void hf_layer_attention_chunk_kernel(
    uint32_t layer_index, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, uint32_t attention_chunks, float *scratch,
    float *kv_keys, float *kv_values, float *partial_values, float *partial_m,
    float *partial_l) {
  __shared__ uint32_t current_position_shared;
  if (threadIdx.x == 0) {
    current_position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();
  const uint32_t current_position = current_position_shared;
  if (current_position >= max_steps || head_dim > blockDim.x) {
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
  const bool lane = threadIdx.x < head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float local_m = -INFINITY;
  float local_l = 0.0f;
  float acc = 0.0f;
  for (uint32_t token = chunk_start; token < chunk_end; ++token) {
    const uint64_t token_base =
        (static_cast<uint64_t>(layer_index) * max_steps + token) * kv_hidden +
        kv_start;
    const float key_value = lane ? kv_keys[token_base + threadIdx.x] : 0.0f;
    const float partial =
        lane ? s.q[head_start + threadIdx.x] * key_value : 0.0f;
    const float score = block_sum(partial) * scale;
    const float next_m = fmaxf(local_m, score);
    const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
    const float new_scale = expf(score - next_m);
    if (lane) {
      acc = acc * old_scale + kv_values[token_base + threadIdx.x] * new_scale;
    }
    local_l = local_l * old_scale + new_scale;
    local_m = next_m;
  }
  if (threadIdx.x == 0) {
    partial_m[slot] = local_m;
    partial_l[slot] = local_l;
  }
  if (lane) {
    partial_values[slot * head_dim + threadIdx.x] = acc;
  }
}

__global__ void hf_layer_attention_reduce_kernel(
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, uint32_t attention_chunks, float *scratch,
    const float *partial_values, const float *partial_m, const float *partial_l,
    uint16_t *projection_input) {
  __shared__ float global_m_shared;
  __shared__ float global_l_shared;
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  const uint32_t head = blockIdx.x;
  if (head >= heads || head_dim > blockDim.x) {
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
    global_m_shared = global_m;
    global_l_shared = global_l;
  }
  __syncthreads();
  const uint32_t head_start = head * head_dim;
  if (threadIdx.x < head_dim) {
    float value = 0.0f;
    const float global_l = global_l_shared;
    const float global_m = global_m_shared;
    if (global_l > 0.0f && isfinite(global_l)) {
      for (uint32_t chunk = 0; chunk < attention_chunks; ++chunk) {
        const uint64_t slot =
            (static_cast<uint64_t>(head) * attention_chunks + chunk);
        const float chunk_l = partial_l[slot];
        if (chunk_l <= 0.0f || !isfinite(chunk_l)) {
          continue;
        }
        const float weight = expf(partial_m[slot] - global_m) / global_l;
        value += partial_values[slot * head_dim + threadIdx.x] * weight;
      }
    }
    LayerScratch s =
        layer_scratch_ptrs(scratch, hidden, heads * head_dim,
                           kv_heads * head_dim, intermediate);
    s.attn[head_start + threadIdx.x] = value;
    projection_input[head_start + threadIdx.x] = f32_to_encoded(value, dtype);
  }
}

__global__ void hf_layer_mlp_norm_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
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
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
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
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
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
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
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
    float rms_eps, float rope_theta, float *qkv, float *kv_keys,
    float *kv_values) {
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
    const uint64_t kv_base =
        (static_cast<uint64_t>(layer_index) * max_steps + global_pos) *
            kv_hidden +
        kv_start;
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      float value = v[kv_start + offset];
      if (layout.v_bias != kMissingOffset) {
        value += encoded_to_f32(arena[layout.v_bias + kv_start + offset], dtype);
      }
      v[kv_start + offset] = value;
      kv_keys[kv_base + offset] = k[kv_start + offset];
      kv_values[kv_base + offset] = value;
    }
  }
}

__global__ void hf_prefill_attention_kernel(
    uint32_t layer_index, uint32_t dtype, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t max_steps, uint32_t chunk_start,
    uint32_t chunk_tokens, const float *qkv, const float *kv_keys,
    const float *kv_values, uint16_t *attn_out) {
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
    const uint64_t token_base =
        (static_cast<uint64_t>(layer_index) * max_steps + token) * kv_hidden +
        kv_start;
    float partial = 0.0f;
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      partial += q[head_start + offset] * kv_keys[token_base + offset];
    }
    const float score = block_sum(partial) * scale;
    const float next_m = fmaxf(local_m, score);
    const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
    const float new_scale = expf(score - next_m);
    for (uint32_t offset = threadIdx.x; offset < head_dim; offset += blockDim.x) {
      shared_attn[offset] =
          shared_attn[offset] * old_scale + kv_values[token_base + offset] * new_scale;
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

__global__ void hf_verify_logits_reduce_kernel(
    uint32_t slot_start, uint32_t token_count, uint32_t has_eos_token,
    uint32_t eos_token, const float *scores, uint32_t vocab_size,
    NervaCudaSyntheticTokenSlot *slots) {
  __shared__ float best_values[kDecodeThreads];
  __shared__ uint32_t best_indices[kDecodeThreads];
  const uint32_t local_token = blockIdx.x;
  if (local_token >= token_count) {
    return;
  }
  const float *token_scores =
      scores + static_cast<uint64_t>(local_token) * vocab_size;
  float best_value = -INFINITY;
  uint32_t best_index = 0;
  for (uint32_t index = threadIdx.x; index < vocab_size; index += blockDim.x) {
    const float value = token_scores[index];
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
    const uint32_t position = slot_start + local_token;
    const uint32_t best_index = best_indices[0];
    NervaCudaSyntheticTokenSlot *slot = slots + position;
    slot->request_id = kRequestId;
    slot->sequence_id = kSequenceId;
    slot->token_index = position;
    slot->token = best_index;
    slot->version = position + 1;
    slot->completion = kCompletionDeviceComplete;
    slot->host_copied = 0;
    (void)has_eos_token;
    (void)eos_token;
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

cudaError_t encoded_row_major_gemv_lt(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t dtype, float beta, float *output) {
  if (handle == nullptr || rows == 0 || cols == 0) {
    return cudaErrorInvalidValue;
  }
  const float alpha = 1.0f;
  const cudaDataType_t data_type = encoded_cuda_type(dtype);
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasStatus_t status =
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t op = CUBLAS_OP_N;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&a_desc, data_type, rows, cols, cols);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&b_desc, data_type, cols, 1, 1);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, rows, 1, 1);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_32F, rows, 1, 1);
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
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmul(handle, op_desc, &alpha, matrix, a_desc, input,
                            b_desc, &beta, output, c_desc, output, d_desc,
                            nullptr, workspace, workspace_bytes, stream);
  destroy_lt_descriptors(op_desc, a_desc, b_desc, c_desc, d_desc);
  return cublas_to_cuda(status);
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

}  // namespace

struct NervaCudaHfDecodeSequenceSession {
  uint32_t dtype = 0;
  uint32_t hidden = 0;
  uint32_t heads = 0;
  uint32_t kv_heads = 0;
  uint32_t head_dim = 0;
  uint32_t intermediate = 0;
  uint32_t vocab_size = 0;
  uint32_t layer_count = 0;
  uint32_t max_context_tokens = 0;
  uint32_t prefill_chunk_tokens = 0;
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
  uint64_t prefill_attn_bytes = 0;
  uint64_t prefill_o_bytes = 0;
  uint64_t prefill_gate_up_bytes = 0;
  uint64_t prefill_ff_bytes = 0;
  uint64_t prefill_down_bytes = 0;
  uint64_t decode_attention_values_bytes = 0;
  uint64_t decode_attention_stats_bytes = 0;
  uint64_t verify_logits_bytes = 0;
  uint32_t decode_attention_max_chunks = 0;
  uint64_t packed_qkv_bytes = 0;
  uint64_t packed_gate_up_bytes = 0;
  uint64_t kv_bytes = 0;
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
  uint16_t *device_prefill_attn = nullptr;
  float *device_prefill_o = nullptr;
  float *device_prefill_gate_up = nullptr;
  uint16_t *device_prefill_ff = nullptr;
  float *device_prefill_down = nullptr;
  float *device_decode_attention_values = nullptr;
  float *device_decode_attention_m = nullptr;
  float *device_decode_attention_l = nullptr;
  float *device_verify_logits = nullptr;
  uint16_t *device_qkv_packed = nullptr;
  uint16_t *device_gate_up_packed = nullptr;
  float *device_kv_keys = nullptr;
  float *device_kv_values = nullptr;
  uint32_t *device_prompt_tokens = nullptr;
  NervaCudaSyntheticTokenSlot *host_slots = nullptr;
  NervaCudaSyntheticTokenSlot *device_slots = nullptr;
  uint32_t *device_step = nullptr;
  void *cublas_workspace = nullptr;
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas = nullptr;
  cublasLtHandle_t cublas_lt = nullptr;
  cudaEvent_t device_start = nullptr;
  cudaEvent_t device_stop = nullptr;
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
  if (session->device_stop != nullptr) cudaEventDestroy(session->device_stop);
  if (session->device_start != nullptr) cudaEventDestroy(session->device_start);
  if (session->cublas_lt != nullptr) cublasLtDestroy(session->cublas_lt);
  if (session->cublas != nullptr) cublasDestroy(session->cublas);
  if (session->stream != nullptr) cudaStreamDestroy(session->stream);
  cudaFree(session->cublas_workspace);
  cudaFree(session->device_step);
  cudaFree(session->device_slots);
  cudaFree(session->device_prompt_tokens);
  cudaFree(session->device_kv_values);
  cudaFree(session->device_kv_keys);
  cudaFree(session->device_gate_up_packed);
  cudaFree(session->device_qkv_packed);
  cudaFree(session->device_prefill_down);
  cudaFree(session->device_prefill_ff);
  cudaFree(session->device_prefill_gate_up);
  cudaFree(session->device_prefill_o);
  cudaFree(session->device_prefill_attn);
  cudaFree(session->device_prefill_qkv);
  cudaFree(session->device_prefill_norm);
  cudaFree(session->device_prefill_hidden_b);
  cudaFree(session->device_prefill_hidden_a);
  cudaFree(session->device_decode_attention_l);
  cudaFree(session->device_decode_attention_m);
  cudaFree(session->device_decode_attention_values);
  cudaFree(session->device_verify_logits);
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
         session->prefill_attn_bytes + session->prefill_o_bytes +
         session->prefill_gate_up_bytes + session->prefill_ff_bytes +
         session->prefill_down_bytes + session->decode_attention_values_bytes +
         session->decode_attention_stats_bytes * 2 +
         session->verify_logits_bytes + session->packed_qkv_bytes +
         session->packed_gate_up_bytes + session->kv_bytes +
         session->prompt_bytes + session->slots_bytes + sizeof(uint32_t) +
         kCublasWorkspaceBytes;
}

uint64_t session_fixed_footprint_without_prefill_chunk(
    const NervaCudaHfDecodeSequenceSession *session) {
  return session->arena_bytes + session->layout_bytes + session->scratch_bytes +
         session->projection_input_bytes + session->prefill_hidden_bytes * 2 +
         session->decode_attention_values_bytes +
         session->decode_attention_stats_bytes * 2 +
         session->verify_logits_bytes + session->packed_qkv_bytes +
         session->packed_gate_up_bytes + session->kv_bytes +
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
  const uint64_t min_chunk =
      std::min<uint64_t>(base, std::min<uint64_t>(
          max_context_tokens, static_cast<uint64_t>(kVerifyMaxDraftTokens)));
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
         session->cublas != nullptr;
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
  return cudaEventRecord(session->device_start, session->stream);
}

cudaError_t profile_end(NervaCudaHfDecodeSequenceSession *session,
                        uint64_t *bucket) {
  cudaError_t err = cudaEventRecord(session->device_stop, session->stream);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaEventSynchronize(session->device_stop);
  if (err != cudaSuccess) {
    return err;
  }
  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, session->device_start,
                             session->device_stop);
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
      session->device_kv_values, session->device_slots);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = final_head_gemv(session->cublas, session->device_arena,
                          session->arena_layout, session->dtype,
                          session->hidden, session->vocab_size, device_logits);
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    hf_decode_final_head_reduce_kernel<<<1, kDecodeThreads, 0,
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
        1, kDecodeThreads, 0, session->stream>>>(
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
    err = encoded_row_major_gemv_lt(
        session->cublas_lt, session->stream, session->cublas_workspace,
        kCublasWorkspaceBytes,
        session->device_qkv_packed +
            packed_shape.qkv_elements_per_layer * layer_index,
        session->device_projection_input,
        static_cast<uint32_t>(packed_shape.qkv_rows), session->hidden,
        session->dtype, 0.0f, scratch.q);
    if (err == cudaSuccess && attention_chunks == 0) {
      hf_layer_qkv_attention_encode_kernel<<<session->heads, kDecodeThreads, 0,
                                             session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
      hf_layer_qkv_prepare_kernel<<<session->heads, kDecodeThreads, 0,
                                    session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values);
      err = cudaGetLastError();
      if (err == cudaSuccess) {
        const dim3 grid(session->heads, attention_chunks);
        hf_layer_attention_chunk_kernel<<<grid, kDecodeThreads, 0,
                                          session->stream>>>(
            layer_index, session->hidden, session->heads, session->kv_heads,
            session->head_dim, session->intermediate, session->device_step,
            max_steps, attention_chunks, session->device_scratch,
            session->device_kv_keys, session->device_kv_values,
            session->device_decode_attention_values,
            session->device_decode_attention_m,
            session->device_decode_attention_l);
        err = cudaGetLastError();
      }
      if (err == cudaSuccess) {
        hf_layer_attention_reduce_kernel<<<session->heads, kDecodeThreads, 0,
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
      err = encoded_row_major_gemv_lt(
          session->cublas_lt, session->stream, session->cublas_workspace,
          kCublasWorkspaceBytes, session->device_arena + layout.w_o,
          session->device_projection_input, session->hidden, attention_hidden,
          session->dtype, 0.0f, scratch.residual);
    if (err == cudaSuccess) {
      hf_layer_mlp_norm_encode_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess)
      err = encoded_row_major_gemv_lt(
          session->cublas_lt, session->stream, session->cublas_workspace,
          kCublasWorkspaceBytes,
          session->device_gate_up_packed +
              packed_shape.gate_up_elements_per_layer * layer_index,
          session->device_projection_input,
          static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden,
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
      err = encoded_row_major_gemv_lt(
          session->cublas_lt, session->stream, session->cublas_workspace,
          kCublasWorkspaceBytes, session->device_arena + layout.w_down,
          session->device_projection_input, session->hidden,
          session->intermediate, session->dtype, 0.0f, scratch.down);
    if (err == cudaSuccess && layer_index + 1 < session->layer_count) {
      const SequenceLayerLayout next_layout =
          session->host_layouts[layer_index + 1];
      hf_layer_finish_next_attn_norm_encode_kernel<<<
          1, kDecodeThreads, 0, session->stream>>>(
          session->device_arena, output_offset, next_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
      hf_layer_finish_final_norm_encode_kernel<<<
          1, kDecodeThreads, 0, session->stream>>>(
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
    err = encoded_row_major_gemv_lt(
        session->cublas_lt, session->stream, session->cublas_workspace,
        kCublasWorkspaceBytes,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        session->dtype, 0.0f, device_logits);
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    hf_decode_final_head_reduce_kernel<<<1, kDecodeThreads, 0,
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
        1, kDecodeThreads, 0, session->stream>>>(
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
      err = encoded_row_major_gemv_lt(
          session->cublas_lt, session->stream, session->cublas_workspace,
          kCublasWorkspaceBytes,
          session->device_qkv_packed +
              packed_shape.qkv_elements_per_layer * layer_index,
          session->device_projection_input,
          static_cast<uint32_t>(packed_shape.qkv_rows), session->hidden,
          session->dtype, 0.0f, scratch.q);
    if (err == cudaSuccess) err = profile_end(session, &qkv_projection_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess && attention_chunks == 0) {
      hf_layer_qkv_attention_encode_kernel<<<session->heads, kDecodeThreads, 0,
                                             session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
      hf_layer_qkv_prepare_kernel<<<session->heads, kDecodeThreads, 0,
                                    session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values);
      err = cudaGetLastError();
      if (err == cudaSuccess) {
        const dim3 grid(session->heads, attention_chunks);
        hf_layer_attention_chunk_kernel<<<grid, kDecodeThreads, 0,
                                          session->stream>>>(
            layer_index, session->hidden, session->heads, session->kv_heads,
            session->head_dim, session->intermediate, session->device_step,
            max_steps, attention_chunks, session->device_scratch,
            session->device_kv_keys, session->device_kv_values,
            session->device_decode_attention_values,
            session->device_decode_attention_m,
            session->device_decode_attention_l);
        err = cudaGetLastError();
      }
      if (err == cudaSuccess) {
        hf_layer_attention_reduce_kernel<<<session->heads, kDecodeThreads, 0,
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
      err = encoded_row_major_gemv_lt(
          session->cublas_lt, session->stream, session->cublas_workspace,
          kCublasWorkspaceBytes, session->device_arena + layout.w_o,
          session->device_projection_input, session->hidden, attention_hidden,
          session->dtype, 0.0f, scratch.residual);
    if (err == cudaSuccess) {
      err = profile_end(session, &attention_output_projection_ns);
    }

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess) {
      hf_layer_mlp_norm_encode_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = profile_end(session, &norm_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess)
      err = encoded_row_major_gemv_lt(
          session->cublas_lt, session->stream, session->cublas_workspace,
          kCublasWorkspaceBytes,
          session->device_gate_up_packed +
              packed_shape.gate_up_elements_per_layer * layer_index,
          session->device_projection_input,
          static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden,
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
      err = encoded_row_major_gemv_lt(
          session->cublas_lt, session->stream, session->cublas_workspace,
          kCublasWorkspaceBytes, session->device_arena + layout.w_down,
          session->device_projection_input, session->hidden,
          session->intermediate, session->dtype, 0.0f, scratch.down);
    if (err == cudaSuccess) err = profile_end(session, &down_projection_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess && layer_index + 1 < session->layer_count) {
      const SequenceLayerLayout next_layout =
          session->host_layouts[layer_index + 1];
      hf_layer_finish_next_attn_norm_encode_kernel<<<
          1, kDecodeThreads, 0, session->stream>>>(
          session->device_arena, output_offset, next_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
      hf_layer_finish_final_norm_encode_kernel<<<
          1, kDecodeThreads, 0, session->stream>>>(
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
    err = encoded_row_major_gemv_lt(
        session->cublas_lt, session->stream, session->cublas_workspace,
        kCublasWorkspaceBytes,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        session->dtype, 0.0f, device_logits);
  }
  if (err == cudaSuccess) err = profile_end(session, &lm_head_projection_ns);

  if (err == cudaSuccess) err = profile_begin(session);
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    hf_decode_final_head_reduce_kernel<<<1, kDecodeThreads, 0,
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
      hf_prefill_attn_norm_kernel<<<chunk_tokens, kDecodeThreads, 0,
                                    session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          chunk_start, chunk_tokens, session->rms_eps, hidden_in,
          session->device_prefill_norm);
      err = cudaGetLastError();
      out->kernel_launches += 1;
      if (err == cudaSuccess) {
        err = encoded_row_major_gemm_tokens(
            session->cublas,
            session->device_qkv_packed +
                packed_shape.qkv_elements_per_layer * layer_index,
            session->device_prefill_norm,
            static_cast<uint32_t>(packed_shape.qkv_rows), session->hidden,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_qkv);
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) {
        const dim3 grid(chunk_tokens, std::max(session->heads, session->kv_heads));
        hf_prefill_qkv_publish_kernel<<<grid, kDecodeThreads, 0,
                                        session->stream>>>(
            session->device_arena, layout, layer_index, session->dtype,
            session->heads, session->kv_heads, session->head_dim,
            session->max_context_tokens, chunk_start, chunk_tokens,
            session->rms_eps, session->rope_theta, session->device_prefill_qkv,
            session->device_kv_keys, session->device_kv_values);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) {
        const dim3 grid(chunk_tokens, session->heads);
        hf_prefill_attention_kernel<<<grid, kDecodeThreads,
                                      session->head_dim * sizeof(float),
                                      session->stream>>>(
            layer_index, session->dtype, session->heads, session->kv_heads,
            session->head_dim, session->max_context_tokens, chunk_start,
            chunk_tokens, session->device_prefill_qkv, session->device_kv_keys,
            session->device_kv_values, session->device_prefill_attn);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) {
        err = encoded_row_major_gemm_tokens(
            session->cublas, session->device_arena + layout.w_o,
            session->device_prefill_attn, session->hidden, attention_hidden,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_o);
        out->kernel_launches += 1;
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
      if (err == cudaSuccess) {
        err = encoded_row_major_gemm_tokens(
            session->cublas,
            session->device_gate_up_packed +
                packed_shape.gate_up_elements_per_layer * layer_index,
            session->device_prefill_norm,
            static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_gate_up);
        out->kernel_launches += 1;
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
      if (err == cudaSuccess) {
        err = encoded_row_major_gemm_tokens(
            session->cublas, session->device_arena + layout.w_down,
            session->device_prefill_ff, session->hidden, session->intermediate,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_down);
        out->kernel_launches += 1;
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
    }
    std::swap(hidden_in, hidden_out);
  }
  if (err == cudaSuccess) {
    hf_prefill_final_norm_last_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, session->dtype,
        session->hidden, prompt_token_count, session->rms_eps, hidden_in,
        session->device_projection_input);
    err = cudaGetLastError();
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) {
    hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(
        session->device_step, prompt_token_count - 1u);
    err = cudaGetLastError();
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = encoded_row_major_gemv_lt(
        session->cublas_lt, session->stream, session->cublas_workspace,
        kCublasWorkspaceBytes,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        session->dtype, 0.0f, device_logits);
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    hf_decode_final_head_reduce_kernel<<<1, kDecodeThreads, 0,
                                         session->stream>>>(
        session->device_step, session->max_context_tokens, has_eos_token,
        eos_token, device_logits, session->vocab_size, session->device_slots);
    err = cudaGetLastError();
    out->kernel_launches += 1;
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

cudaError_t launch_cublas_session_verify_draft(
    NervaCudaHfDecodeSequenceSession *session, uint32_t chunk_start,
    uint32_t draft_token_count, uint32_t has_eos_token, uint32_t eos_token,
    NervaCudaHfDecodeSequenceResult *out) {
  if (draft_token_count == 0) {
    return cudaSuccess;
  }
  if (draft_token_count > kVerifyMaxDraftTokens ||
      draft_token_count > session->prefill_chunk_tokens ||
      chunk_start >= session->max_context_tokens ||
      chunk_start + draft_token_count > session->max_context_tokens ||
      !use_cublas_layer_path(session)) {
    return cudaErrorInvalidValue;
  }
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const PackedProjectionShape packed_shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  cudaError_t err = cudaSuccess;
  hf_prefill_embed_range_kernel<<<draft_token_count, kDecodeThreads, 0,
                                  session->stream>>>(
      session->device_arena, session->arena_layout, session->hidden,
      session->device_prompt_tokens, draft_token_count, chunk_start,
      session->device_prefill_hidden_a);
  err = cudaGetLastError();
  out->kernel_launches += 1;
  uint16_t *hidden_in = session->device_prefill_hidden_a;
  uint16_t *hidden_out = session->device_prefill_hidden_b;
  for (uint32_t layer_index = 0;
       err == cudaSuccess && layer_index < session->layer_count;
       ++layer_index) {
    const SequenceLayerLayout layout = session->host_layouts[layer_index];
    hf_prefill_attn_norm_kernel<<<draft_token_count, kDecodeThreads, 0,
                                  session->stream>>>(
        session->device_arena, layout, session->dtype, session->hidden,
        chunk_start, draft_token_count, session->rms_eps, hidden_in,
        session->device_prefill_norm);
    err = cudaGetLastError();
    out->kernel_launches += 1;
    if (err == cudaSuccess) {
      err = encoded_row_major_gemm_tokens(
          session->cublas,
          session->device_qkv_packed +
              packed_shape.qkv_elements_per_layer * layer_index,
          session->device_prefill_norm,
          static_cast<uint32_t>(packed_shape.qkv_rows), session->hidden,
          draft_token_count, session->dtype, 0.0f, session->device_prefill_qkv);
      out->kernel_launches += 1;
    }
    if (err == cudaSuccess) {
      const dim3 grid(draft_token_count,
                      std::max(session->heads, session->kv_heads));
      hf_prefill_qkv_publish_kernel<<<grid, kDecodeThreads, 0,
                                      session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->heads, session->kv_heads, session->head_dim,
          session->max_context_tokens, chunk_start, draft_token_count,
          session->rms_eps, session->rope_theta, session->device_prefill_qkv,
          session->device_kv_keys, session->device_kv_values);
      err = cudaGetLastError();
      out->kernel_launches += 1;
    }
    if (err == cudaSuccess) {
      const dim3 grid(draft_token_count, session->heads);
      hf_prefill_attention_kernel<<<grid, kDecodeThreads,
                                    session->head_dim * sizeof(float),
                                    session->stream>>>(
          layer_index, session->dtype, session->heads, session->kv_heads,
          session->head_dim, session->max_context_tokens, chunk_start,
          draft_token_count, session->device_prefill_qkv,
          session->device_kv_keys, session->device_kv_values,
          session->device_prefill_attn);
      err = cudaGetLastError();
      out->kernel_launches += 1;
    }
    if (err == cudaSuccess) {
      err = encoded_row_major_gemm_tokens(
          session->cublas, session->device_arena + layout.w_o,
          session->device_prefill_attn, session->hidden, attention_hidden,
          draft_token_count, session->dtype, 0.0f, session->device_prefill_o);
      out->kernel_launches += 1;
    }
    if (err == cudaSuccess) {
      hf_prefill_mlp_norm_kernel<<<draft_token_count, kDecodeThreads, 0,
                                   session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          chunk_start, draft_token_count, session->rms_eps, hidden_in,
          session->device_prefill_o, session->device_prefill_norm);
      err = cudaGetLastError();
      out->kernel_launches += 1;
    }
    if (err == cudaSuccess) {
      err = encoded_row_major_gemm_tokens(
          session->cublas,
          session->device_gate_up_packed +
              packed_shape.gate_up_elements_per_layer * layer_index,
          session->device_prefill_norm,
          static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden,
          draft_token_count, session->dtype, 0.0f,
          session->device_prefill_gate_up);
      out->kernel_launches += 1;
    }
    if (err == cudaSuccess) {
      const uint32_t blocks = static_cast<uint32_t>(
          (static_cast<uint64_t>(draft_token_count) * session->intermediate +
           kDecodeThreads - 1) /
          kDecodeThreads);
      hf_prefill_ff_kernel<<<blocks, kDecodeThreads, 0, session->stream>>>(
          session->dtype, session->intermediate, draft_token_count,
          session->device_prefill_gate_up, session->device_prefill_ff);
      err = cudaGetLastError();
      out->kernel_launches += 1;
    }
    if (err == cudaSuccess) {
      err = encoded_row_major_gemm_tokens(
          session->cublas, session->device_arena + layout.w_down,
          session->device_prefill_ff, session->hidden, session->intermediate,
          draft_token_count, session->dtype, 0.0f, session->device_prefill_down);
      out->kernel_launches += 1;
    }
    if (err == cudaSuccess) {
      const uint32_t blocks = static_cast<uint32_t>(
          (static_cast<uint64_t>(draft_token_count) * session->hidden +
           kDecodeThreads - 1) /
          kDecodeThreads);
      hf_prefill_finish_kernel<<<blocks, kDecodeThreads, 0, session->stream>>>(
          session->dtype, session->hidden, chunk_start, draft_token_count,
          session->device_prefill_o, session->device_prefill_down, hidden_out);
      err = cudaGetLastError();
      out->kernel_launches += 1;
    }
    std::swap(hidden_in, hidden_out);
  }
  if (err == cudaSuccess) {
    hf_prefill_final_norm_range_kernel<<<draft_token_count, kDecodeThreads, 0,
                                         session->stream>>>(
        session->device_arena, session->arena_layout, session->dtype,
        session->hidden, chunk_start, draft_token_count, session->rms_eps,
        hidden_in, session->device_prefill_norm);
    err = cudaGetLastError();
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) {
    err = encoded_row_major_gemm_tokens(
        session->cublas, session->device_arena + session->arena_layout.lm_head,
        session->device_prefill_norm, session->vocab_size, session->hidden,
        draft_token_count, session->dtype, 0.0f, session->device_verify_logits);
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) {
    hf_verify_logits_reduce_kernel<<<draft_token_count, kDecodeThreads, 0,
                                     session->stream>>>(
        chunk_start, draft_token_count, has_eos_token, eos_token,
        session->device_verify_logits, session->vocab_size, session->device_slots);
    err = cudaGetLastError();
    out->kernel_launches += 1;
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
  if (session_graph_matches(session, max_steps, prompt_token_count,
                            has_eos_token, eos_token, attention_chunks)) {
    out->graph_nodes = session->cached_graph_nodes;
    out->graph_cache_hits = 1;
    copy_cached_profile(session, out);
    return cudaSuccess;
  }
  reset_session_graph(session);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  bool capture_started = false;
  cudaError_t err =
      cudaStreamBeginCapture(session->stream, cudaStreamCaptureModeGlobal);
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
    session->cached_attention_chunks = attention_chunks;
    session->cached_graph_nodes = out->graph_nodes;
    out->graph_captures = 1;
    graph = nullptr;
    graph_exec = nullptr;
  }
  if (err == cudaSuccess && use_cublas_layer_path(session)) {
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
  const uint64_t kv_bytes =
      request->layer_count * static_cast<uint64_t>(context_steps) * kv_hidden * sizeof(float) * 2;
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
  float *device_kv_keys = nullptr;
  float *device_kv_values = nullptr;
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
        device_scratch, device_kv_keys, device_kv_values, device_slots);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    float *device_logits = device_scratch + hidden * 2;
    err = final_head_gemv(cublas, device_arena, arena_layout, request->dtype,
                          request->hidden, request->vocab_size, device_logits);
  }
  if (err == cudaSuccess) {
    float *device_logits = device_scratch + hidden * 2;
    hf_decode_final_head_reduce_kernel<<<1, kDecodeThreads, 0, stream>>>(
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
        arena_bytes + layout_bytes + scratch_bytes + kv_bytes + prompt_bytes +
        slots_bytes + sizeof(uint32_t) + kCublasWorkspaceBytes;
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
  session->intermediate = request->intermediate;
  session->vocab_size = request->vocab_size;
  session->layer_count = request->layer_count;
  session->max_context_tokens = request->max_context_tokens;
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
  session->verify_logits_bytes =
      vocab_size * static_cast<uint64_t>(kVerifyMaxDraftTokens) * sizeof(float);
  if (pack_cublas) {
    session->packed_qkv_bytes =
        packed_shape.qkv_elements_per_layer * request->layer_count *
        sizeof(uint16_t);
    session->packed_gate_up_bytes =
        packed_shape.gate_up_elements_per_layer * request->layer_count *
        sizeof(uint16_t);
  }
  session->kv_bytes =
      request->layer_count * static_cast<uint64_t>(request->max_context_tokens) *
      kv_hidden * sizeof(float) * 2;
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
    failure_stage = kCreateStageVerifyLogitsAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_verify_logits),
                     session->verify_logits_bytes);
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
  if (err == cudaSuccess) {
    failure_stage = kCreateStageCublasConfigure;
    err = configure_cublas(session->cublas, session->stream,
                           session->cublas_workspace,
                           kCublasWorkspaceBytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageStartEventCreate;
    err = cudaEventCreate(&session->device_start);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageStopEventCreate;
    err = cudaEventCreate(&session->device_stop);
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
    failure_stage = kCreateStagePackReplicas;
    err = pack_session_weight_replicas(session);
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
    err = cudaMemsetAsync(session->device_kv_keys, 0, session->kv_bytes / 2,
                          session->stream);
  if (err == cudaSuccess)
    err = cudaMemsetAsync(session->device_kv_values, 0, session->kv_bytes / 2,
                          session->stream);
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
          session->device_kv_values, session->device_slots);
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
      hf_decode_final_head_reduce_kernel<<<1, kDecodeThreads, 0,
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
    err = cudaMemsetAsync(session->device_kv_keys, 0, session->kv_bytes / 2,
                          session->stream);
  if (err == cudaSuccess)
    err = cudaMemsetAsync(session->device_kv_values, 0, session->kv_bytes / 2,
                          session->stream);
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
        decode_attention_chunks_for_cursor(session, session->active_cursor);
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

extern "C" int nerva_cuda_hf_decode_sequence_session_verify_block(
    const NervaCudaHfDecodeSequenceSessionVerifyBlockRequest *request,
    NervaCudaHfDecodeSequenceResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->draft_tokens == nullptr || request->output_tokens == nullptr ||
      request->draft_token_count == 0 ||
      request->output_token_capacity < request->draft_token_count ||
      request->draft_token_count > kVerifyMaxDraftTokens) {
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  if (!session->active_started || session->active_finished ||
      session->active_prompt_token_count == 0 || !use_cublas_layer_path(session)) {
    return -1;
  }
  for (uint32_t index = 0; index < request->draft_token_count; ++index) {
    if (request->draft_tokens[index] >= session->vocab_size) {
      return -1;
    }
  }
  const uint32_t prompt_count = session->active_prompt_token_count;
  const uint32_t slot_start =
      prompt_count - 1u + session->active_observed_tokens;
  if (slot_start >= session->max_context_tokens ||
      slot_start + request->draft_token_count > session->max_context_tokens) {
    return -1;
  }
  const uint32_t seed_token =
      session->active_observed_tokens == 0
          ? session->active_seed_token
          : session->host_slots[slot_start - 1u].token;
  fill_session_result_header(session, out, request->draft_token_count,
                             seed_token);

  cudaError_t err = cudaSuccess;
  bool device_timing_started = false;
  if (session->active_cursor == slot_start) {
    const uint32_t attention_chunks =
        decode_attention_chunks_for_cursor(session, session->active_cursor);
    err = ensure_session_graph(session, session->max_context_tokens,
                               prompt_count, session->active_has_eos_token,
                               session->active_eos_token, attention_chunks,
                               session->active_cursor, out);
    if (err == cudaSuccess) {
      err = cudaEventRecord(session->device_start, session->stream);
      device_timing_started = err == cudaSuccess;
    }
    if (err == cudaSuccess) {
      err = cudaGraphLaunch(session->cached_graph_exec, session->stream);
      if (err == cudaSuccess) {
        out->graph_replays += 1;
        out->graph_launches += 1;
        out->kernel_launches += out->graph_nodes == 0 ? 1 : out->graph_nodes;
      }
    }
  } else if (session->active_cursor != slot_start + 1u) {
    return -1;
  }

  const uint32_t feed_count = request->draft_token_count - 1u;
  if (err == cudaSuccess && feed_count != 0) {
    const uint64_t draft_bytes =
        static_cast<uint64_t>(feed_count) * sizeof(uint32_t);
    err = cudaMemcpyAsync(session->device_prompt_tokens, request->draft_tokens,
                          draft_bytes, cudaMemcpyHostToDevice, session->stream);
    out->h2d_bytes += draft_bytes;
  }
  if (err == cudaSuccess) {
    if (!device_timing_started) {
      err = cudaEventRecord(session->device_start, session->stream);
      device_timing_started = err == cudaSuccess;
    }
  }
  if (err == cudaSuccess && feed_count != 0) {
    err = launch_cublas_session_verify_draft(
        session, slot_start + 1u, feed_count, session->active_has_eos_token,
        session->active_eos_token, out);
  }
  if (err == cudaSuccess && device_timing_started) {
    err = cudaEventRecord(session->device_stop, session->stream);
  }
  const uint64_t slots_bytes =
      static_cast<uint64_t>(request->draft_token_count) *
      sizeof(NervaCudaSyntheticTokenSlot);
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
  if (err == cudaSuccess && device_timing_started) {
    float device_ms = 0.0f;
    err = cudaEventElapsedTime(&device_ms, session->device_start,
                               session->device_stop);
    if (err == cudaSuccess && device_ms > 0.0f) {
      out->device_elapsed_ns = static_cast<uint64_t>(device_ms * 1000000.0f);
      if (out->device_elapsed_ns == 0) out->device_elapsed_ns = 1;
    }
  }
  bool hit_eos = false;
  if (err == cudaSuccess) {
    NervaCudaSyntheticTokenSlot *observed_slots =
        session->host_slots + slot_start;
    bool valid_slots = true;
    for (uint32_t index = 0; index < request->draft_token_count; ++index) {
      const NervaCudaSyntheticTokenSlot &slot = observed_slots[index];
      if (slot.request_id != kRequestId || slot.sequence_id != kSequenceId ||
          slot.completion != kCompletionDeviceComplete ||
          slot.token_index != slot_start + index) {
        valid_slots = false;
        break;
      }
      request->output_tokens[index] = slot.token;
      out->observed_tokens = index + 1u;
      if (session->active_has_eos_token != 0 &&
          slot.token == session->active_eos_token) {
        hit_eos = true;
        break;
      }
      if (slot.token != request->draft_tokens[index]) {
        break;
      }
    }
    if (out->observed_tokens != 0 && valid_slots) {
      out->last_token = request->output_tokens[out->observed_tokens - 1u];
      out->observed_token_hash =
          hash_tokens(request->output_tokens, out->observed_tokens);
      out->resident_kv_bytes = session->kv_bytes;
      out->kv_tokens = slot_start + out->observed_tokens;
      out->device_arena_bytes = session_device_footprint(session);
      out->pinned_host_bytes = session->slots_bytes;
      out->host_causality_edges = 0;
      out->status = 0;
      session->active_observed_tokens += out->observed_tokens;
      session->active_cursor = slot_start + out->observed_tokens;
      session->active_finished =
          hit_eos || session->active_cursor >= session->max_context_tokens;
    }
  }
  if (err == cudaSuccess && out->status == 0) {
    hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(
        session->device_step, session->active_cursor);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
      err = cudaStreamSynchronize(session->stream);
      out->sync_calls += 1;
    }
  }
  if (err != cudaSuccess) {
    fail(out, err);
  }
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
