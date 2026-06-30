#include "nerva_cuda_api.h"
#include "deepseek_quant.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kHidden = 3;
constexpr uint32_t kIntermediate = 2;
constexpr uint32_t kNumExperts = 2;
constexpr uint32_t kTopK = 2;
constexpr float kSwigluLimit = 1.0f;
constexpr uint32_t kOutputValues = kHidden;
constexpr uint32_t kMegaMoeBlockK = 128;
constexpr uint32_t kMegaMoeGroupK = 32;
constexpr uint32_t kMegaMoeGroupsPerBlock =
    kMegaMoeBlockK / kMegaMoeGroupK;
constexpr uint32_t kMegaMoeExpertThreads = 128;

constexpr float kInput[kHidden] = {1.2f, -0.7f, 0.3f};
constexpr uint32_t kExpertIds[kTopK] = {1, 0};
constexpr float kExpertWeights[kTopK] = {0.75f, 0.25f};

constexpr float kWGate[kNumExperts * kIntermediate * kHidden] = {
    1.0f, -0.5f, 0.25f, -0.25f, 0.75f, 1.25f,
    0.5f, 0.2f,  -0.1f, -1.0f,  0.4f,  0.3f,
};
constexpr float kWUp[kNumExperts * kIntermediate * kHidden] = {
    -0.2f, 0.4f, 1.1f,  0.8f, -0.6f, 0.2f,
    1.5f, -0.3f, 0.1f, 0.7f, 0.6f,  -0.4f,
};
constexpr float kWDown[kNumExperts * kHidden * kIntermediate] = {
    0.3f, -0.2f, 0.4f,  0.1f, -0.5f, 0.2f,
    -0.7f, 0.6f, -0.1f, 0.25f, 0.35f, -0.45f,
};

struct DeviceMoeOutput {
  float output[kOutputValues];
};

__host__ __device__ float silu(float value) {
  return value / (1.0f + expf(-value));
}

__host__ __device__ float clamp_value(float value, float min_value, float max_value) {
  return fminf(fmaxf(value, min_value), max_value);
}

__host__ __device__ float swiglu(float gate, float up) {
  const float clamped_gate = fminf(gate, kSwigluLimit);
  const float clamped_up = clamp_value(up, -kSwigluLimit, kSwigluLimit);
  return silu(clamped_gate) * clamped_up;
}

__host__ __device__ float dot(const float *left, const float *right, uint32_t len) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < len; ++i) {
    sum += left[i] * right[i];
  }
  return sum;
}

__host__ __device__ void compute_deepseek_moe(float *output) {
  const float input[kHidden] = {1.2f, -0.7f, 0.3f};
  const uint32_t expert_ids[kTopK] = {1, 0};
  const float expert_weights[kTopK] = {0.75f, 0.25f};
  const float w_gate[kNumExperts * kIntermediate * kHidden] = {
      1.0f, -0.5f, 0.25f, -0.25f, 0.75f, 1.25f,
      0.5f, 0.2f,  -0.1f, -1.0f,  0.4f,  0.3f,
  };
  const float w_up[kNumExperts * kIntermediate * kHidden] = {
      -0.2f, 0.4f, 1.1f,  0.8f, -0.6f, 0.2f,
      1.5f, -0.3f, 0.1f, 0.7f, 0.6f,  -0.4f,
  };
  const float w_down[kNumExperts * kHidden * kIntermediate] = {
      0.3f, -0.2f, 0.4f,  0.1f, -0.5f, 0.2f,
      -0.7f, 0.6f, -0.1f, 0.25f, 0.35f, -0.45f,
  };

  for (uint32_t hidden = 0; hidden < kHidden; ++hidden) {
    output[hidden] = 0.0f;
  }

  float activation[kIntermediate];
  constexpr uint32_t expert_stride = kIntermediate * kHidden;
  constexpr uint32_t down_expert_stride = kHidden * kIntermediate;

  for (uint32_t rank = 0; rank < kTopK; ++rank) {
    const uint32_t expert = expert_ids[rank];
    const float route_weight = expert_weights[rank];
    const uint32_t expert_base = expert * expert_stride;
    const uint32_t down_base = expert * down_expert_stride;

    for (uint32_t row = 0; row < kIntermediate; ++row) {
      const uint32_t start = expert_base + row * kHidden;
      const float gate = dot(w_gate + start, input, kHidden);
      const float up = dot(w_up + start, input, kHidden);
      activation[row] = swiglu(gate, up);
    }

    for (uint32_t hidden = 0; hidden < kHidden; ++hidden) {
      const uint32_t start = down_base + hidden * kIntermediate;
      output[hidden] += route_weight * dot(w_down + start, activation, kIntermediate);
    }
  }
}

__global__ void deepseek_moe_smoke_kernel(DeviceMoeOutput *out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  compute_deepseek_moe(out->output);
}

__device__ float swiglu_dynamic(float gate,
                                float up,
                                uint32_t clamp_swiglu,
                                float swiglu_limit) {
  if (clamp_swiglu != 0) {
    gate = fminf(gate, swiglu_limit);
    up = clamp_value(up, -swiglu_limit, swiglu_limit);
  }
  return silu(gate) * up;
}

__device__ uint8_t f32_to_f8_e4m3fn_nearest(float value) {
  if (!isfinite(value)) {
    return 0x7fu;
  }
  uint8_t best_bits = 0;
  float best_diff = INFINITY;
  for (uint32_t bits = 0; bits < 256; ++bits) {
    const uint8_t candidate_bits = static_cast<uint8_t>(bits);
    const float candidate =
        nerva::deepseek::f8_e4m3fn_bits_to_f32(candidate_bits);
    if (!isfinite(candidate)) {
      continue;
    }
    const float diff = fabsf(candidate - value);
    if (diff < best_diff) {
      best_diff = diff;
      best_bits = candidate_bits;
    }
  }
  return best_bits;
}

__device__ uint32_t ceil_e8m0_exponent_for_scale(float scale) {
  const uint32_t bits = __float_as_uint(scale);
  uint32_t exp = (bits >> 23) & 0xffu;
  exp += (bits & 0x7fffffu) != 0u ? 1u : 0u;
  exp = exp < 1u ? 1u : exp;
  exp = exp > 254u ? 254u : exp;
  return exp;
}

__global__ void deepseek_megamoe_prepare_kernel(
    const float *hidden_states,
    const int64_t *topk_ids,
    const float *topk_weights,
    const uint8_t *is_padding,
    const int64_t *logical_to_physical_map,
    const uint32_t *logical_replica_count,
    uint8_t *x_fp8,
    uint32_t *x_scales,
    int64_t *topk_ids_out,
    float *topk_weights_out,
    uint32_t *expert_load_out,
    uint32_t num_tokens,
    uint32_t hidden_size,
    uint32_t top_k,
    uint32_t num_logical_experts,
    uint32_t map_slots,
    uint32_t expert_load_size,
    uint32_t record_expert_load,
    uint32_t hidden_blocks,
    int32_t *prepare_error) {
  const uint32_t token = blockIdx.x;
  const uint32_t hidden_block = blockIdx.y;
  const uint32_t lane = threadIdx.x;
  if (token >= num_tokens || hidden_block >= hidden_blocks ||
      lane >= kMegaMoeBlockK) {
    return;
  }

  __shared__ float group_amax[kMegaMoeGroupsPerBlock];
  __shared__ uint32_t group_exp[kMegaMoeGroupsPerBlock];

  if (lane < kMegaMoeGroupsPerBlock) {
    const uint32_t group_start =
        hidden_block * kMegaMoeBlockK + lane * kMegaMoeGroupK;
    float amax = 0.0f;
    for (uint32_t offset = 0; offset < kMegaMoeGroupK; ++offset) {
      const uint32_t hidden = group_start + offset;
      const float value =
          hidden < hidden_size ? hidden_states[token * hidden_size + hidden] : 0.0f;
      amax = fmaxf(amax, fabsf(value));
    }
    amax = fmaxf(amax, 1.0e-4f);
    const float scale = amax / 448.0f;
    group_amax[lane] = __uint_as_float(ceil_e8m0_exponent_for_scale(scale) << 23);
    group_exp[lane] = ceil_e8m0_exponent_for_scale(scale);
  }
  __syncthreads();

  const uint32_t hidden = hidden_block * kMegaMoeBlockK + lane;
  if (hidden < hidden_size) {
    const uint32_t group = lane / kMegaMoeGroupK;
    const float scale = group_amax[group];
    const float value = hidden_states[token * hidden_size + hidden] / scale;
    x_fp8[token * hidden_size + hidden] = f32_to_f8_e4m3fn_nearest(value);
  }

  if (lane == 0) {
    uint32_t packed = 0;
    for (uint32_t group = 0; group < kMegaMoeGroupsPerBlock; ++group) {
      packed |= (group_exp[group] & 0xffu) << (group * 8u);
    }
    x_scales[token * hidden_blocks + hidden_block] = packed;
  }

  if (hidden_block == 0 && lane < top_k) {
    const bool padding = is_padding != nullptr && is_padding[token] != 0u;
    const uint64_t route_offset = static_cast<uint64_t>(token) * top_k + lane;
    int64_t staged_id = padding ? -1 : topk_ids[route_offset];
    if (!padding && logical_to_physical_map != nullptr &&
        logical_replica_count != nullptr) {
      const bool valid_logical =
          staged_id >= 0 &&
          static_cast<uint64_t>(staged_id) < num_logical_experts;
      if (valid_logical) {
        const uint32_t logical_id = static_cast<uint32_t>(staged_id);
        uint32_t replica_count = logical_replica_count[logical_id];
        replica_count = replica_count > map_slots ? map_slots : replica_count;
        replica_count = replica_count == 0u ? 1u : replica_count;
        const uint32_t hashed = static_cast<uint32_t>(
            (static_cast<uint64_t>(token) * 2654435769ull) & 0xffffffffull);
        const uint32_t replica_idx = hashed % replica_count;
        const uint32_t map_index = logical_id * map_slots + replica_idx;
        staged_id = logical_to_physical_map[map_index];
      } else {
        staged_id = -1;
      }
    }
    topk_ids_out[route_offset] = staged_id;
    topk_weights_out[route_offset] = padding ? 0.0f : topk_weights[route_offset];
    if (record_expert_load != 0u && expert_load_out != nullptr &&
        staged_id >= 0 && static_cast<uint64_t>(staged_id) < expert_load_size) {
      atomicAdd(&expert_load_out[static_cast<uint32_t>(staged_id)], 1u);
    }
  }

  if (lane == 0 && hidden_block == 0) {
    *prepare_error = 0;
  }
}

__device__ float decode_megamoe_x(const uint8_t *x_fp8,
                                  const uint32_t *x_scales,
                                  uint32_t token,
                                  uint32_t hidden,
                                  uint32_t hidden_size,
                                  uint32_t hidden_blocks) {
  const uint32_t hidden_block = hidden / kMegaMoeBlockK;
  const uint32_t block_offset = hidden - hidden_block * kMegaMoeBlockK;
  const uint32_t group = block_offset / kMegaMoeGroupK;
  const uint32_t packed_scale = x_scales[token * hidden_blocks + hidden_block];
  const uint8_t scale_exp =
      static_cast<uint8_t>((packed_scale >> (group * 8u)) & 0xffu);
  const float scale = nerva::deepseek::e8m0_exponent_bits_to_f32(scale_exp);
  return nerva::deepseek::f8_e4m3fn_bits_to_f32(
             x_fp8[static_cast<uint64_t>(token) * hidden_size + hidden]) *
         scale;
}

__device__ float decode_megamoe_fp4_weight(const uint8_t *packed,
                                           const uint8_t *scales,
                                           uint64_t row_base_packed,
                                           uint64_t row_base_scales,
                                           uint32_t col) {
  const uint8_t byte = packed[row_base_packed + col / 2u];
  const uint8_t nibble = (col & 1u) == 0u ? (byte & 0x0fu) : (byte >> 4u);
  const uint8_t scale_exp = scales[row_base_scales + col / 32u];
  return nerva::deepseek::mxfp4_e2m1_nibble_to_f32(nibble) *
         nerva::deepseek::e8m0_exponent_bits_to_f32(scale_exp);
}

__global__ void deepseek_megamoe_gate_up_kernel(
    const uint8_t *x_fp8,
    const uint32_t *x_scales,
    const int64_t *topk_ids,
    const uint8_t *w13_packed,
    const uint8_t *w13_scales,
    float *activation,
    uint32_t num_tokens,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    uint32_t num_experts,
    uint32_t top_k,
    float swiglu_limit,
    uint32_t hidden_blocks,
    int32_t *expert_error) {
  if (*expert_error != 0) {
    return;
  }

  const uint32_t tid = threadIdx.x;
  const uint32_t route = blockIdx.x;
  const uint32_t intermediate = blockIdx.y;
  if (route >= num_tokens * top_k || intermediate >= intermediate_size) {
    return;
  }
  const uint32_t token = route / top_k;
  const uint64_t w13_rows = static_cast<uint64_t>(intermediate_size) * 2u;
  const uint64_t w13_packed_cols = hidden_size / 2u;
  const uint64_t w13_scale_cols = hidden_size / 32u;

  const int64_t expert_id_signed = topk_ids[route];
  if (expert_id_signed < 0) {
    if (tid == 0) {
      activation[static_cast<uint64_t>(route) * intermediate_size +
                 intermediate] = 0.0f;
    }
    return;
  }
  const uint32_t expert_id = static_cast<uint32_t>(expert_id_signed);
  if (expert_id >= num_experts) {
    *expert_error = -2;
    return;
  }

  float gate = 0.0f;
  float up = 0.0f;
  const uint64_t gate_row =
      static_cast<uint64_t>(expert_id) * w13_rows + intermediate;
  const uint64_t up_row = static_cast<uint64_t>(expert_id) * w13_rows +
                          intermediate_size + intermediate;
  const uint64_t gate_packed_base = gate_row * w13_packed_cols;
  const uint64_t up_packed_base = up_row * w13_packed_cols;
  const uint64_t gate_scale_base = gate_row * w13_scale_cols;
  const uint64_t up_scale_base = up_row * w13_scale_cols;
  for (uint32_t hidden = tid; hidden < hidden_size; hidden += blockDim.x) {
    const float x = decode_megamoe_x(
        x_fp8, x_scales, token, hidden, hidden_size, hidden_blocks);
    gate += x * decode_megamoe_fp4_weight(w13_packed,
                                          w13_scales,
                                          gate_packed_base,
                                          gate_scale_base,
                                          hidden);
    up += x * decode_megamoe_fp4_weight(w13_packed,
                                        w13_scales,
                                        up_packed_base,
                                        up_scale_base,
                                        hidden);
  }
  __shared__ float gate_partial[kMegaMoeExpertThreads];
  __shared__ float up_partial[kMegaMoeExpertThreads];
  gate_partial[tid] = gate;
  up_partial[tid] = up;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2u; stride > 0; stride >>= 1u) {
    if (tid < stride) {
      gate_partial[tid] += gate_partial[tid + stride];
      up_partial[tid] += up_partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    activation[static_cast<uint64_t>(route) * intermediate_size +
               intermediate] =
        swiglu_dynamic(gate_partial[0], up_partial[0], 1u, swiglu_limit);
  }
}

__global__ void deepseek_megamoe_down_kernel(const int64_t *topk_ids,
                                             const float *topk_weights,
                                             const uint8_t *w2_packed,
                                             const uint8_t *w2_scales,
                                             const float *activation,
                                             float *output,
                                             uint32_t num_tokens,
                                             uint32_t hidden_size,
                                             uint32_t intermediate_size,
                                             uint32_t num_experts,
                                             uint32_t top_k,
                                             int32_t *expert_error) {
  if (*expert_error != 0) {
    return;
  }

  const uint32_t tid = threadIdx.x;
  const uint32_t output_index = blockIdx.x;
  if (output_index >= num_tokens * hidden_size) {
    return;
  }
  const uint32_t token = output_index / hidden_size;
  const uint32_t out_hidden = output_index - token * hidden_size;
  const uint64_t w2_packed_cols = intermediate_size / 2u;
  const uint64_t w2_scale_cols = intermediate_size / 32u;
  float token_sum = 0.0f;

  for (uint32_t rank = 0; rank < top_k; ++rank) {
    const uint64_t route_offset = static_cast<uint64_t>(token) * top_k + rank;
    const int64_t expert_id_signed = topk_ids[route_offset];
    if (expert_id_signed < 0) {
      continue;
    }
    const uint32_t expert_id = static_cast<uint32_t>(expert_id_signed);
    if (expert_id >= num_experts) {
      *expert_error = -2;
      return;
    }
    const float route_weight = topk_weights[route_offset];
    const uint64_t w2_row =
        static_cast<uint64_t>(expert_id) * hidden_size + out_hidden;
    const uint64_t w2_packed_base = w2_row * w2_packed_cols;
    const uint64_t w2_scale_base = w2_row * w2_scale_cols;
    float routed_sum = 0.0f;
    for (uint32_t intermediate = tid; intermediate < intermediate_size;
         intermediate += blockDim.x) {
      routed_sum +=
          activation[route_offset * intermediate_size + intermediate] *
          decode_megamoe_fp4_weight(w2_packed,
                                    w2_scales,
                                    w2_packed_base,
                                    w2_scale_base,
                                    intermediate);
    }
    token_sum += route_weight * routed_sum;
  }

  __shared__ float partial[kMegaMoeExpertThreads];
  partial[tid] = token_sum;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2u; stride > 0; stride >>= 1u) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[output_index] = partial[0];
  }
}

__global__ void deepseek_moe_forward_kernel(const float *input,
                                            const uint32_t *expert_ids,
                                            const float *expert_weights,
                                            const float *w_gate,
                                            const float *w_up,
                                            const float *w_down,
                                            float *activation,
                                            float *output,
                                            uint32_t hidden_size,
                                            uint32_t intermediate_size,
                                            uint32_t num_experts,
                                            uint32_t top_k,
                                            uint32_t clamp_swiglu,
                                            float swiglu_limit,
                                            int32_t *moe_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (hidden_size == 0 || intermediate_size == 0 || num_experts == 0 ||
      top_k == 0) {
    *moe_error = -1;
    return;
  }

  for (uint32_t hidden = 0; hidden < hidden_size; ++hidden) {
    output[hidden] = 0.0f;
  }

  const uint32_t expert_stride = intermediate_size * hidden_size;
  const uint32_t down_expert_stride = hidden_size * intermediate_size;
  for (uint32_t rank = 0; rank < top_k; ++rank) {
    const uint32_t expert = expert_ids[rank];
    if (expert >= num_experts) {
      *moe_error = -2;
      return;
    }
    const float route_weight = expert_weights[rank];
    const uint32_t expert_base = expert * expert_stride;
    const uint32_t down_base = expert * down_expert_stride;

    for (uint32_t row = 0; row < intermediate_size; ++row) {
      const uint32_t start = expert_base + row * hidden_size;
      const float gate = dot(w_gate + start, input, hidden_size);
      const float up = dot(w_up + start, input, hidden_size);
      activation[row] = swiglu_dynamic(gate, up, clamp_swiglu, swiglu_limit);
    }

    for (uint32_t hidden = 0; hidden < hidden_size; ++hidden) {
      const uint32_t start = down_base + hidden * intermediate_size;
      output[hidden] +=
          route_weight * dot(w_down + start, activation, intermediate_size);
    }
  }
  *moe_error = 0;
}

uint32_t f32_bits(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint64_t mix_hash_u32(uint64_t hash, uint32_t value) {
  for (uint32_t byte = 0; byte < 4; ++byte) {
    hash ^= static_cast<uint8_t>((value >> (byte * 8)) & 0xffu);
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t hash_bytes(const void *data, uint64_t bytes) {
  const auto *ptr = static_cast<const uint8_t *>(data);
  uint64_t hash = 1469598103934665603ull;
  for (uint64_t i = 0; i < bytes; ++i) {
    hash ^= ptr[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t hash_f32_bits(const float *values, uint32_t len) {
  uint64_t hash = 1469598103934665603ull;
  for (uint32_t i = 0; i < len; ++i) {
    hash = mix_hash_u32(hash, f32_bits(values[i]));
  }
  return hash;
}

void compare_outputs(const float *actual,
                     const float *expected,
                     uint32_t len,
                     uint64_t *mismatches,
                     float *max_abs_diff) {
  *mismatches = 0;
  *max_abs_diff = 0.0f;
  for (uint32_t i = 0; i < len; ++i) {
    const float diff = fabsf(actual[i] - expected[i]);
    if (diff > *max_abs_diff) {
      *max_abs_diff = diff;
    }
    if (diff > 1e-6f) {
      *mismatches += 1;
    }
  }
}

void clear_result(NervaCudaDeepSeekMoeSmokeResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->hidden_size = kHidden;
  out->intermediate_size = kIntermediate;
  out->num_experts = kNumExperts;
  out->top_k = kTopK;
  out->swiglu_limit = kSwigluLimit;
  memcpy(out->expert_ids, kExpertIds, sizeof(out->expert_ids));
  memcpy(out->expert_weights, kExpertWeights, sizeof(out->expert_weights));
}

int fail(NervaCudaDeepSeekMoeSmokeResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

void clear_forward_result(const NervaCudaDeepSeekMoeForwardRequest *request,
                          NervaCudaDeepSeekMoeForwardResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->moe_error = -1;
  if (request != nullptr) {
    out->hidden_size = request->hidden_size;
    out->intermediate_size = request->intermediate_size;
    out->num_experts = request->num_experts;
    out->top_k = request->top_k;
    out->clamp_swiglu = request->clamp_swiglu;
    out->swiglu_limit = request->swiglu_limit;
  }
}

void clear_megamoe_prepare_result(
    const NervaCudaDeepSeekMegaMoePrepareRequest *request,
    NervaCudaDeepSeekMegaMoePrepareResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->prepare_error = -1;
  if (request != nullptr) {
    out->num_tokens = request->num_tokens;
    out->hidden_size = request->hidden_size;
    out->top_k = request->top_k;
    out->hidden_blocks =
        request->hidden_size == 0
            ? 0
            : (request->hidden_size + kMegaMoeBlockK - 1) / kMegaMoeBlockK;
  }
}

void clear_megamoe_experts_result(
    const NervaCudaDeepSeekMegaMoeExpertsRequest *request,
    NervaCudaDeepSeekMegaMoeExpertsResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->expert_error = -1;
  if (request != nullptr) {
    out->num_tokens = request->num_tokens;
    out->hidden_size = request->hidden_size;
    out->intermediate_size = request->intermediate_size;
    out->num_experts = request->num_experts;
    out->top_k = request->top_k;
  }
}

int fail_forward(NervaCudaDeepSeekMoeForwardResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

int fail_megamoe_prepare(NervaCudaDeepSeekMegaMoePrepareResult *out,
                         cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

int fail_megamoe_experts(NervaCudaDeepSeekMegaMoeExpertsResult *out,
                         cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

bool validate_forward_request(const NervaCudaDeepSeekMoeForwardRequest *request) {
  return request != nullptr && request->input != nullptr &&
         request->expert_ids != nullptr && request->expert_weights != nullptr &&
         request->w_gate != nullptr && request->w_up != nullptr &&
         request->w_down != nullptr && request->output != nullptr &&
         request->hidden_size > 0 && request->intermediate_size > 0 &&
         request->num_experts > 0 && request->top_k > 0;
}

bool validate_megamoe_prepare_request(
    const NervaCudaDeepSeekMegaMoePrepareRequest *request) {
  if (request == nullptr || request->hidden_states == nullptr ||
      request->topk_ids == nullptr || request->topk_weights == nullptr ||
      request->x_fp8 == nullptr || request->x_scales == nullptr ||
      request->topk_ids_out == nullptr ||
      request->topk_weights_out == nullptr || request->num_tokens == 0 ||
      request->hidden_size == 0 || request->top_k == 0 ||
      request->hidden_size % kMegaMoeBlockK != 0) {
    return false;
  }
  const bool has_mapping =
      request->logical_to_physical_map != nullptr ||
      request->logical_replica_count != nullptr ||
      request->num_logical_experts != 0 || request->map_slots != 0 ||
      request->record_expert_load != 0 || request->expert_load_size != 0;
  if (!has_mapping) {
    return true;
  }
  if (request->logical_to_physical_map == nullptr ||
      request->logical_replica_count == nullptr ||
      request->num_logical_experts == 0 || request->map_slots == 0) {
    return false;
  }
  if (request->record_expert_load != 0 &&
      (request->expert_load_size == 0 || request->expert_load_out == nullptr)) {
    return false;
  }
  return true;
}

bool validate_megamoe_experts_request(
    const NervaCudaDeepSeekMegaMoeExpertsRequest *request) {
  return request != nullptr && request->x_fp8 != nullptr &&
         request->x_scales != nullptr && request->topk_ids != nullptr &&
         request->topk_weights != nullptr && request->w13_packed != nullptr &&
         request->w13_scales != nullptr && request->w2_packed != nullptr &&
         request->w2_scales != nullptr && request->output != nullptr &&
         request->num_tokens > 0 && request->hidden_size > 0 &&
         request->intermediate_size > 0 && request->num_experts > 0 &&
         request->top_k > 0 && request->hidden_size % kMegaMoeBlockK == 0 &&
         request->intermediate_size % 32u == 0;
}

}  // namespace

extern "C" int nerva_cuda_deepseek_moe_smoke(
    NervaCudaDeepSeekMoeSmokeResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);

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

  DeviceMoeOutput *device_output = nullptr;
  DeviceMoeOutput *host_output = nullptr;
  cudaStream_t stream = nullptr;
  const uint64_t output_bytes = sizeof(DeviceMoeOutput);

  err = cudaMalloc(reinterpret_cast<void **>(&device_output), output_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes = output_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&host_output),
                      output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  deepseek_moe_smoke_kernel<<<1, 1, 0, stream>>>(device_output);
  out->kernel_launches += 1;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(host_output,
                        device_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(out->output, host_output->output, sizeof(out->output));
  out->output_hash = hash_f32_bits(out->output, kOutputValues);

  float expected[kOutputValues];
  compute_deepseek_moe(expected);
  compare_outputs(out->output,
                  expected,
                  kOutputValues,
                  &out->mismatches,
                  &out->max_abs_diff);
  out->status = out->mismatches == 0 ? 0 : -1;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (host_output != nullptr) cudaFreeHost(host_output);
  if (device_output != nullptr) cudaFree(device_output);

  if (err != cudaSuccess) {
    return fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_moe_forward(
    const NervaCudaDeepSeekMoeForwardRequest *request,
    NervaCudaDeepSeekMoeForwardResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_forward_result(request, out);
  if (!validate_forward_request(request)) {
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_forward(out, err);
  }
  if (out->device_count <= 0) {
    return fail_forward(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_forward(out, err);
  }

  float *d_input = nullptr;
  uint32_t *d_expert_ids = nullptr;
  float *d_expert_weights = nullptr;
  float *d_w_gate = nullptr;
  float *d_w_up = nullptr;
  float *d_w_down = nullptr;
  float *d_activation = nullptr;
  float *d_output = nullptr;
  float *h_output = nullptr;
  int32_t *d_moe_error = nullptr;
  int32_t h_moe_error = -1;
  cudaStream_t stream = nullptr;

  const uint64_t hidden = request->hidden_size;
  const uint64_t intermediate = request->intermediate_size;
  const uint64_t num_experts = request->num_experts;
  const uint64_t top_k = request->top_k;
  const uint64_t input_bytes = sizeof(float) * hidden;
  const uint64_t expert_ids_bytes = sizeof(uint32_t) * top_k;
  const uint64_t expert_weights_bytes = sizeof(float) * top_k;
  const uint64_t expert_matrix_bytes =
      sizeof(float) * num_experts * intermediate * hidden;
  const uint64_t down_bytes = sizeof(float) * num_experts * hidden * intermediate;
  const uint64_t activation_bytes = sizeof(float) * intermediate;
  const uint64_t output_bytes = sizeof(float) * hidden;
  const uint64_t moe_error_bytes = sizeof(int32_t);

  err = cudaMalloc(reinterpret_cast<void **>(&d_input), input_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_expert_ids), expert_ids_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_expert_weights),
                   expert_weights_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_w_gate), expert_matrix_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_w_up), expert_matrix_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_w_down), down_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_activation), activation_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output), output_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_moe_error), moe_error_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes = input_bytes + expert_ids_bytes +
                            expert_weights_bytes + expert_matrix_bytes * 2 +
                            down_bytes + activation_bytes + output_bytes +
                            moe_error_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output),
                      output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_input,
                        request->input,
                        input_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_expert_ids,
                        request->expert_ids,
                        expert_ids_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_expert_weights,
                        request->expert_weights,
                        expert_weights_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_w_gate,
                        request->w_gate,
                        expert_matrix_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_w_up,
                        request->w_up,
                        expert_matrix_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_w_down,
                        request->w_down,
                        down_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_moe_error,
                        &h_moe_error,
                        moe_error_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = input_bytes + expert_ids_bytes + expert_weights_bytes +
                   expert_matrix_bytes * 2 + down_bytes + moe_error_bytes;

  deepseek_moe_forward_kernel<<<1, 1, 0, stream>>>(
      d_input,
      d_expert_ids,
      d_expert_weights,
      d_w_gate,
      d_w_up,
      d_w_down,
      d_activation,
      d_output,
      request->hidden_size,
      request->intermediate_size,
      request->num_experts,
      request->top_k,
      request->clamp_swiglu,
      request->swiglu_limit,
      d_moe_error);
  out->kernel_launches += 1;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_output,
                        d_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(&h_moe_error,
                        d_moe_error,
                        moe_error_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_bytes + moe_error_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  out->moe_error = h_moe_error;
  if (h_moe_error == 0) {
    memcpy(request->output, h_output, output_bytes);
    out->output_hash =
        hash_f32_bits(request->output, request->hidden_size);
    out->status = 0;
  }

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output != nullptr) cudaFreeHost(h_output);
  if (d_moe_error != nullptr) cudaFree(d_moe_error);
  if (d_output != nullptr) cudaFree(d_output);
  if (d_activation != nullptr) cudaFree(d_activation);
  if (d_w_down != nullptr) cudaFree(d_w_down);
  if (d_w_up != nullptr) cudaFree(d_w_up);
  if (d_w_gate != nullptr) cudaFree(d_w_gate);
  if (d_expert_weights != nullptr) cudaFree(d_expert_weights);
  if (d_expert_ids != nullptr) cudaFree(d_expert_ids);
  if (d_input != nullptr) cudaFree(d_input);

  if (err != cudaSuccess) {
    return fail_forward(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_megamoe_prepare(
    const NervaCudaDeepSeekMegaMoePrepareRequest *request,
    NervaCudaDeepSeekMegaMoePrepareResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_megamoe_prepare_result(request, out);
  if (!validate_megamoe_prepare_request(request)) {
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_megamoe_prepare(out, err);
  }
  if (out->device_count <= 0) {
    return fail_megamoe_prepare(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_megamoe_prepare(out, err);
  }

  float *d_hidden = nullptr;
  int64_t *d_topk_ids = nullptr;
  float *d_topk_weights = nullptr;
  uint8_t *d_is_padding = nullptr;
  int64_t *d_logical_to_physical_map = nullptr;
  uint32_t *d_logical_replica_count = nullptr;
  uint32_t *d_expert_load_out = nullptr;
  uint8_t *d_x_fp8 = nullptr;
  uint32_t *d_x_scales = nullptr;
  int64_t *d_topk_ids_out = nullptr;
  float *d_topk_weights_out = nullptr;
  int32_t *d_prepare_error = nullptr;
  uint8_t *h_x_fp8 = nullptr;
  uint32_t *h_x_scales = nullptr;
  int64_t *h_topk_ids_out = nullptr;
  float *h_topk_weights_out = nullptr;
  uint32_t *h_expert_load_out = nullptr;
  int32_t h_prepare_error = -1;
  cudaStream_t stream = nullptr;
  dim3 grid(1, 1, 1);

  const uint64_t tokens = request->num_tokens;
  const uint64_t hidden = request->hidden_size;
  const uint64_t top_k = request->top_k;
  const uint64_t hidden_blocks = hidden / kMegaMoeBlockK;
  const bool mapping_enabled =
      request->logical_to_physical_map != nullptr &&
      request->logical_replica_count != nullptr;
  const bool record_expert_load = request->record_expert_load != 0u;
  const uint64_t hidden_bytes = sizeof(float) * tokens * hidden;
  const uint64_t topk_ids_bytes = sizeof(int64_t) * tokens * top_k;
  const uint64_t topk_weights_bytes = sizeof(float) * tokens * top_k;
  const uint64_t padding_bytes =
      request->is_padding == nullptr ? 0 : sizeof(uint8_t) * tokens;
  const uint64_t map_bytes =
      mapping_enabled
          ? sizeof(int64_t) * request->num_logical_experts * request->map_slots
          : 0;
  const uint64_t replica_count_bytes =
      mapping_enabled ? sizeof(uint32_t) * request->num_logical_experts : 0;
  const uint64_t expert_load_bytes =
      record_expert_load ? sizeof(uint32_t) * request->expert_load_size : 0;
  const uint64_t x_fp8_bytes = sizeof(uint8_t) * tokens * hidden;
  const uint64_t x_scales_bytes = sizeof(uint32_t) * tokens * hidden_blocks;
  const uint64_t prepare_error_bytes = sizeof(int32_t);

  err = cudaMalloc(reinterpret_cast<void **>(&d_hidden), hidden_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_topk_ids), topk_ids_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_topk_weights),
                   topk_weights_bytes);
  if (err != cudaSuccess) goto cleanup;
  if (padding_bytes > 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&d_is_padding), padding_bytes);
    if (err != cudaSuccess) goto cleanup;
  }
  if (map_bytes > 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&d_logical_to_physical_map),
                     map_bytes);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(reinterpret_cast<void **>(&d_logical_replica_count),
                     replica_count_bytes);
    if (err != cudaSuccess) goto cleanup;
  }
  if (expert_load_bytes > 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&d_expert_load_out),
                     expert_load_bytes);
    if (err != cudaSuccess) goto cleanup;
  }
  err = cudaMalloc(reinterpret_cast<void **>(&d_x_fp8), x_fp8_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_x_scales), x_scales_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_topk_ids_out), topk_ids_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_topk_weights_out),
                   topk_weights_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_prepare_error),
                   prepare_error_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      hidden_bytes + topk_ids_bytes + topk_weights_bytes + padding_bytes +
      map_bytes + replica_count_bytes + expert_load_bytes +
      x_fp8_bytes + x_scales_bytes + topk_ids_bytes + topk_weights_bytes +
      prepare_error_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_x_fp8),
                      x_fp8_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_x_scales),
                      x_scales_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_topk_ids_out),
                      topk_ids_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_topk_weights_out),
                      topk_weights_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  if (expert_load_bytes > 0) {
    err = cudaHostAlloc(reinterpret_cast<void **>(&h_expert_load_out),
                        expert_load_bytes,
                        cudaHostAllocDefault);
    if (err != cudaSuccess) goto cleanup;
  }
  out->pinned_host_bytes =
      x_fp8_bytes + x_scales_bytes + topk_ids_bytes + topk_weights_bytes +
      expert_load_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_hidden,
                        request->hidden_states,
                        hidden_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_topk_ids,
                        request->topk_ids,
                        topk_ids_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_topk_weights,
                        request->topk_weights,
                        topk_weights_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  if (padding_bytes > 0) {
    err = cudaMemcpyAsync(d_is_padding,
                          request->is_padding,
                          padding_bytes,
                          cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) goto cleanup;
  }
  if (map_bytes > 0) {
    err = cudaMemcpyAsync(d_logical_to_physical_map,
                          request->logical_to_physical_map,
                          map_bytes,
                          cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpyAsync(d_logical_replica_count,
                          request->logical_replica_count,
                          replica_count_bytes,
                          cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) goto cleanup;
  }
  if (expert_load_bytes > 0) {
    err = cudaMemsetAsync(d_expert_load_out, 0, expert_load_bytes, stream);
    if (err != cudaSuccess) goto cleanup;
  }
  err = cudaMemcpyAsync(d_prepare_error,
                        &h_prepare_error,
                        prepare_error_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes =
      hidden_bytes + topk_ids_bytes + topk_weights_bytes + padding_bytes +
      map_bytes + replica_count_bytes + prepare_error_bytes;

  grid = dim3(request->num_tokens, static_cast<uint32_t>(hidden_blocks), 1);
  deepseek_megamoe_prepare_kernel<<<grid, kMegaMoeBlockK, 0, stream>>>(
      d_hidden,
      d_topk_ids,
      d_topk_weights,
      d_is_padding,
      d_logical_to_physical_map,
      d_logical_replica_count,
      d_x_fp8,
      d_x_scales,
      d_topk_ids_out,
      d_topk_weights_out,
      d_expert_load_out,
      request->num_tokens,
      request->hidden_size,
      request->top_k,
      request->num_logical_experts,
      request->map_slots,
      request->expert_load_size,
      request->record_expert_load,
      static_cast<uint32_t>(hidden_blocks),
      d_prepare_error);
  out->kernel_launches += 1;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_x_fp8,
                        d_x_fp8,
                        x_fp8_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_x_scales,
                        d_x_scales,
                        x_scales_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_topk_ids_out,
                        d_topk_ids_out,
                        topk_ids_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_topk_weights_out,
                        d_topk_weights_out,
                        topk_weights_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(&h_prepare_error,
                        d_prepare_error,
                        prepare_error_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  if (expert_load_bytes > 0) {
    err = cudaMemcpyAsync(h_expert_load_out,
                          d_expert_load_out,
                          expert_load_bytes,
                          cudaMemcpyDeviceToHost,
                          stream);
    if (err != cudaSuccess) goto cleanup;
  }
  out->d2h_bytes =
      x_fp8_bytes + x_scales_bytes + topk_ids_bytes + topk_weights_bytes +
      expert_load_bytes + prepare_error_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  out->prepare_error = h_prepare_error;
  if (h_prepare_error == 0) {
    memcpy(request->x_fp8, h_x_fp8, x_fp8_bytes);
    memcpy(request->x_scales, h_x_scales, x_scales_bytes);
    memcpy(request->topk_ids_out, h_topk_ids_out, topk_ids_bytes);
    memcpy(request->topk_weights_out, h_topk_weights_out, topk_weights_bytes);
    if (expert_load_bytes > 0) {
      memcpy(request->expert_load_out, h_expert_load_out, expert_load_bytes);
      out->expert_load_hash =
          hash_bytes(request->expert_load_out, expert_load_bytes);
    }
    out->x_fp8_hash = hash_bytes(request->x_fp8, x_fp8_bytes);
    out->x_scales_hash = hash_bytes(request->x_scales, x_scales_bytes);
    out->topk_hash = hash_bytes(request->topk_ids_out, topk_ids_bytes);
    out->topk_hash =
        hash_bytes(request->topk_weights_out, topk_weights_bytes) ^ out->topk_hash;
    out->status = 0;
  }

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_expert_load_out != nullptr) cudaFreeHost(h_expert_load_out);
  if (h_topk_weights_out != nullptr) cudaFreeHost(h_topk_weights_out);
  if (h_topk_ids_out != nullptr) cudaFreeHost(h_topk_ids_out);
  if (h_x_scales != nullptr) cudaFreeHost(h_x_scales);
  if (h_x_fp8 != nullptr) cudaFreeHost(h_x_fp8);
  if (d_prepare_error != nullptr) cudaFree(d_prepare_error);
  if (d_topk_weights_out != nullptr) cudaFree(d_topk_weights_out);
  if (d_topk_ids_out != nullptr) cudaFree(d_topk_ids_out);
  if (d_x_scales != nullptr) cudaFree(d_x_scales);
  if (d_x_fp8 != nullptr) cudaFree(d_x_fp8);
  if (d_expert_load_out != nullptr) cudaFree(d_expert_load_out);
  if (d_logical_replica_count != nullptr) cudaFree(d_logical_replica_count);
  if (d_logical_to_physical_map != nullptr) cudaFree(d_logical_to_physical_map);
  if (d_is_padding != nullptr) cudaFree(d_is_padding);
  if (d_topk_weights != nullptr) cudaFree(d_topk_weights);
  if (d_topk_ids != nullptr) cudaFree(d_topk_ids);
  if (d_hidden != nullptr) cudaFree(d_hidden);

  if (err != cudaSuccess) {
    return fail_megamoe_prepare(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_megamoe_experts(
    const NervaCudaDeepSeekMegaMoeExpertsRequest *request,
    NervaCudaDeepSeekMegaMoeExpertsResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_megamoe_experts_result(request, out);
  if (!validate_megamoe_experts_request(request)) {
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_megamoe_experts(out, err);
  }
  if (out->device_count <= 0) {
    return fail_megamoe_experts(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_megamoe_experts(out, err);
  }

  uint8_t *d_x_fp8 = nullptr;
  uint32_t *d_x_scales = nullptr;
  int64_t *d_topk_ids = nullptr;
  float *d_topk_weights = nullptr;
  uint8_t *d_w13_packed = nullptr;
  uint8_t *d_w13_scales = nullptr;
  uint8_t *d_w2_packed = nullptr;
  uint8_t *d_w2_scales = nullptr;
  float *d_activation = nullptr;
  float *d_output = nullptr;
  int32_t *d_expert_error = nullptr;
  float *h_output = nullptr;
  int32_t h_expert_error = 0;
  cudaStream_t stream = nullptr;

  const uint64_t tokens = request->num_tokens;
  const uint64_t hidden = request->hidden_size;
  const uint64_t intermediate = request->intermediate_size;
  const uint64_t experts = request->num_experts;
  const uint64_t top_k = request->top_k;
  const uint64_t hidden_blocks = hidden / kMegaMoeBlockK;
  const uint64_t x_fp8_bytes = sizeof(uint8_t) * tokens * hidden;
  const uint64_t x_scales_bytes = sizeof(uint32_t) * tokens * hidden_blocks;
  const uint64_t topk_ids_bytes = sizeof(int64_t) * tokens * top_k;
  const uint64_t topk_weights_bytes = sizeof(float) * tokens * top_k;
  const uint64_t w13_rows = experts * 2u * intermediate;
  const uint64_t w13_packed_bytes = sizeof(uint8_t) * w13_rows * (hidden / 2u);
  const uint64_t w13_scales_bytes =
      sizeof(uint8_t) * w13_rows * (hidden / 32u);
  const uint64_t w2_rows = experts * hidden;
  const uint64_t w2_packed_bytes =
      sizeof(uint8_t) * w2_rows * (intermediate / 2u);
  const uint64_t w2_scales_bytes =
      sizeof(uint8_t) * w2_rows * (intermediate / 32u);
  const uint64_t activation_values = tokens * top_k * intermediate;
  const uint64_t activation_bytes = sizeof(float) * activation_values;
  const uint64_t output_values = tokens * hidden;
  const uint64_t output_bytes = sizeof(float) * output_values;
  const uint64_t expert_error_bytes = sizeof(int32_t);

  if (tokens * top_k > UINT32_MAX || intermediate > UINT32_MAX ||
      output_values > UINT32_MAX) {
    out->expert_error = -3;
    return -1;
  }

  err = cudaMalloc(reinterpret_cast<void **>(&d_x_fp8), x_fp8_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_x_scales), x_scales_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_topk_ids), topk_ids_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_topk_weights),
                   topk_weights_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_w13_packed),
                   w13_packed_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_w13_scales),
                   w13_scales_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_w2_packed), w2_packed_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_w2_scales), w2_scales_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_activation), activation_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output), output_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_expert_error),
                   expert_error_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      x_fp8_bytes + x_scales_bytes + topk_ids_bytes + topk_weights_bytes +
      w13_packed_bytes + w13_scales_bytes + w2_packed_bytes +
      w2_scales_bytes + activation_bytes + output_bytes + expert_error_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output),
                      output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_x_fp8,
                        request->x_fp8,
                        x_fp8_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_x_scales,
                        request->x_scales,
                        x_scales_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_topk_ids,
                        request->topk_ids,
                        topk_ids_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_topk_weights,
                        request->topk_weights,
                        topk_weights_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_w13_packed,
                        request->w13_packed,
                        w13_packed_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_w13_scales,
                        request->w13_scales,
                        w13_scales_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_w2_packed,
                        request->w2_packed,
                        w2_packed_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_w2_scales,
                        request->w2_scales,
                        w2_scales_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_expert_error,
                        &h_expert_error,
                        expert_error_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes =
      x_fp8_bytes + x_scales_bytes + topk_ids_bytes + topk_weights_bytes +
      w13_packed_bytes + w13_scales_bytes + w2_packed_bytes +
      w2_scales_bytes + expert_error_bytes;

  deepseek_megamoe_gate_up_kernel<<<
      dim3(static_cast<uint32_t>(tokens * top_k),
           static_cast<uint32_t>(intermediate),
           1),
      kMegaMoeExpertThreads,
      0,
      stream>>>(
      d_x_fp8,
      d_x_scales,
      d_topk_ids,
      d_w13_packed,
      d_w13_scales,
      d_activation,
      request->num_tokens,
      request->hidden_size,
      request->intermediate_size,
      request->num_experts,
      request->top_k,
      request->swiglu_limit,
      static_cast<uint32_t>(hidden_blocks),
      d_expert_error);
  out->kernel_launches += 1;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  deepseek_megamoe_down_kernel<<<
      static_cast<uint32_t>(output_values),
      kMegaMoeExpertThreads,
      0,
      stream>>>(d_topk_ids,
                d_topk_weights,
                d_w2_packed,
                d_w2_scales,
                d_activation,
                d_output,
                request->num_tokens,
                request->hidden_size,
                request->intermediate_size,
                request->num_experts,
                request->top_k,
                d_expert_error);
  out->kernel_launches += 1;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_output,
                        d_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(&h_expert_error,
                        d_expert_error,
                        expert_error_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_bytes + expert_error_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  out->expert_error = h_expert_error;
  if (h_expert_error == 0) {
    memcpy(request->output, h_output, output_bytes);
    out->output_hash = hash_f32_bits(
        request->output,
        static_cast<uint32_t>(output_values));
    out->status = 0;
  }

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output != nullptr) cudaFreeHost(h_output);
  if (d_expert_error != nullptr) cudaFree(d_expert_error);
  if (d_output != nullptr) cudaFree(d_output);
  if (d_activation != nullptr) cudaFree(d_activation);
  if (d_w2_scales != nullptr) cudaFree(d_w2_scales);
  if (d_w2_packed != nullptr) cudaFree(d_w2_packed);
  if (d_w13_scales != nullptr) cudaFree(d_w13_scales);
  if (d_w13_packed != nullptr) cudaFree(d_w13_packed);
  if (d_topk_weights != nullptr) cudaFree(d_topk_weights);
  if (d_topk_ids != nullptr) cudaFree(d_topk_ids);
  if (d_x_scales != nullptr) cudaFree(d_x_scales);
  if (d_x_fp8 != nullptr) cudaFree(d_x_fp8);

  if (err != cudaSuccess) {
    return fail_megamoe_experts(out, err);
  }
  return out->status == 0 ? 0 : -1;
}
