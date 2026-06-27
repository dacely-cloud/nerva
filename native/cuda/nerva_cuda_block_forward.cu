#include "nerva_cuda_api.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kDTypeF16 = 0;
constexpr uint32_t kDTypeBF16 = 1;
constexpr uint64_t kMissingOffset = UINT64_MAX;
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x00000100000001b3ull;

struct BlockLayout {
  uint64_t input;
  uint64_t output;
  uint64_t rms_attn;
  uint64_t rms_mlp;
  uint64_t w_q;
  uint64_t w_k;
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

__device__ void mat_vec(const uint16_t *matrix, const float *input, uint32_t rows,
                        uint32_t cols, uint32_t dtype, float *output) {
  for (uint32_t row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < cols; ++col) {
      sum += encoded_to_f32(matrix[row * cols + col], dtype) * input[col];
    }
    output[row] = sum;
  }
}

__device__ void rms_norm(const float *input, const uint16_t *weight, uint32_t hidden,
                         uint32_t dtype, float eps, float *output) {
  float mean_square = 0.0f;
  for (uint32_t index = 0; index < hidden; ++index) {
    mean_square += input[index] * input[index];
  }
  const float scale = rsqrtf(mean_square / static_cast<float>(hidden) + eps);
  for (uint32_t index = 0; index < hidden; ++index) {
    output[index] = input[index] * scale * encoded_to_f32(weight[index], dtype);
  }
}

__device__ void add_bias(const uint16_t *arena, uint64_t offset, uint32_t len,
                         uint32_t dtype, float *output) {
  if (offset == kMissingOffset) {
    return;
  }
  const uint16_t *bias = arena + offset;
  for (uint32_t index = 0; index < len; ++index) {
    output[index] += encoded_to_f32(bias[index], dtype);
  }
}

__device__ void apply_rope(float *values, uint32_t heads, uint32_t head_dim,
                           uint32_t position, float theta) {
  if (theta <= 0.0f) {
    return;
  }
  const uint32_t half = head_dim / 2;
  for (uint32_t head = 0; head < heads; ++head) {
    const uint32_t start = head * head_dim;
    for (uint32_t offset = 0; offset < half; ++offset) {
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
  }
}

__global__ void block_forward_kernel(uint16_t *arena, BlockLayout layout, uint32_t dtype,
                                     uint32_t hidden, uint32_t heads, uint32_t kv_heads,
                                     uint32_t head_dim, uint32_t intermediate,
                                     uint32_t position, float rms_eps, float rope_theta,
                                     float *scratch) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
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
  float *gate = mlp_norm + hidden;
  float *up = gate + intermediate;
  float *ff = up + intermediate;
  float *down = ff + intermediate;

  for (uint32_t index = 0; index < hidden; ++index) {
    input[index] = encoded_to_f32(arena[layout.input + index], dtype);
  }
  rms_norm(input, arena + layout.rms_attn, hidden, dtype, rms_eps, attn_norm);
  mat_vec(arena + layout.w_q, attn_norm, attention_hidden, hidden, dtype, q);
  mat_vec(arena + layout.w_k, attn_norm, kv_hidden, hidden, dtype, k);
  mat_vec(arena + layout.w_v, attn_norm, kv_hidden, hidden, dtype, v);
  add_bias(arena, layout.q_bias, attention_hidden, dtype, q);
  add_bias(arena, layout.k_bias, kv_hidden, dtype, k);
  add_bias(arena, layout.v_bias, kv_hidden, dtype, v);
  apply_rope(q, heads, head_dim, position, rope_theta);
  apply_rope(k, kv_heads, head_dim, position, rope_theta);

  for (uint32_t head = 0; head < heads; ++head) {
    const uint32_t kv_head = head / (heads / kv_heads);
    for (uint32_t offset = 0; offset < head_dim; ++offset) {
      attn[head * head_dim + offset] = v[kv_head * head_dim + offset];
    }
  }
  mat_vec(arena + layout.w_o, attn, hidden, attention_hidden, dtype, residual);
  add_bias(arena, layout.o_bias, hidden, dtype, residual);
  for (uint32_t index = 0; index < hidden; ++index) {
    residual[index] += input[index];
  }

  rms_norm(residual, arena + layout.rms_mlp, hidden, dtype, rms_eps, mlp_norm);
  mat_vec(arena + layout.w_gate, mlp_norm, intermediate, hidden, dtype, gate);
  mat_vec(arena + layout.w_up, mlp_norm, intermediate, hidden, dtype, up);
  for (uint32_t index = 0; index < intermediate; ++index) {
    ff[index] = silu(gate[index]) * up[index];
  }
  mat_vec(arena + layout.w_down, ff, hidden, intermediate, dtype, down);
  for (uint32_t index = 0; index < hidden; ++index) {
    arena[layout.output + index] = f32_to_encoded(residual[index] + down[index], dtype);
  }
}

uint64_t hash_u16s(const uint16_t *values, uint64_t len) {
  uint64_t hash = kFnvOffset;
  for (uint64_t index = 0; index < len; ++index) {
    hash ^= static_cast<uint64_t>(values[index] & 0xffu);
    hash *= kFnvPrime;
    hash ^= static_cast<uint64_t>((values[index] >> 8) & 0xffu);
    hash *= kFnvPrime;
  }
  return hash;
}

uint64_t push(BlockLayout &layout, uint64_t BlockLayout::*field, uint64_t &cursor,
              uint64_t len) {
  layout.*field = cursor;
  cursor += len;
  return layout.*field;
}

uint64_t push_optional(BlockLayout &layout, uint64_t BlockLayout::*field, uint64_t &cursor,
                       uint64_t len, const uint16_t *ptr) {
  if (ptr == nullptr) {
    layout.*field = kMissingOffset;
    return kMissingOffset;
  }
  return push(layout, field, cursor, len);
}

bool valid_request(const NervaCudaBlockForwardRequest *request) {
  return request != nullptr && request->output != nullptr && request->input != nullptr &&
         request->rms_attn_weight != nullptr && request->rms_mlp_weight != nullptr &&
         request->w_q != nullptr && request->w_k != nullptr && request->w_v != nullptr &&
         request->w_o != nullptr && request->w_gate != nullptr && request->w_up != nullptr &&
         request->w_down != nullptr && request->hidden > 0 && request->heads > 0 &&
         request->kv_heads > 0 && request->head_dim > 0 && request->intermediate > 0 &&
         request->kv_heads <= request->heads && request->heads % request->kv_heads == 0 &&
         request->dtype <= kDTypeBF16 &&
         (request->rope_theta <= 0.0f || request->head_dim % 2 == 0);
}

void clear_result(const NervaCudaBlockForwardRequest *request, NervaCudaBlockForwardResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request != nullptr) {
    out->dtype = request->dtype;
    out->hidden = request->hidden;
    out->heads = request->heads;
    out->kv_heads = request->kv_heads;
    out->head_dim = request->head_dim;
    out->intermediate = request->intermediate;
  }
}

}  // namespace

extern "C" int nerva_cuda_block_forward_u16(const NervaCudaBlockForwardRequest *request,
                                            NervaCudaBlockForwardResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(request, out);
  if (!valid_request(request)) {
    return -1;
  }
  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }
  if (out->device_count <= 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorNoDevice);
    return -1;
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  const uint64_t hidden = request->hidden;
  const uint64_t attention_hidden = request->heads * request->head_dim;
  const uint64_t kv_hidden = request->kv_heads * request->head_dim;
  const uint64_t intermediate = request->intermediate;
  BlockLayout layout{};
  uint64_t elements = 0;
  push(layout, &BlockLayout::input, elements, hidden);
  push(layout, &BlockLayout::output, elements, hidden);
  push(layout, &BlockLayout::rms_attn, elements, hidden);
  push(layout, &BlockLayout::rms_mlp, elements, hidden);
  push(layout, &BlockLayout::w_q, elements, attention_hidden * hidden);
  push(layout, &BlockLayout::w_k, elements, kv_hidden * hidden);
  push(layout, &BlockLayout::w_v, elements, kv_hidden * hidden);
  push(layout, &BlockLayout::w_o, elements, hidden * attention_hidden);
  push_optional(layout, &BlockLayout::q_bias, elements, attention_hidden, request->q_bias);
  push_optional(layout, &BlockLayout::k_bias, elements, kv_hidden, request->k_bias);
  push_optional(layout, &BlockLayout::v_bias, elements, kv_hidden, request->v_bias);
  push_optional(layout, &BlockLayout::o_bias, elements, hidden, request->o_bias);
  push(layout, &BlockLayout::w_gate, elements, intermediate * hidden);
  push(layout, &BlockLayout::w_up, elements, intermediate * hidden);
  push(layout, &BlockLayout::w_down, elements, hidden * intermediate);
  const uint64_t arena_bytes = elements * sizeof(uint16_t);
  const uint64_t scratch_elements =
      hidden * 4 + attention_hidden * 2 + kv_hidden * 2 + intermediate * 4;
  const uint64_t scratch_bytes = scratch_elements * sizeof(float);

  uint16_t *host_arena = nullptr;
  uint16_t *device_arena = nullptr;
  float *device_scratch = nullptr;
  cudaStream_t stream = nullptr;
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_arena), arena_bytes, cudaHostAllocDefault);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_arena), arena_bytes);
  if (err == cudaSuccess)
    err = cudaMalloc(reinterpret_cast<void **>(&device_scratch), scratch_bytes);
  if (err == cudaSuccess) err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    cudaFree(device_scratch);
    cudaFree(device_arena);
    cudaFreeHost(host_arena);
    return -1;
  }

  memset(host_arena, 0, arena_bytes);
  memcpy(host_arena + layout.input, request->input, hidden * sizeof(uint16_t));
  memcpy(host_arena + layout.rms_attn, request->rms_attn_weight, hidden * sizeof(uint16_t));
  memcpy(host_arena + layout.rms_mlp, request->rms_mlp_weight, hidden * sizeof(uint16_t));
  memcpy(host_arena + layout.w_q, request->w_q, attention_hidden * hidden * sizeof(uint16_t));
  memcpy(host_arena + layout.w_k, request->w_k, kv_hidden * hidden * sizeof(uint16_t));
  memcpy(host_arena + layout.w_v, request->w_v, kv_hidden * hidden * sizeof(uint16_t));
  memcpy(host_arena + layout.w_o, request->w_o, hidden * attention_hidden * sizeof(uint16_t));
  if (request->q_bias) memcpy(host_arena + layout.q_bias, request->q_bias, attention_hidden * 2);
  if (request->k_bias) memcpy(host_arena + layout.k_bias, request->k_bias, kv_hidden * 2);
  if (request->v_bias) memcpy(host_arena + layout.v_bias, request->v_bias, kv_hidden * 2);
  if (request->o_bias) memcpy(host_arena + layout.o_bias, request->o_bias, hidden * 2);
  memcpy(host_arena + layout.w_gate, request->w_gate, intermediate * hidden * sizeof(uint16_t));
  memcpy(host_arena + layout.w_up, request->w_up, intermediate * hidden * sizeof(uint16_t));
  memcpy(host_arena + layout.w_down, request->w_down, hidden * intermediate * sizeof(uint16_t));

  err = cudaMemcpyAsync(device_arena, host_arena, arena_bytes, cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess) {
    out->h2d_bytes = arena_bytes;
    block_forward_kernel<<<1, 1, 0, stream>>>(
        device_arena, layout, request->dtype, request->hidden, request->heads, request->kv_heads,
        request->head_dim, request->intermediate, request->position, request->rms_eps,
        request->rope_theta, device_scratch);
    out->kernel_launches = 1;
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_arena + layout.output, device_arena + layout.output,
                          hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream);
    out->d2h_bytes = hidden * sizeof(uint16_t);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
    out->sync_calls = 1;
  }

  if (err == cudaSuccess) {
    memcpy(request->output, host_arena + layout.output, hidden * sizeof(uint16_t));
    out->output_hash = hash_u16s(request->output, hidden);
    out->resident_weight_bytes = arena_bytes - (hidden * 2 * sizeof(uint16_t));
    out->device_arena_bytes = arena_bytes + scratch_bytes;
    out->pinned_host_bytes = arena_bytes;
    out->status = out->output_hash != 0 ? 0 : -1;
  } else {
    out->cuda_error = static_cast<int32_t>(err);
  }

  cudaStreamDestroy(stream);
  cudaFree(device_scratch);
  cudaFree(device_arena);
  cudaFreeHost(host_arena);
  return out->status == 0 ? 0 : -1;
}
