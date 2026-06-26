#include "nerva_cuda_api.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr uint32_t kHidden = 2;
constexpr uint32_t kIntermediate = 2;
constexpr float kRmsEps = 1.0e-5f;
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x00000100000001b3ull;
constexpr uint16_t kF16Zero = 0x0000u;
constexpr uint16_t kF16Half = 0x3800u;
constexpr uint16_t kF16One = 0x3c00u;
constexpr uint16_t kF16Two = 0x4000u;
constexpr uint32_t kInputOffset = 0;
constexpr uint32_t kOutputOffset = 2;
constexpr uint32_t kRmsAttnOffset = 4;
constexpr uint32_t kRmsMlpOffset = 6;
constexpr uint32_t kWqOffset = 8;
constexpr uint32_t kWkOffset = 12;
constexpr uint32_t kWvOffset = 16;
constexpr uint32_t kWoOffset = 20;
constexpr uint32_t kWGateOffset = 24;
constexpr uint32_t kWUpOffset = 28;
constexpr uint32_t kWDownOffset = 32;
constexpr uint32_t kArenaElements = 36;
constexpr uint32_t kResidentWeightElements = 32;

__device__ float silu(float value) {
  return value / (1.0f + expf(-value));
}

__device__ uint16_t f32_to_f16_bits(float value) {
  return __half_as_ushort(__float2half_rn(value));
}

__device__ float f16_bits_to_f32(uint16_t value) {
  return __half2float(__ushort_as_half(value));
}

__device__ void rms_norm2(const float *input, const uint16_t *weight, float *output) {
  const float mean_square = (input[0] * input[0] + input[1] * input[1]) /
                            static_cast<float>(kHidden);
  const float scale = rsqrtf(mean_square + kRmsEps);
  output[0] = input[0] * scale * f16_bits_to_f32(weight[0]);
  output[1] = input[1] * scale * f16_bits_to_f32(weight[1]);
}

__device__ void mat_vec2(const uint16_t *matrix, const float *input, float *output) {
  output[0] = f16_bits_to_f32(matrix[0]) * input[0] +
              f16_bits_to_f32(matrix[1]) * input[1];
  output[1] = f16_bits_to_f32(matrix[2]) * input[0] +
              f16_bits_to_f32(matrix[3]) * input[1];
}

__global__ void tiny_block_kernel(uint16_t *output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  const float input[kHidden] = {1.0f, 2.0f};

  float attn_norm[kHidden]{};
  const float input_mean_square =
      (input[0] * input[0] + input[1] * input[1]) / static_cast<float>(kHidden);
  const float attn_scale = rsqrtf(input_mean_square + kRmsEps);
  attn_norm[0] = input[0] * attn_scale;
  attn_norm[1] = input[1] * attn_scale;

  float residual[kHidden]{};
  residual[0] = input[0] + attn_norm[0];
  residual[1] = input[1] + attn_norm[1];

  const float residual_mean_square =
      (residual[0] * residual[0] + residual[1] * residual[1]) /
      static_cast<float>(kHidden);
  const float mlp_scale = rsqrtf(residual_mean_square + kRmsEps);
  const float mlp_norm[kHidden] = {residual[0] * mlp_scale, residual[1] * mlp_scale};

  float ff[kIntermediate]{};
  ff[0] = silu(0.5f * mlp_norm[0]) * mlp_norm[0];
  ff[1] = silu(0.5f * mlp_norm[1]) * mlp_norm[1];

  output[0] = f32_to_f16_bits(residual[0] + ff[0]);
  output[1] = f32_to_f16_bits(residual[1] + ff[1]);
}

__global__ void loaded_tiny_block_kernel(uint16_t *arena) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  const uint16_t *encoded_input = arena + kInputOffset;
  uint16_t *encoded_output = arena + kOutputOffset;
  const uint16_t *rms_attn_weight = arena + kRmsAttnOffset;
  const uint16_t *rms_mlp_weight = arena + kRmsMlpOffset;
  const uint16_t *w_q = arena + kWqOffset;
  const uint16_t *w_k = arena + kWkOffset;
  const uint16_t *w_v = arena + kWvOffset;
  const uint16_t *w_o = arena + kWoOffset;
  const uint16_t *w_gate = arena + kWGateOffset;
  const uint16_t *w_up = arena + kWUpOffset;
  const uint16_t *w_down = arena + kWDownOffset;

  const float input[kHidden] = {f16_bits_to_f32(encoded_input[0]),
                                f16_bits_to_f32(encoded_input[1])};

  float attn_norm[kHidden]{};
  rms_norm2(input, rms_attn_weight, attn_norm);

  float q[kHidden]{};
  float k[kHidden]{};
  float v[kHidden]{};
  mat_vec2(w_q, attn_norm, q);
  mat_vec2(w_k, attn_norm, k);
  mat_vec2(w_v, attn_norm, v);
  (void)q;
  (void)k;

  float attn[kHidden] = {v[0], v[1]};
  float residual[kHidden]{};
  mat_vec2(w_o, attn, residual);
  residual[0] += input[0];
  residual[1] += input[1];

  float mlp_norm[kHidden]{};
  rms_norm2(residual, rms_mlp_weight, mlp_norm);

  float gate[kIntermediate]{};
  float up[kIntermediate]{};
  mat_vec2(w_gate, mlp_norm, gate);
  mat_vec2(w_up, mlp_norm, up);

  float ff[kIntermediate]{};
  ff[0] = silu(gate[0]) * up[0];
  ff[1] = silu(gate[1]) * up[1];

  float down[kHidden]{};
  mat_vec2(w_down, ff, down);
  encoded_output[0] = f32_to_f16_bits(residual[0] + down[0]);
  encoded_output[1] = f32_to_f16_bits(residual[1] + down[1]);
}

uint64_t hash_u16s(const uint16_t *values, size_t len) {
  uint64_t hash = kFnvOffset;
  for (size_t index = 0; index < len; ++index) {
    const uint16_t value = values[index];
    const uint8_t low = static_cast<uint8_t>(value & 0xffu);
    const uint8_t high = static_cast<uint8_t>((value >> 8) & 0xffu);
    hash ^= static_cast<uint64_t>(low);
    hash *= kFnvPrime;
    hash ^= static_cast<uint64_t>(high);
    hash *= kFnvPrime;
  }
  return hash;
}

void clear_result(NervaCudaTinyBlockResult *out) {
  out->status = -1;
  out->cuda_error = 0;
  out->device_count = 0;
  out->hidden = kHidden;
  out->intermediate = kIntermediate;
  out->output[0] = 0;
  out->output[1] = 0;
  out->output_hash = 0;
  out->device_arena_bytes = sizeof(uint16_t) * kHidden;
  out->pinned_host_bytes = sizeof(uint16_t) * kHidden;
  out->kernel_launches = 0;
  out->sync_calls = 0;
  out->d2h_bytes = 0;
  out->hot_path_allocations = 0;
}

int fail(NervaCudaTinyBlockResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

void clear_loaded_result(NervaCudaLoadedTinyBlockResult *out) {
  out->status = -1;
  out->cuda_error = 0;
  out->device_count = 0;
  out->hidden = kHidden;
  out->intermediate = kIntermediate;
  out->output[0] = 0;
  out->output[1] = 0;
  out->output_hash = 0;
  out->resident_weight_bytes = sizeof(uint16_t) * kResidentWeightElements;
  out->device_arena_bytes = sizeof(uint16_t) * kArenaElements;
  out->pinned_host_bytes = sizeof(uint16_t) * kArenaElements;
  out->h2d_bytes = 0;
  out->d2h_bytes = 0;
  out->kernel_launches = 0;
  out->sync_calls = 0;
  out->hot_path_allocations = 0;
}

int fail(NervaCudaLoadedTinyBlockResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

void fill_identity_matrix(uint16_t *matrix, uint16_t diagonal) {
  matrix[0] = diagonal;
  matrix[1] = kF16Zero;
  matrix[2] = kF16Zero;
  matrix[3] = diagonal;
}

void fill_loaded_host_arena(uint16_t *arena) {
  for (uint32_t index = 0; index < kArenaElements; ++index) {
    arena[index] = kF16Zero;
  }
  arena[kInputOffset] = kF16One;
  arena[kInputOffset + 1] = kF16Two;
  arena[kRmsAttnOffset] = kF16One;
  arena[kRmsAttnOffset + 1] = kF16One;
  arena[kRmsMlpOffset] = kF16One;
  arena[kRmsMlpOffset + 1] = kF16One;
  fill_identity_matrix(arena + kWqOffset, kF16One);
  fill_identity_matrix(arena + kWkOffset, kF16One);
  fill_identity_matrix(arena + kWvOffset, kF16One);
  fill_identity_matrix(arena + kWoOffset, kF16One);
  fill_identity_matrix(arena + kWGateOffset, kF16Half);
  fill_identity_matrix(arena + kWUpOffset, kF16One);
  fill_identity_matrix(arena + kWDownOffset, kF16One);
}

}  // namespace

extern "C" int nerva_cuda_tiny_block_smoke(NervaCudaTinyBlockResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);

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

  uint16_t *device_output = nullptr;
  err = cudaMalloc(reinterpret_cast<void **>(&device_output), sizeof(uint16_t) * kHidden);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  uint16_t *host_output = nullptr;
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_output), sizeof(uint16_t) * kHidden,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFree(device_output);
    return fail(out, err);
  }
  host_output[0] = 0;
  host_output[1] = 0;

  cudaStream_t stream = nullptr;
  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    cudaFreeHost(host_output);
    cudaFree(device_output);
    return fail(out, err);
  }

  tiny_block_kernel<<<1, 1, 0, stream>>>(device_output);
  out->kernel_launches = 1;
  err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_output, device_output, sizeof(uint16_t) * kHidden,
                          cudaMemcpyDeviceToHost, stream);
    out->d2h_bytes = sizeof(uint16_t) * kHidden;
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
    out->sync_calls = 1;
  }

  if (err == cudaSuccess) {
    out->output[0] = host_output[0];
    out->output[1] = host_output[1];
    out->output_hash = hash_u16s(host_output, kHidden);
    out->status = out->output_hash != 0 ? 0 : -1;
  } else {
    out->cuda_error = static_cast<int32_t>(err);
    out->status = -1;
  }

  cudaStreamDestroy(stream);
  cudaFreeHost(host_output);
  cudaFree(device_output);
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_loaded_tiny_block_smoke(NervaCudaLoadedTinyBlockResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_loaded_result(out);

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

  uint16_t *device_arena = nullptr;
  err = cudaMalloc(reinterpret_cast<void **>(&device_arena), sizeof(uint16_t) * kArenaElements);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  uint16_t *host_arena = nullptr;
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_arena), sizeof(uint16_t) * kArenaElements,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFree(device_arena);
    return fail(out, err);
  }
  fill_loaded_host_arena(host_arena);

  cudaStream_t stream = nullptr;
  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    cudaFreeHost(host_arena);
    cudaFree(device_arena);
    return fail(out, err);
  }

  err = cudaMemcpyAsync(device_arena, host_arena, sizeof(uint16_t) * kArenaElements,
                        cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess) {
    out->h2d_bytes = sizeof(uint16_t) * kArenaElements;
    err = cudaStreamSynchronize(stream);
    out->sync_calls += 1;
  }

  if (err == cudaSuccess) {
    loaded_tiny_block_kernel<<<1, 1, 0, stream>>>(device_arena);
    out->kernel_launches = 1;
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_arena + kOutputOffset, device_arena + kOutputOffset,
                          sizeof(uint16_t) * kHidden, cudaMemcpyDeviceToHost, stream);
    out->d2h_bytes = sizeof(uint16_t) * kHidden;
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
    out->sync_calls += 1;
  }

  if (err == cudaSuccess) {
    out->output[0] = host_arena[kOutputOffset];
    out->output[1] = host_arena[kOutputOffset + 1];
    out->output_hash = hash_u16s(host_arena + kOutputOffset, kHidden);
    out->status = out->output_hash != 0 ? 0 : -1;
  } else {
    out->cuda_error = static_cast<int32_t>(err);
    out->status = -1;
  }

  cudaStreamDestroy(stream);
  cudaFreeHost(host_arena);
  cudaFree(device_arena);
  return out->status == 0 ? 0 : -1;
}
