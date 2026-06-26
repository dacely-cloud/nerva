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

__device__ float silu(float value) {
  return value / (1.0f + expf(-value));
}

__device__ uint16_t f32_to_f16_bits(float value) {
  return __half_as_ushort(__float2half_rn(value));
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
