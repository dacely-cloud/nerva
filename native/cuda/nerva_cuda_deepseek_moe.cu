#include "nerva_cuda_api.h"

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
