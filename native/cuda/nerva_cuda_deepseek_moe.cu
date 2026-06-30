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

int fail_forward(NervaCudaDeepSeekMoeForwardResult *out, cudaError_t err) {
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
