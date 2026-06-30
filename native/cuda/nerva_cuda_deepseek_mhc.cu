#include "nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kMaxHcMult = 64;
constexpr uint64_t kMaxHcHidden = 65536;

__host__ __device__ float sigmoidf_stable(float value) {
  if (value >= 0.0f) {
    const float z = expf(-value);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(value);
  return z / (1.0f + z);
}

__global__ void deepseek_mhc_head_kernel(const float *hidden_states,
                                         const float *fn_weights,
                                         const float *hc_base,
                                         float *output,
                                         uint32_t tokens,
                                         uint32_t hc_mult,
                                         uint32_t hidden_size,
                                         float rms_eps,
                                         float hc_eps,
                                         float hc_scale,
                                         int32_t *mhc_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (tokens == 0 || hc_mult == 0 || hidden_size == 0 ||
      hc_mult > kMaxHcMult) {
    *mhc_error = -1;
    return;
  }
  const uint64_t hc_hidden_size =
      static_cast<uint64_t>(hc_mult) * hidden_size;
  if (hc_hidden_size > kMaxHcHidden) {
    *mhc_error = -2;
    return;
  }

  float gates[kMaxHcMult];
  for (uint32_t token = 0; token < tokens; ++token) {
    const uint64_t token_offset = static_cast<uint64_t>(token) * hc_hidden_size;
    const float *token_values = hidden_states + token_offset;
    float sqrsum = 0.0f;
    for (uint64_t index = 0; index < hc_hidden_size; ++index) {
      const float value = token_values[index];
      sqrsum += value * value;
    }
    const float rms_scale =
        rsqrtf(sqrsum / static_cast<float>(hc_hidden_size) + rms_eps);

    for (uint32_t channel = 0; channel < hc_mult; ++channel) {
      const float *row =
          fn_weights + static_cast<uint64_t>(channel) * hc_hidden_size;
      float mix = 0.0f;
      for (uint64_t index = 0; index < hc_hidden_size; ++index) {
        mix += row[index] * token_values[index];
      }
      mix *= rms_scale;
      gates[channel] =
          sigmoidf_stable(mix * hc_scale + hc_base[channel]) + hc_eps;
    }

    for (uint32_t hidden = 0; hidden < hidden_size; ++hidden) {
      float value = 0.0f;
      for (uint32_t channel = 0; channel < hc_mult; ++channel) {
        value += gates[channel] *
                 token_values[static_cast<uint64_t>(channel) * hidden_size +
                              hidden];
      }
      output[static_cast<uint64_t>(token) * hidden_size + hidden] = value;
    }
  }
  *mhc_error = 0;
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

uint64_t hash_f32_bits(const float *values, uint64_t len) {
  uint64_t hash = 1469598103934665603ull;
  for (uint64_t i = 0; i < len; ++i) {
    hash = mix_hash_u32(hash, f32_bits(values[i]));
  }
  return hash;
}

bool checked_mul_u64(uint64_t lhs, uint64_t rhs, uint64_t *out) {
  if (out == nullptr || (lhs != 0 && rhs > UINT64_MAX / lhs)) {
    return false;
  }
  *out = lhs * rhs;
  return true;
}

void clear_result(NervaCudaDeepSeekMhcHeadResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

int fail(NervaCudaDeepSeekMhcHeadResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

}  // namespace

extern "C" int nerva_cuda_deepseek_mhc_head(
    const NervaCudaDeepSeekMhcHeadRequest *request,
    NervaCudaDeepSeekMhcHeadResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  if (request == nullptr || request->tokens == 0 || request->hc_mult == 0 ||
      request->hidden_size == 0 || !isfinite(request->rms_eps) ||
      !isfinite(request->hc_eps) || !isfinite(request->hc_scale) ||
      request->hidden_states == nullptr || request->fn_weights == nullptr ||
      request->hc_base == nullptr || request->output == nullptr) {
    return -1;
  }
  out->tokens = request->tokens;
  out->hc_mult = request->hc_mult;
  out->hidden_size = request->hidden_size;
  out->rms_eps = request->rms_eps;
  out->hc_eps = request->hc_eps;
  out->hc_scale = request->hc_scale;

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

  uint64_t hc_hidden_values = 0;
  uint64_t hidden_values = 0;
  uint64_t input_values = 0;
  uint64_t fn_values = 0;
  if (!checked_mul_u64(request->hc_mult, request->hidden_size,
                       &hc_hidden_values) ||
      !checked_mul_u64(request->tokens, request->hidden_size,
                       &hidden_values) ||
      !checked_mul_u64(request->tokens, hc_hidden_values, &input_values) ||
      !checked_mul_u64(request->hc_mult, hc_hidden_values, &fn_values)) {
    return -1;
  }

  const uint64_t input_bytes = input_values * sizeof(float);
  const uint64_t fn_bytes = fn_values * sizeof(float);
  const uint64_t base_bytes = static_cast<uint64_t>(request->hc_mult) * sizeof(float);
  const uint64_t output_bytes = hidden_values * sizeof(float);
  const uint64_t error_bytes = sizeof(int32_t);

  float *device_hidden_states = nullptr;
  float *device_fn_weights = nullptr;
  float *device_hc_base = nullptr;
  float *device_output = nullptr;
  int32_t *device_mhc_error = nullptr;

  err = cudaMalloc(reinterpret_cast<void **>(&device_hidden_states),
                   input_bytes);
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&device_fn_weights), fn_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&device_hc_base), base_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&device_output), output_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&device_mhc_error),
                     error_bytes);
  }
  if (err != cudaSuccess) {
    cudaFree(device_hidden_states);
    cudaFree(device_fn_weights);
    cudaFree(device_hc_base);
    cudaFree(device_output);
    cudaFree(device_mhc_error);
    return fail(out, err);
  }
  out->device_arena_bytes =
      input_bytes + fn_bytes + base_bytes + output_bytes + error_bytes;

  err = cudaMemcpy(device_hidden_states, request->hidden_states, input_bytes,
                   cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    err = cudaMemcpy(device_fn_weights, request->fn_weights, fn_bytes,
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(device_hc_base, request->hc_base, base_bytes,
                     cudaMemcpyHostToDevice);
  }
  int32_t host_mhc_error = -99;
  if (err == cudaSuccess) {
    err = cudaMemcpy(device_mhc_error, &host_mhc_error, error_bytes,
                     cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess) {
    cudaFree(device_hidden_states);
    cudaFree(device_fn_weights);
    cudaFree(device_hc_base);
    cudaFree(device_output);
    cudaFree(device_mhc_error);
    return fail(out, err);
  }
  out->h2d_bytes = input_bytes + fn_bytes + base_bytes + error_bytes;

  deepseek_mhc_head_kernel<<<1, 1>>>(
      device_hidden_states, device_fn_weights, device_hc_base, device_output,
      request->tokens, request->hc_mult, request->hidden_size, request->rms_eps,
      request->hc_eps, request->hc_scale, device_mhc_error);
  out->kernel_launches = 1;
  err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaDeviceSynchronize();
    out->sync_calls = 1;
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(&host_mhc_error, device_mhc_error, error_bytes,
                     cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess && host_mhc_error == 0) {
    err = cudaMemcpy(request->output, device_output, output_bytes,
                     cudaMemcpyDeviceToHost);
    out->d2h_bytes = output_bytes + error_bytes;
  } else if (err == cudaSuccess) {
    out->d2h_bytes = error_bytes;
  }
  out->mhc_error = host_mhc_error;
  if (err == cudaSuccess && host_mhc_error == 0) {
    out->output_hash = hash_f32_bits(request->output, hidden_values);
  }

  cudaFree(device_hidden_states);
  cudaFree(device_fn_weights);
  cudaFree(device_hc_base);
  cudaFree(device_output);
  cudaFree(device_mhc_error);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  out->status = host_mhc_error == 0 ? 0 : -1;
  return out->status == 0 ? 0 : -1;
}
