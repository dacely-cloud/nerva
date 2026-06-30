#include "nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kMaxMhcPrePostHcMult = 8;
constexpr uint64_t kMaxHcHidden = 65536;
constexpr uint64_t kMaxMhcPrePostMixes =
    static_cast<uint64_t>(kMaxMhcPrePostHcMult) *
    (2u + kMaxMhcPrePostHcMult);

__host__ __device__ float sigmoidf_stable(float value) {
  if (value >= 0.0f) {
    const float z = expf(-value);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(value);
  return z / (1.0f + z);
}

__device__ void mhc_post_token(const float *x, const float *residual,
                               const float *post_layer_mix,
                               const float *comb_res_mix, float *output,
                               uint32_t token, uint32_t hc_mult,
                               uint32_t hidden_size) {
  const uint64_t hc_hidden_size =
      static_cast<uint64_t>(hc_mult) * hidden_size;
  const uint64_t residual_offset =
      static_cast<uint64_t>(token) * hc_hidden_size;
  const uint64_t comb_offset =
      static_cast<uint64_t>(token) * hc_mult * hc_mult;
  for (uint32_t out_channel = 0; out_channel < hc_mult; ++out_channel) {
    for (uint32_t hidden = 0; hidden < hidden_size; ++hidden) {
      float mixed = 0.0f;
      for (uint32_t in_channel = 0; in_channel < hc_mult; ++in_channel) {
        mixed += comb_res_mix[comb_offset +
                              static_cast<uint64_t>(in_channel) * hc_mult +
                              out_channel] *
                 residual[residual_offset +
                          static_cast<uint64_t>(in_channel) * hidden_size +
                          hidden];
      }
      mixed += post_layer_mix[static_cast<uint64_t>(token) * hc_mult +
                              out_channel] *
               x[static_cast<uint64_t>(token) * hidden_size + hidden];
      output[residual_offset +
             static_cast<uint64_t>(out_channel) * hidden_size + hidden] =
          mixed;
    }
  }
}

__device__ void mhc_pre_token(const float *residual, const float *fn_weights,
                              const float *hc_scale, const float *hc_base,
                              float *post_mix, float *comb_mix,
                              float *layer_input, uint32_t token,
                              uint32_t hc_mult, uint32_t hidden_size,
                              uint32_t sinkhorn_repeat, float rms_eps,
                              float hc_pre_eps, float hc_sinkhorn_eps,
                              float hc_post_mult_value) {
  const uint64_t hc_hidden_size =
      static_cast<uint64_t>(hc_mult) * hidden_size;
  const uint64_t hc_mult2 = static_cast<uint64_t>(hc_mult) * hc_mult;
  const uint64_t hc_mult3 = static_cast<uint64_t>(hc_mult) * (2u + hc_mult);
  const uint64_t token_offset = static_cast<uint64_t>(token) * hc_hidden_size;
  const float *residual_token = residual + token_offset;

  float sqrsum = 0.0f;
  for (uint64_t index = 0; index < hc_hidden_size; ++index) {
    const float value = residual_token[index];
    sqrsum += value * value;
  }
  const float rms_scale =
      rsqrtf(sqrsum / static_cast<float>(hc_hidden_size) + rms_eps);

  float mixes[kMaxMhcPrePostMixes];
  float pre_mix[kMaxMhcPrePostHcMult];
  float comb[kMaxMhcPrePostHcMult * kMaxMhcPrePostHcMult];
  for (uint64_t mix = 0; mix < hc_mult3; ++mix) {
    const float *row = fn_weights + mix * hc_hidden_size;
    float value = 0.0f;
    for (uint64_t index = 0; index < hc_hidden_size; ++index) {
      value += row[index] * residual_token[index];
    }
    mixes[mix] = value * rms_scale;
  }

  for (uint32_t channel = 0; channel < hc_mult; ++channel) {
    pre_mix[channel] =
        sigmoidf_stable(mixes[channel] * hc_scale[0] + hc_base[channel]) +
        hc_pre_eps;
    post_mix[static_cast<uint64_t>(token) * hc_mult + channel] =
        sigmoidf_stable(mixes[hc_mult + channel] * hc_scale[1] +
                        hc_base[hc_mult + channel]) *
        hc_post_mult_value;
  }

  for (uint32_t row = 0; row < hc_mult; ++row) {
    const uint64_t logits_start =
        static_cast<uint64_t>(2u * hc_mult) +
        static_cast<uint64_t>(row) * hc_mult;
    for (uint32_t col = 0; col < hc_mult; ++col) {
      comb[static_cast<uint64_t>(row) * hc_mult + col] =
          mixes[logits_start + col] * hc_scale[2] +
          hc_base[logits_start + col];
    }
    float row_max = comb[static_cast<uint64_t>(row) * hc_mult];
    for (uint32_t col = 1; col < hc_mult; ++col) {
      row_max = fmaxf(row_max,
                      comb[static_cast<uint64_t>(row) * hc_mult + col]);
    }
    float row_sum = 0.0f;
    for (uint32_t col = 0; col < hc_mult; ++col) {
      const float value =
          expf(comb[static_cast<uint64_t>(row) * hc_mult + col] - row_max);
      comb[static_cast<uint64_t>(row) * hc_mult + col] = value;
      row_sum += value;
    }
    for (uint32_t col = 0; col < hc_mult; ++col) {
      comb[static_cast<uint64_t>(row) * hc_mult + col] =
          comb[static_cast<uint64_t>(row) * hc_mult + col] / row_sum +
          hc_sinkhorn_eps;
    }
  }

  for (uint32_t col = 0; col < hc_mult; ++col) {
    float col_sum = 0.0f;
    for (uint32_t row = 0; row < hc_mult; ++row) {
      col_sum += comb[static_cast<uint64_t>(row) * hc_mult + col];
    }
    for (uint32_t row = 0; row < hc_mult; ++row) {
      comb[static_cast<uint64_t>(row) * hc_mult + col] /=
          col_sum + hc_sinkhorn_eps;
    }
  }
  for (uint32_t repeat = 1; repeat < sinkhorn_repeat; ++repeat) {
    for (uint32_t row = 0; row < hc_mult; ++row) {
      float row_sum = 0.0f;
      for (uint32_t col = 0; col < hc_mult; ++col) {
        row_sum += comb[static_cast<uint64_t>(row) * hc_mult + col];
      }
      for (uint32_t col = 0; col < hc_mult; ++col) {
        comb[static_cast<uint64_t>(row) * hc_mult + col] /=
            row_sum + hc_sinkhorn_eps;
      }
    }
    for (uint32_t col = 0; col < hc_mult; ++col) {
      float col_sum = 0.0f;
      for (uint32_t row = 0; row < hc_mult; ++row) {
        col_sum += comb[static_cast<uint64_t>(row) * hc_mult + col];
      }
      for (uint32_t row = 0; row < hc_mult; ++row) {
        comb[static_cast<uint64_t>(row) * hc_mult + col] /=
            col_sum + hc_sinkhorn_eps;
      }
    }
  }

  const uint64_t comb_offset = static_cast<uint64_t>(token) * hc_mult2;
  for (uint64_t index = 0; index < hc_mult2; ++index) {
    comb_mix[comb_offset + index] = comb[index];
  }

  for (uint32_t hidden = 0; hidden < hidden_size; ++hidden) {
    float value = 0.0f;
    for (uint32_t channel = 0; channel < hc_mult; ++channel) {
      value += pre_mix[channel] *
               residual_token[static_cast<uint64_t>(channel) * hidden_size +
                              hidden];
    }
    layer_input[static_cast<uint64_t>(token) * hidden_size + hidden] = value;
  }
}

__global__ void deepseek_mhc_fused_post_pre_kernel(
    const float *x, const float *residual, const float *post_layer_mix,
    const float *comb_res_mix, const float *fn_weights, const float *hc_scale,
    const float *hc_base, float *new_residual, float *new_post_mix,
    float *new_comb_mix, float *layer_input, uint32_t tokens,
    uint32_t hc_mult, uint32_t hidden_size, uint32_t sinkhorn_repeat,
    float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
    float hc_post_mult_value, int32_t *mhc_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (tokens == 0 || hc_mult == 0 || hidden_size == 0 ||
      sinkhorn_repeat == 0 || hc_mult > kMaxMhcPrePostHcMult) {
    *mhc_error = -1;
    return;
  }
  const uint64_t hc_hidden_size =
      static_cast<uint64_t>(hc_mult) * hidden_size;
  const uint64_t hc_mult3 = static_cast<uint64_t>(hc_mult) * (2u + hc_mult);
  if (hc_hidden_size > kMaxHcHidden || hc_mult3 > kMaxMhcPrePostMixes) {
    *mhc_error = -2;
    return;
  }

  for (uint32_t token = 0; token < tokens; ++token) {
    mhc_post_token(x, residual, post_layer_mix, comb_res_mix, new_residual,
                   token, hc_mult, hidden_size);
    mhc_pre_token(new_residual, fn_weights, hc_scale, hc_base, new_post_mix,
                  new_comb_mix, layer_input, token, hc_mult, hidden_size,
                  sinkhorn_repeat, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                  hc_post_mult_value);
  }
  *mhc_error = 0;
}

uint32_t f32_bits(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint64_t mix_hash_u32(uint64_t hash, uint32_t value) {
  hash ^= static_cast<uint64_t>(value) + 0x9e3779b97f4a7c15ull +
          (hash << 6) + (hash >> 2);
  return hash;
}

uint64_t hash_f32_bits(const float *values, uint64_t count) {
  uint64_t hash = 0xcbf29ce484222325ull;
  for (uint64_t i = 0; i < count; ++i) {
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

void clear_fused_result(
    const NervaCudaDeepSeekMhcFusedPostPreRequest *request,
    NervaCudaDeepSeekMhcFusedPostPreResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request != nullptr) {
    out->tokens = request->tokens;
    out->hc_mult = request->hc_mult;
    out->hidden_size = request->hidden_size;
    out->sinkhorn_repeat = request->sinkhorn_repeat;
    out->rms_eps = request->rms_eps;
    out->hc_pre_eps = request->hc_pre_eps;
    out->hc_sinkhorn_eps = request->hc_sinkhorn_eps;
    out->hc_post_mult_value = request->hc_post_mult_value;
  }
}

int fail_fused(NervaCudaDeepSeekMhcFusedPostPreResult *out,
               cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

void free_fused_buffers(float *device_x, float *device_residual,
                        float *device_post_layer_mix,
                        float *device_comb_res_mix, float *device_fn_weights,
                        float *device_hc_scale, float *device_hc_base,
                        float *device_new_residual,
                        float *device_new_post_mix,
                        float *device_new_comb_mix,
                        float *device_layer_input,
                        int32_t *device_mhc_error) {
  cudaFree(device_x);
  cudaFree(device_residual);
  cudaFree(device_post_layer_mix);
  cudaFree(device_comb_res_mix);
  cudaFree(device_fn_weights);
  cudaFree(device_hc_scale);
  cudaFree(device_hc_base);
  cudaFree(device_new_residual);
  cudaFree(device_new_post_mix);
  cudaFree(device_new_comb_mix);
  cudaFree(device_layer_input);
  cudaFree(device_mhc_error);
}

}  // namespace

extern "C" int nerva_cuda_deepseek_mhc_fused_post_pre(
    const NervaCudaDeepSeekMhcFusedPostPreRequest *request,
    NervaCudaDeepSeekMhcFusedPostPreResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_fused_result(request, out);
  if (request == nullptr || request->tokens == 0 || request->hc_mult == 0 ||
      request->hidden_size == 0 || request->sinkhorn_repeat == 0 ||
      !isfinite(request->rms_eps) || !isfinite(request->hc_pre_eps) ||
      !isfinite(request->hc_sinkhorn_eps) ||
      !isfinite(request->hc_post_mult_value) || request->x == nullptr ||
      request->residual == nullptr || request->post_layer_mix == nullptr ||
      request->comb_res_mix == nullptr || request->fn_weights == nullptr ||
      request->hc_scale == nullptr || request->hc_base == nullptr ||
      request->new_residual == nullptr || request->new_post_mix == nullptr ||
      request->new_comb_mix == nullptr || request->layer_input == nullptr) {
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_fused(out, err);
  }
  if (out->device_count <= 0) {
    return fail_fused(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_fused(out, err);
  }

  uint64_t hc_hidden_values = 0;
  uint64_t hidden_values = 0;
  uint64_t residual_values = 0;
  uint64_t post_values = 0;
  uint64_t comb_values = 0;
  uint64_t hc_mult2 = 0;
  uint64_t hc_mult3 = 0;
  uint64_t fn_values = 0;
  if (!checked_mul_u64(request->hc_mult, request->hidden_size,
                       &hc_hidden_values) ||
      !checked_mul_u64(request->tokens, request->hidden_size,
                       &hidden_values) ||
      !checked_mul_u64(request->tokens, hc_hidden_values, &residual_values) ||
      !checked_mul_u64(request->tokens, request->hc_mult, &post_values) ||
      !checked_mul_u64(request->hc_mult, request->hc_mult, &hc_mult2) ||
      !checked_mul_u64(request->hc_mult, 2u + request->hc_mult, &hc_mult3) ||
      !checked_mul_u64(post_values, request->hc_mult, &comb_values) ||
      !checked_mul_u64(hc_mult3, hc_hidden_values, &fn_values)) {
    return -1;
  }

  const uint64_t x_bytes = hidden_values * sizeof(float);
  const uint64_t residual_bytes = residual_values * sizeof(float);
  const uint64_t post_bytes = post_values * sizeof(float);
  const uint64_t comb_bytes = comb_values * sizeof(float);
  const uint64_t fn_bytes = fn_values * sizeof(float);
  const uint64_t scale_bytes = 3u * sizeof(float);
  const uint64_t base_bytes = hc_mult3 * sizeof(float);
  const uint64_t layer_bytes = hidden_values * sizeof(float);
  const uint64_t error_bytes = sizeof(int32_t);

  float *device_x = nullptr;
  float *device_residual = nullptr;
  float *device_post_layer_mix = nullptr;
  float *device_comb_res_mix = nullptr;
  float *device_fn_weights = nullptr;
  float *device_hc_scale = nullptr;
  float *device_hc_base = nullptr;
  float *device_new_residual = nullptr;
  float *device_new_post_mix = nullptr;
  float *device_new_comb_mix = nullptr;
  float *device_layer_input = nullptr;
  int32_t *device_mhc_error = nullptr;

  err = cudaMalloc(reinterpret_cast<void **>(&device_x), x_bytes);
  if (err == cudaSuccess)
    err = cudaMalloc(reinterpret_cast<void **>(&device_residual),
                     residual_bytes);
  if (err == cudaSuccess)
    err = cudaMalloc(reinterpret_cast<void **>(&device_post_layer_mix),
                     post_bytes);
  if (err == cudaSuccess)
    err = cudaMalloc(reinterpret_cast<void **>(&device_comb_res_mix),
                     comb_bytes);
  if (err == cudaSuccess)
    err = cudaMalloc(reinterpret_cast<void **>(&device_fn_weights),
                     fn_bytes);
  if (err == cudaSuccess)
    err = cudaMalloc(reinterpret_cast<void **>(&device_hc_scale),
                     scale_bytes);
  if (err == cudaSuccess)
    err = cudaMalloc(reinterpret_cast<void **>(&device_hc_base), base_bytes);
  if (err == cudaSuccess)
    err = cudaMalloc(reinterpret_cast<void **>(&device_new_residual),
                     residual_bytes);
  if (err == cudaSuccess)
    err = cudaMalloc(reinterpret_cast<void **>(&device_new_post_mix),
                     post_bytes);
  if (err == cudaSuccess)
    err = cudaMalloc(reinterpret_cast<void **>(&device_new_comb_mix),
                     comb_bytes);
  if (err == cudaSuccess)
    err = cudaMalloc(reinterpret_cast<void **>(&device_layer_input),
                     layer_bytes);
  if (err == cudaSuccess)
    err = cudaMalloc(reinterpret_cast<void **>(&device_mhc_error),
                     error_bytes);
  if (err != cudaSuccess) {
    free_fused_buffers(device_x, device_residual, device_post_layer_mix,
                       device_comb_res_mix, device_fn_weights, device_hc_scale,
                       device_hc_base, device_new_residual,
                       device_new_post_mix, device_new_comb_mix,
                       device_layer_input, device_mhc_error);
    return fail_fused(out, err);
  }
  out->device_arena_bytes =
      x_bytes + residual_bytes + post_bytes + comb_bytes + fn_bytes +
      scale_bytes + base_bytes + residual_bytes + post_bytes + comb_bytes +
      layer_bytes + error_bytes;

  err = cudaMemcpy(device_x, request->x, x_bytes, cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(device_residual, request->residual, residual_bytes,
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(device_post_layer_mix, request->post_layer_mix,
                     post_bytes, cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(device_comb_res_mix, request->comb_res_mix, comb_bytes,
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(device_fn_weights, request->fn_weights, fn_bytes,
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(device_hc_scale, request->hc_scale, scale_bytes,
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(device_hc_base, request->hc_base, base_bytes,
                     cudaMemcpyHostToDevice);
  int32_t host_mhc_error = -99;
  if (err == cudaSuccess)
    err = cudaMemcpy(device_mhc_error, &host_mhc_error, error_bytes,
                     cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    free_fused_buffers(device_x, device_residual, device_post_layer_mix,
                       device_comb_res_mix, device_fn_weights, device_hc_scale,
                       device_hc_base, device_new_residual,
                       device_new_post_mix, device_new_comb_mix,
                       device_layer_input, device_mhc_error);
    return fail_fused(out, err);
  }
  out->h2d_bytes = x_bytes + residual_bytes + post_bytes + comb_bytes +
                   fn_bytes + scale_bytes + base_bytes + error_bytes;

  deepseek_mhc_fused_post_pre_kernel<<<1, 1>>>(
      device_x, device_residual, device_post_layer_mix, device_comb_res_mix,
      device_fn_weights, device_hc_scale, device_hc_base, device_new_residual,
      device_new_post_mix, device_new_comb_mix, device_layer_input,
      request->tokens, request->hc_mult, request->hidden_size,
      request->sinkhorn_repeat, request->rms_eps, request->hc_pre_eps,
      request->hc_sinkhorn_eps, request->hc_post_mult_value, device_mhc_error);
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
    err = cudaMemcpy(request->new_residual, device_new_residual,
                     residual_bytes, cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess && host_mhc_error == 0) {
    err = cudaMemcpy(request->new_post_mix, device_new_post_mix, post_bytes,
                     cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess && host_mhc_error == 0) {
    err = cudaMemcpy(request->new_comb_mix, device_new_comb_mix, comb_bytes,
                     cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess && host_mhc_error == 0) {
    err = cudaMemcpy(request->layer_input, device_layer_input, layer_bytes,
                     cudaMemcpyDeviceToHost);
  }
  out->mhc_error = host_mhc_error;
  out->d2h_bytes =
      host_mhc_error == 0
          ? residual_bytes + post_bytes + comb_bytes + layer_bytes +
                error_bytes
          : error_bytes;
  if (err == cudaSuccess && host_mhc_error == 0) {
    out->new_residual_hash =
        hash_f32_bits(request->new_residual, residual_values);
    out->new_post_mix_hash = hash_f32_bits(request->new_post_mix, post_values);
    out->new_comb_mix_hash = hash_f32_bits(request->new_comb_mix, comb_values);
    out->layer_input_hash = hash_f32_bits(request->layer_input, hidden_values);
  }

  free_fused_buffers(device_x, device_residual, device_post_layer_mix,
                     device_comb_res_mix, device_fn_weights, device_hc_scale,
                     device_hc_base, device_new_residual,
                     device_new_post_mix, device_new_comb_mix,
                     device_layer_input, device_mhc_error);
  if (err != cudaSuccess) {
    return fail_fused(out, err);
  }
  out->status = host_mhc_error == 0 ? 0 : -1;
  return out->status == 0 ? 0 : -1;
}
