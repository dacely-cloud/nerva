#include "nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kHeads = 2;
constexpr uint32_t kTokens = 3;
constexpr uint32_t kKvLoraRank = 3;
constexpr uint32_t kQkNopeHeadDim = 2;
constexpr uint32_t kQkRopeHeadDim = 1;
constexpr uint32_t kVHeadDim = 2;
constexpr uint32_t kOutputValues = kHeads * kVHeadDim;
constexpr float kSoftmaxScale = 0.7f;
constexpr uint32_t kMaxDynamicKvLoraRank = 1024;

constexpr float kQNope[kHeads * kQkNopeHeadDim] = {
    0.2f, -0.3f,
    0.4f, 0.1f,
};
constexpr float kQPe[kHeads * kQkRopeHeadDim] = {
    0.15f,
    -0.25f,
};
constexpr float kKvC[kTokens * kKvLoraRank] = {
    0.3f, -0.1f, 0.2f,
    -0.4f, 0.5f, 0.1f,
    0.2f, 0.4f, -0.3f,
};
constexpr float kKPe[kTokens * kQkRopeHeadDim] = {
    0.05f,
    -0.2f,
    0.3f,
};
constexpr float kWUk[kKvLoraRank * kHeads * kQkNopeHeadDim] = {
    0.3f, -0.2f, 0.1f, 0.4f,
    -0.5f, 0.2f, 0.6f, -0.1f,
    0.7f, 0.3f, -0.2f, 0.5f,
};
constexpr float kWUv[kKvLoraRank * kHeads * kVHeadDim] = {
    0.2f, -0.4f, 0.5f, 0.1f,
    -0.3f, 0.6f, 0.4f, -0.2f,
    0.7f, 0.2f, -0.1f, 0.3f,
};

__device__ float dot_device(const float *left, const float *right, uint32_t len) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < len; ++i) {
    sum += left[i] * right[i];
  }
  return sum;
}

__global__ void deepseek_mla_smoke_kernel(float *output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const float q_nope[kHeads * kQkNopeHeadDim] = {
      0.2f, -0.3f,
      0.4f, 0.1f,
  };
  const float q_pe_fixture[kHeads * kQkRopeHeadDim] = {
      0.15f,
      -0.25f,
  };
  const float kv_c[kTokens * kKvLoraRank] = {
      0.3f, -0.1f, 0.2f,
      -0.4f, 0.5f, 0.1f,
      0.2f, 0.4f, -0.3f,
  };
  const float k_pe_fixture[kTokens * kQkRopeHeadDim] = {
      0.05f,
      -0.2f,
      0.3f,
  };
  const float w_uk[kKvLoraRank * kHeads * kQkNopeHeadDim] = {
      0.3f, -0.2f, 0.1f, 0.4f,
      -0.5f, 0.2f, 0.6f, -0.1f,
      0.7f, 0.3f, -0.2f, 0.5f,
  };
  const float w_uv[kKvLoraRank * kHeads * kVHeadDim] = {
      0.2f, -0.4f, 0.5f, 0.1f,
      -0.3f, 0.6f, 0.4f, -0.2f,
      0.7f, 0.2f, -0.1f, 0.3f,
  };

  for (uint32_t i = 0; i < kOutputValues; ++i) {
    output[i] = 0.0f;
  }

  for (uint32_t head = 0; head < kHeads; ++head) {
    float ql_nope[kKvLoraRank];
    for (uint32_t latent = 0; latent < kKvLoraRank; ++latent) {
      float sum = 0.0f;
      for (uint32_t nope = 0; nope < kQkNopeHeadDim; ++nope) {
        const uint32_t q_idx = head * kQkNopeHeadDim + nope;
        const uint32_t w_idx =
            (latent * kHeads + head) * kQkNopeHeadDim + nope;
        sum += q_nope[q_idx] * w_uk[w_idx];
      }
      ql_nope[latent] = sum;
    }

    float latent_output[kKvLoraRank] = {0.0f, 0.0f, 0.0f};
    float local_m = -INFINITY;
    float local_l = 0.0f;
    for (uint32_t token = 0; token < kTokens; ++token) {
      const float *kv = kv_c + token * kKvLoraRank;
      const float *k_pe = k_pe_fixture + token * kQkRopeHeadDim;
      const float *q_pe = q_pe_fixture + head * kQkRopeHeadDim;
      const float score =
          (dot_device(ql_nope, kv, kKvLoraRank) +
           dot_device(q_pe, k_pe, kQkRopeHeadDim)) *
          kSoftmaxScale;
      const float next_m = fmaxf(local_m, score);
      const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
      const float new_scale = expf(score - next_m);
      for (uint32_t latent = 0; latent < kKvLoraRank; ++latent) {
        latent_output[latent] =
            latent_output[latent] * old_scale + kv[latent] * new_scale;
      }
      local_l = local_l * old_scale + new_scale;
      local_m = next_m;
    }

    for (uint32_t latent = 0; latent < kKvLoraRank; ++latent) {
      latent_output[latent] /= local_l;
    }

    for (uint32_t v = 0; v < kVHeadDim; ++v) {
      float sum = 0.0f;
      for (uint32_t latent = 0; latent < kKvLoraRank; ++latent) {
        const uint32_t w_idx = (latent * kHeads + head) * kVHeadDim + v;
        sum += latent_output[latent] * w_uv[w_idx];
      }
      output[head * kVHeadDim + v] = sum;
    }
  }
}

__global__ void deepseek_mla_decode_kernel(const float *q_nope,
                                           const float *q_pe,
                                           const float *kv_c,
                                           const float *k_pe,
                                           const float *w_uk,
                                           const float *w_uv,
                                           float *output,
                                           uint32_t heads,
                                           uint32_t tokens,
                                           uint32_t kv_lora_rank,
                                           uint32_t qk_nope_head_dim,
                                           uint32_t qk_rope_head_dim,
                                           uint32_t v_head_dim,
                                           float softmax_scale,
                                           int32_t *decode_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (heads == 0 || tokens == 0 || kv_lora_rank == 0 ||
      kv_lora_rank > kMaxDynamicKvLoraRank || qk_nope_head_dim == 0 ||
      v_head_dim == 0) {
    *decode_error = -1;
    return;
  }

  for (uint32_t i = 0; i < heads * v_head_dim; ++i) {
    output[i] = 0.0f;
  }

  float ql_nope[kMaxDynamicKvLoraRank];
  float latent_output[kMaxDynamicKvLoraRank];

  for (uint32_t head = 0; head < heads; ++head) {
    for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
      float sum = 0.0f;
      for (uint32_t nope = 0; nope < qk_nope_head_dim; ++nope) {
        const uint32_t q_idx = head * qk_nope_head_dim + nope;
        const uint32_t w_idx =
            (latent * heads + head) * qk_nope_head_dim + nope;
        sum += q_nope[q_idx] * w_uk[w_idx];
      }
      ql_nope[latent] = sum;
      latent_output[latent] = 0.0f;
    }

    float local_m = -INFINITY;
    float local_l = 0.0f;
    for (uint32_t token = 0; token < tokens; ++token) {
      const float *kv = kv_c + token * kv_lora_rank;
      const float *k_pe_token = k_pe + token * qk_rope_head_dim;
      const float *q_pe_head = q_pe + head * qk_rope_head_dim;
      const float score =
          (dot_device(ql_nope, kv, kv_lora_rank) +
           dot_device(q_pe_head, k_pe_token, qk_rope_head_dim)) *
          softmax_scale;
      const float next_m = fmaxf(local_m, score);
      const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
      const float new_scale = expf(score - next_m);
      for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
        latent_output[latent] =
            latent_output[latent] * old_scale + kv[latent] * new_scale;
      }
      local_l = local_l * old_scale + new_scale;
      local_m = next_m;
    }

    if (local_l == 0.0f) {
      *decode_error = -2;
      return;
    }
    for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
      latent_output[latent] /= local_l;
    }

    for (uint32_t v = 0; v < v_head_dim; ++v) {
      float sum = 0.0f;
      for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
        const uint32_t w_idx = (latent * heads + head) * v_head_dim + v;
        sum += latent_output[latent] * w_uv[w_idx];
      }
      output[head * v_head_dim + v] = sum;
    }
  }
  *decode_error = 0;
}

__global__ void deepseek_qkv_rmsnorm_kernel(const float *q,
                                            const float *kv,
                                            const float *q_weight,
                                            const float *kv_weight,
                                            float *q_out,
                                            float *kv_out,
                                            uint32_t q_size,
                                            uint32_t kv_size,
                                            float eps) {
  const uint32_t token_idx = blockIdx.x;
  const uint32_t task_idx = blockIdx.y;
  const bool is_q = task_idx == 0;
  const uint32_t size = is_q ? q_size : kv_size;
  const float *input = is_q ? q : kv;
  const float *weight = is_q ? q_weight : kv_weight;
  float *output = is_q ? q_out : kv_out;
  const uint64_t row_base =
      static_cast<uint64_t>(token_idx) * static_cast<uint64_t>(size);

  float local_sum = 0.0f;
  for (uint32_t dim = threadIdx.x; dim < size; dim += blockDim.x) {
    const float x = input[row_base + dim];
    local_sum += x * x;
  }

  __shared__ float partial[256];
  partial[threadIdx.x] = local_sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float rrms = rsqrtf(partial[0] / static_cast<float>(size) + eps);
  for (uint32_t dim = threadIdx.x; dim < size; dim += blockDim.x) {
    output[row_base + dim] = input[row_base + dim] * rrms * weight[dim];
  }
}

float host_softmax_score(uint32_t head, uint32_t token) {
  float score = 0.0f;
  const float *kv = kKvC + token * kKvLoraRank;
  for (uint32_t nope = 0; nope < kQkNopeHeadDim; ++nope) {
    float k_nope = 0.0f;
    for (uint32_t latent = 0; latent < kKvLoraRank; ++latent) {
      const uint32_t w_idx =
          (latent * kHeads + head) * kQkNopeHeadDim + nope;
      k_nope += kv[latent] * kWUk[w_idx];
    }
    score += kQNope[head * kQkNopeHeadDim + nope] * k_nope;
  }
  for (uint32_t rope = 0; rope < kQkRopeHeadDim; ++rope) {
    score += kQPe[head * kQkRopeHeadDim + rope] *
             kKPe[token * kQkRopeHeadDim + rope];
  }
  return score * kSoftmaxScale;
}

float host_value(uint32_t head, uint32_t token, uint32_t v) {
  float value = 0.0f;
  const float *kv = kKvC + token * kKvLoraRank;
  for (uint32_t latent = 0; latent < kKvLoraRank; ++latent) {
    const uint32_t w_idx = (latent * kHeads + head) * kVHeadDim + v;
    value += kv[latent] * kWUv[w_idx];
  }
  return value;
}

void expected_expanded_mha(float *expected) {
  for (uint32_t i = 0; i < kOutputValues; ++i) {
    expected[i] = 0.0f;
  }

  for (uint32_t head = 0; head < kHeads; ++head) {
    float scores[kTokens];
    float max_score = -INFINITY;
    for (uint32_t token = 0; token < kTokens; ++token) {
      scores[token] = host_softmax_score(head, token);
      max_score = fmaxf(max_score, scores[token]);
    }

    float normalizer = 0.0f;
    for (uint32_t token = 0; token < kTokens; ++token) {
      normalizer += expf(scores[token] - max_score);
    }

    for (uint32_t token = 0; token < kTokens; ++token) {
      const float prob = expf(scores[token] - max_score) / normalizer;
      for (uint32_t v = 0; v < kVHeadDim; ++v) {
        expected[head * kVHeadDim + v] += prob * host_value(head, token, v);
      }
    }
  }
}

uint32_t f32_bits(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint64_t hash_f32_bits(const float *values, uint32_t len) {
  uint64_t hash = 1469598103934665603ull;
  for (uint32_t i = 0; i < len; ++i) {
    const uint32_t bits = f32_bits(values[i]);
    for (uint32_t byte = 0; byte < 4; ++byte) {
      hash ^= static_cast<uint8_t>((bits >> (byte * 8)) & 0xffu);
      hash *= 1099511628211ull;
    }
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

void clear_result(NervaCudaDeepSeekMlaSmokeResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->heads = kHeads;
  out->tokens = kTokens;
  out->kv_lora_rank = kKvLoraRank;
  out->qk_nope_head_dim = kQkNopeHeadDim;
  out->qk_rope_head_dim = kQkRopeHeadDim;
  out->v_head_dim = kVHeadDim;
  out->softmax_scale = kSoftmaxScale;
}

int fail(NervaCudaDeepSeekMlaSmokeResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

void clear_decode_result(const NervaCudaDeepSeekMlaDecodeRequest *request,
                         NervaCudaDeepSeekMlaDecodeResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->decode_error = -1;
  if (request != nullptr) {
    out->heads = request->heads;
    out->tokens = request->tokens;
    out->kv_lora_rank = request->kv_lora_rank;
    out->qk_nope_head_dim = request->qk_nope_head_dim;
    out->qk_rope_head_dim = request->qk_rope_head_dim;
    out->v_head_dim = request->v_head_dim;
    out->softmax_scale = request->softmax_scale;
  }
}

int fail_decode(NervaCudaDeepSeekMlaDecodeResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

void clear_qkv_rmsnorm_result(const NervaCudaDeepSeekQKvRmsNormRequest *request,
                              NervaCudaDeepSeekQKvRmsNormResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request != nullptr) {
    out->num_tokens = request->num_tokens;
    out->q_size = request->q_size;
    out->kv_size = request->kv_size;
    out->eps = request->eps;
  }
}

int fail_qkv_rmsnorm(NervaCudaDeepSeekQKvRmsNormResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

bool validate_decode_request(const NervaCudaDeepSeekMlaDecodeRequest *request) {
  return request != nullptr && request->q_nope != nullptr &&
         request->q_pe != nullptr && request->kv_c != nullptr &&
         request->k_pe != nullptr && request->w_uk != nullptr &&
         request->w_uv != nullptr && request->output != nullptr &&
         request->heads > 0 && request->tokens > 0 &&
         request->kv_lora_rank > 0 &&
         request->kv_lora_rank <= kMaxDynamicKvLoraRank &&
         request->qk_nope_head_dim > 0 && request->v_head_dim > 0;
}

bool validate_qkv_rmsnorm_request(
    const NervaCudaDeepSeekQKvRmsNormRequest *request) {
  return request != nullptr && request->q != nullptr && request->kv != nullptr &&
         request->q_weight != nullptr && request->kv_weight != nullptr &&
         request->q_out != nullptr && request->kv_out != nullptr &&
         request->num_tokens > 0 && request->q_size > 0 &&
         request->kv_size > 0 && isfinite(request->eps) &&
         request->eps >= 0.0f;
}

}  // namespace

extern "C" int nerva_cuda_deepseek_mla_smoke(
    NervaCudaDeepSeekMlaSmokeResult *out) {
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

  float *device_output = nullptr;
  float *host_output = nullptr;
  cudaStream_t stream = nullptr;
  const uint64_t output_bytes = sizeof(float) * kOutputValues;

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

  deepseek_mla_smoke_kernel<<<1, 1, 0, stream>>>(device_output);
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

  memcpy(out->output, host_output, output_bytes);
  out->output_hash = hash_f32_bits(out->output, kOutputValues);

  float expected[kOutputValues];
  expected_expanded_mha(expected);
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

extern "C" int nerva_cuda_deepseek_qkv_rmsnorm(
    const NervaCudaDeepSeekQKvRmsNormRequest *request,
    NervaCudaDeepSeekQKvRmsNormResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_qkv_rmsnorm_result(request, out);
  if (!validate_qkv_rmsnorm_request(request)) {
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_qkv_rmsnorm(out, err);
  }
  if (out->device_count <= 0) {
    return fail_qkv_rmsnorm(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_qkv_rmsnorm(out, err);
  }

  float *d_q = nullptr;
  float *d_kv = nullptr;
  float *d_q_weight = nullptr;
  float *d_kv_weight = nullptr;
  float *d_q_out = nullptr;
  float *d_kv_out = nullptr;
  float *h_q_out = nullptr;
  float *h_kv_out = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t q_values =
      static_cast<uint64_t>(request->num_tokens) * request->q_size;
  const uint64_t kv_values =
      static_cast<uint64_t>(request->num_tokens) * request->kv_size;
  const uint64_t q_bytes = q_values * sizeof(float);
  const uint64_t kv_bytes = kv_values * sizeof(float);
  const uint64_t q_weight_bytes =
      static_cast<uint64_t>(request->q_size) * sizeof(float);
  const uint64_t kv_weight_bytes =
      static_cast<uint64_t>(request->kv_size) * sizeof(float);
  if (q_values > UINT32_MAX || kv_values > UINT32_MAX) {
    return -1;
  }

  err = cudaMalloc(reinterpret_cast<void **>(&d_q), q_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_kv), kv_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_q_weight), q_weight_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_kv_weight), kv_weight_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_q_out), q_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_kv_out), kv_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      q_bytes + kv_bytes + q_weight_bytes + kv_weight_bytes + q_bytes +
      kv_bytes;

  err =
      cudaHostAlloc(reinterpret_cast<void **>(&h_q_out), q_bytes, cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_kv_out),
                      kv_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = q_bytes + kv_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(d_q, request->q, q_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err =
      cudaMemcpyAsync(d_kv, request->kv, kv_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(
      d_q_weight, request->q_weight, q_weight_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_kv_weight,
                        request->kv_weight,
                        kv_weight_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = q_bytes + kv_bytes + q_weight_bytes + kv_weight_bytes;

  {
    constexpr uint32_t threads = 256;
    const dim3 grid(request->num_tokens, 2, 1);
    deepseek_qkv_rmsnorm_kernel<<<grid, threads, 0, stream>>>(d_q,
                                                             d_kv,
                                                             d_q_weight,
                                                             d_kv_weight,
                                                             d_q_out,
                                                             d_kv_out,
                                                             request->q_size,
                                                             request->kv_size,
                                                             request->eps);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err =
      cudaMemcpyAsync(h_q_out, d_q_out, q_bytes, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(
      h_kv_out, d_kv_out, kv_bytes, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = q_bytes + kv_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->q_out, h_q_out, q_bytes);
  memcpy(request->kv_out, h_kv_out, kv_bytes);
  out->output_hash =
      hash_f32_bits(request->q_out, static_cast<uint32_t>(q_values));
  out->output_hash *= 1099511628211ull;
  out->output_hash ^=
      hash_f32_bits(request->kv_out, static_cast<uint32_t>(kv_values));
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_kv_out != nullptr) cudaFreeHost(h_kv_out);
  if (h_q_out != nullptr) cudaFreeHost(h_q_out);
  if (d_kv_out != nullptr) cudaFree(d_kv_out);
  if (d_q_out != nullptr) cudaFree(d_q_out);
  if (d_kv_weight != nullptr) cudaFree(d_kv_weight);
  if (d_q_weight != nullptr) cudaFree(d_q_weight);
  if (d_kv != nullptr) cudaFree(d_kv);
  if (d_q != nullptr) cudaFree(d_q);

  if (err != cudaSuccess) {
    return fail_qkv_rmsnorm(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_mla_decode(
    const NervaCudaDeepSeekMlaDecodeRequest *request,
    NervaCudaDeepSeekMlaDecodeResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_decode_result(request, out);
  if (!validate_decode_request(request)) {
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_decode(out, err);
  }
  if (out->device_count <= 0) {
    return fail_decode(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_decode(out, err);
  }

  float *d_q_nope = nullptr;
  float *d_q_pe = nullptr;
  float *d_kv_c = nullptr;
  float *d_k_pe = nullptr;
  float *d_w_uk = nullptr;
  float *d_w_uv = nullptr;
  float *d_output = nullptr;
  float *h_output = nullptr;
  int32_t *d_decode_error = nullptr;
  int32_t h_decode_error = -1;
  cudaStream_t stream = nullptr;

  const uint64_t q_nope_bytes =
      sizeof(float) * static_cast<uint64_t>(request->heads) *
      request->qk_nope_head_dim;
  const uint64_t q_pe_bytes =
      sizeof(float) * static_cast<uint64_t>(request->heads) *
      request->qk_rope_head_dim;
  const uint64_t kv_c_bytes =
      sizeof(float) * static_cast<uint64_t>(request->tokens) *
      request->kv_lora_rank;
  const uint64_t k_pe_bytes =
      sizeof(float) * static_cast<uint64_t>(request->tokens) *
      request->qk_rope_head_dim;
  const uint64_t w_uk_bytes =
      sizeof(float) * static_cast<uint64_t>(request->kv_lora_rank) *
      request->heads * request->qk_nope_head_dim;
  const uint64_t w_uv_bytes =
      sizeof(float) * static_cast<uint64_t>(request->kv_lora_rank) *
      request->heads * request->v_head_dim;
  const uint64_t output_bytes =
      sizeof(float) * static_cast<uint64_t>(request->heads) *
      request->v_head_dim;
  const uint64_t decode_error_bytes = sizeof(int32_t);

  err = cudaMalloc(reinterpret_cast<void **>(&d_q_nope), q_nope_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_q_pe), q_pe_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_kv_c), kv_c_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_k_pe), k_pe_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_w_uk), w_uk_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_w_uv), w_uv_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output), output_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_decode_error),
                   decode_error_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes = q_nope_bytes + q_pe_bytes + kv_c_bytes +
                            k_pe_bytes + w_uk_bytes + w_uv_bytes +
                            output_bytes + decode_error_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output),
                      output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(d_q_nope,
                        request->q_nope,
                        q_nope_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_q_pe,
                        request->q_pe,
                        q_pe_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_kv_c,
                        request->kv_c,
                        kv_c_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_k_pe,
                        request->k_pe,
                        k_pe_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_w_uk,
                        request->w_uk,
                        w_uk_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_w_uv,
                        request->w_uv,
                        w_uv_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_decode_error,
                        &h_decode_error,
                        decode_error_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = q_nope_bytes + q_pe_bytes + kv_c_bytes + k_pe_bytes +
                   w_uk_bytes + w_uv_bytes + decode_error_bytes;

  deepseek_mla_decode_kernel<<<1, 1, 0, stream>>>(
      d_q_nope,
      d_q_pe,
      d_kv_c,
      d_k_pe,
      d_w_uk,
      d_w_uv,
      d_output,
      request->heads,
      request->tokens,
      request->kv_lora_rank,
      request->qk_nope_head_dim,
      request->qk_rope_head_dim,
      request->v_head_dim,
      request->softmax_scale,
      d_decode_error);
  out->kernel_launches += 1;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_output,
                        d_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(&h_decode_error,
                        d_decode_error,
                        decode_error_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_bytes + decode_error_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  out->decode_error = h_decode_error;
  if (h_decode_error == 0) {
    memcpy(request->output, h_output, output_bytes);
    out->output_hash = hash_f32_bits(
        request->output,
        static_cast<uint32_t>(request->heads * request->v_head_dim));
    out->status = 0;
  }

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output != nullptr) cudaFreeHost(h_output);
  if (d_decode_error != nullptr) cudaFree(d_decode_error);
  if (d_output != nullptr) cudaFree(d_output);
  if (d_w_uv != nullptr) cudaFree(d_w_uv);
  if (d_w_uk != nullptr) cudaFree(d_w_uk);
  if (d_k_pe != nullptr) cudaFree(d_k_pe);
  if (d_kv_c != nullptr) cudaFree(d_kv_c);
  if (d_q_pe != nullptr) cudaFree(d_q_pe);
  if (d_q_nope != nullptr) cudaFree(d_q_nope);

  if (err != cudaSuccess) {
    return fail_decode(out, err);
  }
  return out->status == 0 ? 0 : -1;
}
