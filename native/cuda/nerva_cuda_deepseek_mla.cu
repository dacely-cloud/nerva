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
