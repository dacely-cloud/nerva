#include "nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kHidden = 2u;
constexpr uint32_t kHeads = 1u;
constexpr uint32_t kBlocks = 2u;
constexpr uint32_t kTokensPerBlock = 2u;
constexpr uint32_t kTotalTokens = kBlocks * kTokensPerBlock;
constexpr uint32_t kBlockElements = kTokensPerBlock * kHidden;
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x00000100000001b3ull;

struct AttentionPartial {
  float m;
  float l;
  float o[kHidden];
};

__device__ __host__ float attention_scale() {
  return 1.0f / sqrtf(static_cast<float>(kHidden));
}

__device__ __host__ float dot2(const float *a, const float *b) {
  return a[0] * b[0] + a[1] * b[1];
}

__device__ __host__ void compute_partial_block(const float *query,
                                               const float *keys,
                                               const float *values,
                                               AttentionPartial *partial) {
  partial->m = -INFINITY;
  partial->l = 0.0f;
  partial->o[0] = 0.0f;
  partial->o[1] = 0.0f;
  const float scale = attention_scale();

  for (uint32_t token = 0u; token < kTokensPerBlock; ++token) {
    const uint32_t offset = token * kHidden;
    const float score = dot2(query, keys + offset) * scale;
    const float next_m = fmaxf(partial->m, score);
    const float old_scale =
        partial->l == 0.0f ? 0.0f : expf(partial->m - next_m);
    const float new_scale = expf(score - next_m);
    partial->o[0] = partial->o[0] * old_scale + values[offset] * new_scale;
    partial->o[1] =
        partial->o[1] * old_scale + values[offset + 1u] * new_scale;
    partial->l = partial->l * old_scale + new_scale;
    partial->m = next_m;
  }
}

__global__ void hot_attention_kernel(const float *query,
                                     const float *keys,
                                     const float *values,
                                     AttentionPartial *partial) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  compute_partial_block(query, keys, values, partial);
}

void clear_result(NervaCudaTieredAttentionResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->hidden = kHidden;
  out->heads = kHeads;
  out->blocks = kBlocks;
  out->tokens = kTotalTokens;
  out->resident_kv_bytes = sizeof(float) * kBlockElements * 2u;
  out->device_arena_bytes =
      sizeof(float) * kHidden + out->resident_kv_bytes +
      sizeof(AttentionPartial);
  out->pinned_host_bytes =
      sizeof(float) * kHidden + out->resident_kv_bytes +
      sizeof(AttentionPartial);
}

int fail(NervaCudaTieredAttentionResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

uint64_t hash_f32s(const float *values, uint32_t len) {
  uint64_t hash = kFnvOffset;
  for (uint32_t index = 0u; index < len; ++index) {
    uint32_t bits = 0u;
    memcpy(&bits, &values[index], sizeof(bits));
    for (uint32_t byte_index = 0u; byte_index < 4u; ++byte_index) {
      hash ^= static_cast<uint64_t>((bits >> (byte_index * 8u)) & 0xffu);
      hash *= kFnvPrime;
    }
  }
  return hash;
}

void merge_partials(const AttentionPartial &first,
                    const AttentionPartial &second,
                    float *output) {
  const float merged_m = fmaxf(first.m, second.m);
  const float first_scale = first.l == 0.0f ? 0.0f : expf(first.m - merged_m);
  const float second_scale =
      second.l == 0.0f ? 0.0f : expf(second.m - merged_m);
  const float merged_l = first.l * first_scale + second.l * second_scale;
  output[0] = (first.o[0] * first_scale + second.o[0] * second_scale) /
              merged_l;
  output[1] = (first.o[1] * first_scale + second.o[1] * second_scale) /
              merged_l;
}

}  // namespace

extern "C" int nerva_cuda_tiered_attention_smoke(
    NervaCudaTieredAttentionResult *out) {
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

  const float query[kHidden] = {1.0f, 0.25f};
  const float dram_keys[kBlockElements] = {0.2f, 0.0f, 0.0f, 0.4f};
  const float dram_values[kBlockElements] = {1.0f, 0.0f, 0.5f, 0.5f};
  const float vram_keys[kBlockElements] = {0.5f, 0.1f, -0.2f, 0.3f};
  const float vram_values[kBlockElements] = {0.0f, 1.0f, 2.0f, -1.0f};

  AttentionPartial warm_partial{};
  compute_partial_block(query, dram_keys, dram_values, &warm_partial);
  out->cpu_block_events = 1ull;

  float *device_query = nullptr;
  float *device_keys = nullptr;
  float *device_values = nullptr;
  AttentionPartial *device_partial = nullptr;
  float *host_query = nullptr;
  float *host_kv = nullptr;
  AttentionPartial *host_partial = nullptr;
  cudaStream_t stream = nullptr;

  err = cudaMalloc(reinterpret_cast<void **>(&device_query), sizeof(query));
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  err = cudaMalloc(reinterpret_cast<void **>(&device_keys), sizeof(vram_keys));
  if (err != cudaSuccess) {
    cudaFree(device_query);
    return fail(out, err);
  }
  err = cudaMalloc(reinterpret_cast<void **>(&device_values), sizeof(vram_values));
  if (err != cudaSuccess) {
    cudaFree(device_keys);
    cudaFree(device_query);
    return fail(out, err);
  }
  err = cudaMalloc(reinterpret_cast<void **>(&device_partial),
                   sizeof(AttentionPartial));
  if (err != cudaSuccess) {
    cudaFree(device_values);
    cudaFree(device_keys);
    cudaFree(device_query);
    return fail(out, err);
  }
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_query), sizeof(query),
                      cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFree(device_partial);
    cudaFree(device_values);
    cudaFree(device_keys);
    cudaFree(device_query);
    return fail(out, err);
  }
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_kv),
                      sizeof(vram_keys) + sizeof(vram_values),
                      cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_query);
    cudaFree(device_partial);
    cudaFree(device_values);
    cudaFree(device_keys);
    cudaFree(device_query);
    return fail(out, err);
  }
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_partial),
                      sizeof(AttentionPartial), cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_kv);
    cudaFreeHost(host_query);
    cudaFree(device_partial);
    cudaFree(device_values);
    cudaFree(device_keys);
    cudaFree(device_query);
    return fail(out, err);
  }

  memcpy(host_query, query, sizeof(query));
  memcpy(host_kv, vram_keys, sizeof(vram_keys));
  memcpy(host_kv + kBlockElements, vram_values, sizeof(vram_values));
  memset(host_partial, 0, sizeof(*host_partial));

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    cudaFreeHost(host_partial);
    cudaFreeHost(host_kv);
    cudaFreeHost(host_query);
    cudaFree(device_partial);
    cudaFree(device_values);
    cudaFree(device_keys);
    cudaFree(device_query);
    return fail(out, err);
  }

  err = cudaMemcpyAsync(device_query, host_query, sizeof(query),
                        cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess) {
    out->h2d_bytes += sizeof(query);
    err = cudaMemcpyAsync(device_keys, host_kv, sizeof(vram_keys),
                          cudaMemcpyHostToDevice, stream);
    out->h2d_bytes += sizeof(vram_keys);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(device_values, host_kv + kBlockElements,
                          sizeof(vram_values), cudaMemcpyHostToDevice, stream);
    out->h2d_bytes += sizeof(vram_values);
  }
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(device_partial, 0, sizeof(AttentionPartial), stream);
  }
  if (err == cudaSuccess) {
    hot_attention_kernel<<<1, 1, 0, stream>>>(
        device_query, device_keys, device_values, device_partial);
    out->kernel_launches = 1ull;
    out->device_block_events = 1ull;
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_partial, device_partial,
                          sizeof(AttentionPartial), cudaMemcpyDeviceToHost,
                          stream);
    out->d2h_bytes = sizeof(AttentionPartial);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
    out->sync_calls = 1ull;
  }

  if (err == cudaSuccess) {
    merge_partials(warm_partial, *host_partial, out->output);
    out->output_hash = hash_f32s(out->output, kHidden);
    const bool output_valid =
        isfinite(out->output[0]) && isfinite(out->output[1]) &&
        host_partial->l > 0.0f && warm_partial.l > 0.0f;
    out->status = output_valid ? 0 : -1;
  } else {
    out->cuda_error = static_cast<int32_t>(err);
    out->status = -1;
  }

  cudaStreamDestroy(stream);
  cudaFreeHost(host_partial);
  cudaFreeHost(host_kv);
  cudaFreeHost(host_query);
  cudaFree(device_partial);
  cudaFree(device_values);
  cudaFree(device_keys);
  cudaFree(device_query);
  return out->status == 0 ? 0 : -1;
}
