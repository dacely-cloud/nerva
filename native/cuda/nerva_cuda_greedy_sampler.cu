#include "nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kRequestId = 1u;
constexpr uint32_t kSequenceId = 1u;
constexpr uint32_t kVocabSize = 4u;
constexpr uint64_t kTokenIndex = 0ull;
constexpr uint32_t kCompletionDeviceComplete = 1u;

__global__ void greedy_sample_kernel(const float *logits,
                                     uint32_t vocab_size,
                                     NervaCudaSyntheticTokenSlot *slot) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || vocab_size == 0) {
    return;
  }

  uint32_t best_index = 0;
  float best_value = logits[0];
  for (uint32_t index = 1; index < vocab_size; ++index) {
    const float value = logits[index];
    if (isfinite(value) && value > best_value) {
      best_value = value;
      best_index = index;
    }
  }

  slot->request_id = kRequestId;
  slot->sequence_id = kSequenceId;
  slot->token_index = kTokenIndex;
  slot->token = best_index;
  slot->version = slot->version + 1ull;
  slot->completion = kCompletionDeviceComplete;
  slot->host_copied = 0u;
}

void clear_result(NervaCudaGreedySamplerResult *out) {
  out->status = -1;
  out->cuda_error = 0;
  out->device_count = 0;
  out->vocab_size = kVocabSize;
  out->token_index = kTokenIndex;
  out->token = 0;
  out->slot_version = 0;
  out->completion = 0;
  out->device_arena_bytes = sizeof(float) * kVocabSize + sizeof(NervaCudaSyntheticTokenSlot);
  out->pinned_host_bytes = sizeof(float) * kVocabSize + sizeof(NervaCudaSyntheticTokenSlot);
  out->h2d_bytes = 0;
  out->d2h_bytes = 0;
  out->kernel_launches = 0;
  out->sync_calls = 0;
  out->hot_path_allocations = 0;
}

int fail(NervaCudaGreedySamplerResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

}  // namespace

extern "C" int nerva_cuda_greedy_sampler_smoke(NervaCudaGreedySamplerResult *out) {
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

  float *device_logits = nullptr;
  NervaCudaSyntheticTokenSlot *device_slot = nullptr;
  float *host_logits = nullptr;
  NervaCudaSyntheticTokenSlot *host_slot = nullptr;
  cudaStream_t stream = nullptr;

  err = cudaMalloc(reinterpret_cast<void **>(&device_logits), sizeof(float) * kVocabSize);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  err = cudaMalloc(reinterpret_cast<void **>(&device_slot), sizeof(NervaCudaSyntheticTokenSlot));
  if (err != cudaSuccess) {
    cudaFree(device_logits);
    return fail(out, err);
  }
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_logits), sizeof(float) * kVocabSize,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFree(device_slot);
    cudaFree(device_logits);
    return fail(out, err);
  }
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_slot),
                      sizeof(NervaCudaSyntheticTokenSlot), cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_logits);
    cudaFree(device_slot);
    cudaFree(device_logits);
    return fail(out, err);
  }
  host_logits[0] = -2.0f;
  host_logits[1] = 0.5f;
  host_logits[2] = 3.0f;
  host_logits[3] = 2.0f;
  memset(host_slot, 0, sizeof(*host_slot));

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    cudaFreeHost(host_slot);
    cudaFreeHost(host_logits);
    cudaFree(device_slot);
    cudaFree(device_logits);
    return fail(out, err);
  }

  err = cudaMemsetAsync(device_slot, 0, sizeof(NervaCudaSyntheticTokenSlot), stream);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(device_logits, host_logits, sizeof(float) * kVocabSize,
                          cudaMemcpyHostToDevice, stream);
    out->h2d_bytes = sizeof(float) * kVocabSize;
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
    out->sync_calls += 1;
  }

  if (err == cudaSuccess) {
    greedy_sample_kernel<<<1, 1, 0, stream>>>(device_logits, kVocabSize, device_slot);
    out->kernel_launches = 1;
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_slot, device_slot, sizeof(NervaCudaSyntheticTokenSlot),
                          cudaMemcpyDeviceToHost, stream);
    out->d2h_bytes = sizeof(NervaCudaSyntheticTokenSlot);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
    out->sync_calls += 1;
  }

  if (err == cudaSuccess) {
    out->token_index = host_slot->token_index;
    out->token = host_slot->token;
    out->slot_version = host_slot->version;
    out->completion = host_slot->completion;
    out->status = host_slot->request_id == kRequestId &&
                          host_slot->sequence_id == kSequenceId &&
                          host_slot->token_index == kTokenIndex &&
                          host_slot->token == 2u &&
                          host_slot->version == 1ull &&
                          host_slot->completion == kCompletionDeviceComplete
                      ? 0
                      : -1;
  } else {
    out->cuda_error = static_cast<int32_t>(err);
    out->status = -1;
  }

  cudaStreamDestroy(stream);
  cudaFreeHost(host_slot);
  cudaFreeHost(host_logits);
  cudaFree(device_slot);
  cudaFree(device_logits);
  return out->status == 0 ? 0 : -1;
}
