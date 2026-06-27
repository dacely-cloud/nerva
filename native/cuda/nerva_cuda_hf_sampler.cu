#include "nerva_cuda_api.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kDTypeF16 = 0;
constexpr uint32_t kDTypeBF16 = 1;
constexpr uint32_t kRequestId = 1;
constexpr uint32_t kSequenceId = 1;
constexpr uint32_t kCompletionDeviceComplete = 1;
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x00000100000001b3ull;

struct SamplerLayout {
  uint64_t hidden_bits;
  uint64_t final_norm;
  uint64_t lm_head;
};

__device__ float encoded_to_f32(uint16_t value, uint32_t dtype) {
  if (dtype == kDTypeBF16) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
  }
  return __half2float(__ushort_as_half(value));
}

__device__ void rms_norm(const float *input, const uint16_t *weight, uint32_t hidden,
                         uint32_t dtype, float eps, float *output) {
  float mean_square = 0.0f;
  for (uint32_t index = 0; index < hidden; ++index) {
    mean_square += input[index] * input[index];
  }
  const float scale = rsqrtf(mean_square / static_cast<float>(hidden) + eps);
  for (uint32_t index = 0; index < hidden; ++index) {
    output[index] = input[index] * scale * encoded_to_f32(weight[index], dtype);
  }
}

__device__ void mat_vec(const uint16_t *matrix, const float *input, uint32_t rows,
                        uint32_t cols, uint32_t dtype, float *output) {
  for (uint32_t row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < cols; ++col) {
      sum += encoded_to_f32(matrix[row * cols + col], dtype) * input[col];
    }
    output[row] = sum;
  }
}

__global__ void hf_sample_kernel(const uint16_t *arena, SamplerLayout layout,
                                 uint32_t dtype, uint32_t hidden, uint32_t vocab_size,
                                 uint64_t token_index, float rms_eps, float *scratch,
                                 NervaCudaSyntheticTokenSlot *slot) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  float *decoded = scratch;
  float *normed = decoded + hidden;
  float *logits = normed + hidden;
  for (uint32_t index = 0; index < hidden; ++index) {
    decoded[index] = encoded_to_f32(arena[layout.hidden_bits + index], dtype);
  }
  rms_norm(decoded, arena + layout.final_norm, hidden, dtype, rms_eps, normed);
  mat_vec(arena + layout.lm_head, normed, vocab_size, hidden, dtype, logits);

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
  slot->token_index = token_index;
  slot->token = best_index;
  slot->version = slot->version + 1ull;
  slot->completion = kCompletionDeviceComplete;
  slot->host_copied = 0;
}

uint64_t hash_token(uint32_t token) {
  uint64_t hash = kFnvOffset;
  for (uint32_t byte = 0; byte < 4; ++byte) {
    hash ^= static_cast<uint64_t>((token >> (8u * byte)) & 0xffu);
    hash *= kFnvPrime;
  }
  return hash;
}

bool valid_request(const NervaCudaHfSamplerRequest *request) {
  return request != nullptr && request->hidden_bits != nullptr &&
         request->final_norm_weight != nullptr && request->lm_head != nullptr &&
         request->hidden > 0 && request->vocab_size > 0 && request->dtype <= kDTypeBF16;
}

void clear_result(const NervaCudaHfSamplerRequest *request,
                  NervaCudaHfSamplerResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request != nullptr) {
    out->dtype = request->dtype;
    out->hidden = request->hidden;
    out->vocab_size = request->vocab_size;
    out->token_index = request->token_index;
  }
}

int fail(NervaCudaHfSamplerResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  return -1;
}

}  // namespace

extern "C" int nerva_cuda_hf_sample_u16(const NervaCudaHfSamplerRequest *request,
                                        NervaCudaHfSamplerResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(request, out);
  if (!valid_request(request)) {
    return -1;
  }
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

  SamplerLayout layout{};
  const uint64_t hidden = request->hidden;
  const uint64_t vocab_size = request->vocab_size;
  layout.hidden_bits = 0;
  layout.final_norm = hidden;
  layout.lm_head = layout.final_norm + hidden;
  const uint64_t arena_elements = layout.lm_head + vocab_size * hidden;
  const uint64_t arena_bytes = arena_elements * sizeof(uint16_t);
  const uint64_t scratch_bytes = (hidden * 2 + vocab_size) * sizeof(float);

  uint16_t *host_arena = nullptr;
  uint16_t *device_arena = nullptr;
  float *device_scratch = nullptr;
  NervaCudaSyntheticTokenSlot *host_slot = nullptr;
  NervaCudaSyntheticTokenSlot *device_slot = nullptr;
  cudaStream_t stream = nullptr;

  err = cudaHostAlloc(reinterpret_cast<void **>(&host_arena), arena_bytes, cudaHostAllocDefault);
  if (err == cudaSuccess) err = cudaHostAlloc(reinterpret_cast<void **>(&host_slot), sizeof(*host_slot),
                                             cudaHostAllocDefault);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_arena), arena_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_scratch), scratch_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_slot), sizeof(*device_slot));
  if (err == cudaSuccess) err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    fail(out, err);
    cudaFree(device_slot);
    cudaFree(device_scratch);
    cudaFree(device_arena);
    cudaFreeHost(host_slot);
    cudaFreeHost(host_arena);
    return -1;
  }

  memset(host_arena, 0, arena_bytes);
  memset(host_slot, 0, sizeof(*host_slot));
  memcpy(host_arena + layout.hidden_bits, request->hidden_bits, hidden * sizeof(uint16_t));
  memcpy(host_arena + layout.final_norm, request->final_norm_weight, hidden * sizeof(uint16_t));
  memcpy(host_arena + layout.lm_head, request->lm_head, vocab_size * hidden * sizeof(uint16_t));

  err = cudaMemcpyAsync(device_arena, host_arena, arena_bytes, cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess) {
    out->h2d_bytes = arena_bytes;
    err = cudaMemsetAsync(device_slot, 0, sizeof(*device_slot), stream);
  }
  if (err == cudaSuccess) {
    hf_sample_kernel<<<1, 1, 0, stream>>>(
        device_arena, layout, request->dtype, request->hidden, request->vocab_size,
        request->token_index, request->rms_eps, device_scratch, device_slot);
    out->kernel_launches = 1;
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_slot, device_slot, sizeof(*host_slot),
                          cudaMemcpyDeviceToHost, stream);
    out->d2h_bytes = sizeof(*host_slot);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
    out->sync_calls = 1;
  }

  if (err == cudaSuccess) {
    out->token_index = host_slot->token_index;
    out->token = host_slot->token;
    out->slot_version = host_slot->version;
    out->completion = host_slot->completion;
    out->output_hash = hash_token(host_slot->token);
    out->resident_weight_bytes = (hidden + vocab_size * hidden) * sizeof(uint16_t);
    out->device_arena_bytes = arena_bytes + scratch_bytes + sizeof(*device_slot);
    out->pinned_host_bytes = arena_bytes + sizeof(*host_slot);
    out->status = host_slot->request_id == kRequestId &&
                          host_slot->sequence_id == kSequenceId &&
                          host_slot->completion == kCompletionDeviceComplete
                      ? 0
                      : -1;
  } else {
    fail(out, err);
  }

  cudaStreamDestroy(stream);
  cudaFree(device_slot);
  cudaFree(device_scratch);
  cudaFree(device_arena);
  cudaFreeHost(host_slot);
  cudaFreeHost(host_arena);
  return out->status == 0 ? 0 : -1;
}
