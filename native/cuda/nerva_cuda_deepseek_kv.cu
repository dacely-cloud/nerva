#include "nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

namespace {

__global__ void fp8_ds_mla_pack_kernel(const uint8_t *nope_fp8,
                                       const uint16_t *rope_bf16,
                                       const uint8_t *scales,
                                       uint8_t *output_block,
                                       uint32_t block_size,
                                       uint32_t token_index,
                                       uint32_t nope_bytes,
                                       uint32_t rope_bf16_values,
                                       uint32_t scale_dim,
                                       uint32_t token_stride) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t rope_bytes = rope_bf16_values * 2;
  const uint32_t data_bytes = nope_bytes + rope_bytes;
  const uint32_t total_bytes = data_bytes + scale_dim;
  if (idx >= total_bytes) {
    return;
  }

  const uint64_t token_base =
      static_cast<uint64_t>(token_index) * token_stride;
  const uint64_t scale_base =
      static_cast<uint64_t>(block_size) * token_stride +
      static_cast<uint64_t>(token_index) * scale_dim;

  if (idx < nope_bytes) {
    output_block[token_base + idx] = nope_fp8[idx];
    return;
  }
  if (idx < data_bytes) {
    const uint32_t rope_byte = idx - nope_bytes;
    const uint16_t value = rope_bf16[rope_byte / 2];
    output_block[token_base + idx] =
        static_cast<uint8_t>((value >> ((rope_byte & 1u) * 8u)) & 0xffu);
    return;
  }

  const uint32_t scale_idx = idx - data_bytes;
  output_block[scale_base + scale_idx] = scales[scale_idx];
}

uint64_t hash_bytes(const uint8_t *values, uint64_t len) {
  uint64_t hash = 1469598103934665603ull;
  for (uint64_t i = 0; i < len; ++i) {
    hash ^= values[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

void clear_result(NervaCudaDeepSeekKvFp8DsMlaPackResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

int fail(NervaCudaDeepSeekKvFp8DsMlaPackResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

bool validate_request(const NervaCudaDeepSeekKvFp8DsMlaPackRequest *request) {
  return request != nullptr && request->nope_fp8 != nullptr &&
         request->rope_bf16 != nullptr && request->scales != nullptr &&
         request->output_block != nullptr && request->block_size > 0 &&
         request->token_index < request->block_size && request->nope_bytes > 0 &&
         request->rope_bf16_values > 0 && request->scale_dim > 0;
}

}  // namespace

extern "C" int nerva_cuda_deepseek_kv_fp8_ds_mla_pack(
    const NervaCudaDeepSeekKvFp8DsMlaPackRequest *request,
    NervaCudaDeepSeekKvFp8DsMlaPackResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  if (!validate_request(request)) {
    return -1;
  }

  const uint64_t rope_bytes = static_cast<uint64_t>(request->rope_bf16_values) * 2;
  const uint64_t token_stride =
      static_cast<uint64_t>(request->nope_bytes) + rope_bytes;
  const uint64_t block_bytes =
      static_cast<uint64_t>(request->block_size) *
      (token_stride + request->scale_dim);
  if (token_stride > UINT32_MAX || block_bytes == 0) {
    return -1;
  }

  out->block_size = request->block_size;
  out->token_index = request->token_index;
  out->token_stride = static_cast<uint32_t>(token_stride);
  out->scale_dim = request->scale_dim;
  out->block_bytes = block_bytes;

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

  uint8_t *d_nope = nullptr;
  uint16_t *d_rope = nullptr;
  uint8_t *d_scales = nullptr;
  uint8_t *d_output = nullptr;
  uint8_t *h_output = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t nope_bytes = request->nope_bytes;
  const uint64_t rope_input_bytes =
      static_cast<uint64_t>(request->rope_bf16_values) * sizeof(uint16_t);
  const uint64_t scale_bytes = request->scale_dim;

  err = cudaMalloc(reinterpret_cast<void **>(&d_nope), nope_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_rope), rope_input_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_scales), scale_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output), block_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      nope_bytes + rope_input_bytes + scale_bytes + block_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output),
                      block_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = block_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(
      d_nope, request->nope_fp8, nope_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_rope,
                        request->rope_bf16,
                        rope_input_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(
      d_scales, request->scales, scale_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = nope_bytes + rope_input_bytes + scale_bytes;

  err = cudaMemsetAsync(d_output, 0, block_bytes, stream);
  if (err != cudaSuccess) goto cleanup;

  {
    constexpr uint32_t threads = 256;
    const uint32_t copy_bytes =
        request->nope_bytes + request->rope_bf16_values * 2 +
        request->scale_dim;
    const uint32_t blocks = (copy_bytes + threads - 1) / threads;
    fp8_ds_mla_pack_kernel<<<blocks, threads, 0, stream>>>(
        d_nope,
        d_rope,
        d_scales,
        d_output,
        request->block_size,
        request->token_index,
        request->nope_bytes,
        request->rope_bf16_values,
        request->scale_dim,
        out->token_stride);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(
      h_output, d_output, block_bytes, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = block_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->output_block, h_output, block_bytes);
  out->output_hash = hash_bytes(request->output_block, block_bytes);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output != nullptr) cudaFreeHost(h_output);
  if (d_output != nullptr) cudaFree(d_output);
  if (d_scales != nullptr) cudaFree(d_scales);
  if (d_rope != nullptr) cudaFree(d_rope);
  if (d_nope != nullptr) cudaFree(d_nope);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}
