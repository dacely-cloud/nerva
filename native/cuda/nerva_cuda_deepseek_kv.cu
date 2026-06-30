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

__global__ void compressed_slot_mapping_kernel(
    int64_t *output_slots,
    const int32_t *query_start_loc,
    const int32_t *seq_lens,
    const int32_t *block_table,
    uint32_t num_tokens,
    uint32_t num_reqs,
    uint32_t block_table_stride,
    uint32_t block_size,
    uint32_t compress_ratio) {
  const uint32_t req_idx = blockIdx.x;
  if (req_idx >= num_reqs) {
    return;
  }

  const int32_t query_start = query_start_loc[req_idx];
  const int32_t query_end = query_start_loc[req_idx + 1];
  if (query_start < 0 || query_end < query_start) {
    return;
  }

  const uint32_t query_len = static_cast<uint32_t>(query_end - query_start);
  const int32_t start_pos = seq_lens[req_idx] - static_cast<int32_t>(query_len);
  for (uint32_t offset = threadIdx.x; offset < query_len; offset += blockDim.x) {
    const uint32_t output_idx = static_cast<uint32_t>(query_start) + offset;
    if (output_idx >= num_tokens) {
      continue;
    }

    int64_t slot = -1;
    const int32_t pos = start_pos + static_cast<int32_t>(offset);
    if (pos >= 0 && ((pos + 1) % static_cast<int32_t>(compress_ratio)) == 0) {
      const int32_t compressed_pos =
          pos / static_cast<int32_t>(compress_ratio);
      const int32_t block_id =
          compressed_pos / static_cast<int32_t>(block_size);
      const int32_t block_offset =
          compressed_pos % static_cast<int32_t>(block_size);
      if (block_id >= 0 &&
          block_id < static_cast<int32_t>(block_table_stride)) {
        const int32_t block_number =
            block_table[req_idx * block_table_stride + block_id];
        if (block_number >= 0) {
          slot = static_cast<int64_t>(block_number) * block_size + block_offset;
        }
      }
    }
    output_slots[output_idx] = slot;
  }
}

__global__ void c128_topk_metadata_kernel(
    int32_t *global_decode,
    int32_t *decode_lens,
    int32_t *prefill_local,
    const int64_t *positions,
    const int32_t *token_to_req_indices,
    const int32_t *block_table,
    const int64_t *slot_mapping,
    uint32_t num_tokens,
    uint32_t num_decode_tokens,
    uint32_t num_reqs,
    uint32_t block_table_stride,
    uint32_t block_size,
    uint32_t compress_ratio,
    uint32_t max_compressed_tokens) {
  const uint32_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  const int64_t position = positions[token_idx];
  uint32_t num_compressed = 0;
  if (position >= 0) {
    const uint64_t raw =
        (static_cast<uint64_t>(position) + 1ull) / compress_ratio;
    num_compressed = raw > max_compressed_tokens
                         ? max_compressed_tokens
                         : static_cast<uint32_t>(raw);
  }

  if (token_idx < num_decode_tokens) {
    const bool valid_token = slot_mapping[token_idx] >= 0;
    int32_t local_count = 0;
    const int32_t req_idx = token_to_req_indices[token_idx];
    for (uint32_t offset = threadIdx.x; offset < max_compressed_tokens;
         offset += blockDim.x) {
      int32_t slot = -1;
      const bool is_valid = offset < num_compressed;
      if (is_valid && req_idx >= 0 &&
          req_idx < static_cast<int32_t>(num_reqs)) {
        const uint32_t block_id = offset / block_size;
        const uint32_t block_offset = offset % block_size;
        if (block_id < block_table_stride) {
          const int32_t block_number =
              block_table[static_cast<uint32_t>(req_idx) * block_table_stride +
                          block_id];
          if (block_number >= 0) {
            slot = block_number * static_cast<int32_t>(block_size) +
                   static_cast<int32_t>(block_offset);
          }
        }
      }
      global_decode[token_idx * max_compressed_tokens + offset] = slot;
      local_count += is_valid ? 1 : 0;
    }

    __shared__ int32_t counts[256];
    counts[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        counts[threadIdx.x] += counts[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      decode_lens[token_idx] = valid_token ? counts[0] : 0;
    }
  } else {
    const uint32_t prefill_idx = token_idx - num_decode_tokens;
    for (uint32_t offset = threadIdx.x; offset < max_compressed_tokens;
         offset += blockDim.x) {
      prefill_local[prefill_idx * max_compressed_tokens + offset] =
          offset < num_compressed ? static_cast<int32_t>(offset) : -1;
    }
  }
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

void clear_result(NervaCudaDeepSeekCompressedSlotMappingResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

void clear_result(NervaCudaDeepSeekC128TopkMetadataResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

int fail(NervaCudaDeepSeekKvFp8DsMlaPackResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

int fail(NervaCudaDeepSeekCompressedSlotMappingResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

int fail(NervaCudaDeepSeekC128TopkMetadataResult *out, cudaError_t err) {
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

bool validate_request(
    const NervaCudaDeepSeekCompressedSlotMappingRequest *request) {
  return request != nullptr && request->query_start_loc != nullptr &&
         request->seq_lens != nullptr && request->block_table != nullptr &&
         request->output_slots != nullptr && request->num_tokens > 0 &&
         request->num_reqs > 0 && request->block_table_stride > 0 &&
         request->block_size > 0 && request->compress_ratio > 0;
}

bool validate_request(const NervaCudaDeepSeekC128TopkMetadataRequest *request) {
  return request != nullptr && request->positions != nullptr &&
         request->token_to_req_indices != nullptr &&
         request->block_table != nullptr && request->slot_mapping != nullptr &&
         request->global_decode != nullptr && request->decode_lens != nullptr &&
         request->prefill_local != nullptr && request->num_tokens > 0 &&
         request->num_decode_tokens <= request->num_tokens &&
         request->num_reqs > 0 && request->block_table_stride > 0 &&
         request->block_size > 0 && request->compress_ratio > 0 &&
         request->max_compressed_tokens > 0;
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

extern "C" int nerva_cuda_deepseek_compressed_slot_mapping(
    const NervaCudaDeepSeekCompressedSlotMappingRequest *request,
    NervaCudaDeepSeekCompressedSlotMappingResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  if (!validate_request(request)) {
    return -1;
  }

  out->num_tokens = request->num_tokens;
  out->num_reqs = request->num_reqs;
  out->block_table_stride = request->block_table_stride;
  out->block_size = request->block_size;
  out->compress_ratio = request->compress_ratio;

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

  int32_t *d_query_start_loc = nullptr;
  int32_t *d_seq_lens = nullptr;
  int32_t *d_block_table = nullptr;
  int64_t *d_output_slots = nullptr;
  int64_t *h_output_slots = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t query_bytes =
      static_cast<uint64_t>(request->num_reqs + 1) * sizeof(int32_t);
  const uint64_t seq_bytes =
      static_cast<uint64_t>(request->num_reqs) * sizeof(int32_t);
  const uint64_t table_values =
      static_cast<uint64_t>(request->num_reqs) * request->block_table_stride;
  const uint64_t table_bytes = table_values * sizeof(int32_t);
  const uint64_t output_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int64_t);

  err = cudaMalloc(reinterpret_cast<void **>(&d_query_start_loc), query_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_seq_lens), seq_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_block_table), table_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output_slots), output_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      query_bytes + seq_bytes + table_bytes + output_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output_slots),
                      output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(d_query_start_loc,
                        request->query_start_loc,
                        query_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(
      d_seq_lens, request->seq_lens, seq_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_block_table,
                        request->block_table,
                        table_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = query_bytes + seq_bytes + table_bytes;

  err = cudaMemsetAsync(d_output_slots, 0xff, output_bytes, stream);
  if (err != cudaSuccess) goto cleanup;

  {
    constexpr uint32_t threads = 256;
    compressed_slot_mapping_kernel<<<request->num_reqs, threads, 0, stream>>>(
        d_output_slots,
        d_query_start_loc,
        d_seq_lens,
        d_block_table,
        request->num_tokens,
        request->num_reqs,
        request->block_table_stride,
        request->block_size,
        request->compress_ratio);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_output_slots,
                        d_output_slots,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->output_slots, h_output_slots, output_bytes);
  for (uint32_t idx = 0; idx < request->num_tokens; ++idx) {
    if (request->output_slots[idx] >= 0) {
      out->valid_slots += 1;
    } else {
      out->pad_slots += 1;
    }
  }
  out->output_hash = hash_bytes(
      reinterpret_cast<const uint8_t *>(request->output_slots), output_bytes);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output_slots != nullptr) cudaFreeHost(h_output_slots);
  if (d_output_slots != nullptr) cudaFree(d_output_slots);
  if (d_block_table != nullptr) cudaFree(d_block_table);
  if (d_seq_lens != nullptr) cudaFree(d_seq_lens);
  if (d_query_start_loc != nullptr) cudaFree(d_query_start_loc);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_c128_topk_metadata(
    const NervaCudaDeepSeekC128TopkMetadataRequest *request,
    NervaCudaDeepSeekC128TopkMetadataResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  if (!validate_request(request)) {
    return -1;
  }

  out->num_tokens = request->num_tokens;
  out->num_decode_tokens = request->num_decode_tokens;
  out->num_prefill_tokens = request->num_tokens - request->num_decode_tokens;
  out->num_reqs = request->num_reqs;
  out->block_table_stride = request->block_table_stride;
  out->block_size = request->block_size;
  out->compress_ratio = request->compress_ratio;
  out->max_compressed_tokens = request->max_compressed_tokens;

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

  int64_t *d_positions = nullptr;
  int32_t *d_token_to_req = nullptr;
  int32_t *d_block_table = nullptr;
  int64_t *d_slot_mapping = nullptr;
  int32_t *d_global_decode = nullptr;
  int32_t *d_decode_lens = nullptr;
  int32_t *d_prefill_local = nullptr;
  int32_t *h_global_decode = nullptr;
  int32_t *h_decode_lens = nullptr;
  int32_t *h_prefill_local = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t positions_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int64_t);
  const uint64_t token_to_req_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int32_t);
  const uint64_t table_bytes =
      static_cast<uint64_t>(request->num_reqs) *
      request->block_table_stride * sizeof(int32_t);
  const uint64_t slot_mapping_bytes =
      static_cast<uint64_t>(request->num_tokens) * sizeof(int64_t);
  const uint64_t global_decode_values =
      static_cast<uint64_t>(request->num_decode_tokens) *
      request->max_compressed_tokens;
  const uint64_t prefill_values =
      static_cast<uint64_t>(out->num_prefill_tokens) *
      request->max_compressed_tokens;
  const uint64_t global_decode_bytes =
      global_decode_values * sizeof(int32_t);
  const uint64_t decode_lens_bytes =
      static_cast<uint64_t>(request->num_decode_tokens) * sizeof(int32_t);
  const uint64_t prefill_bytes = prefill_values * sizeof(int32_t);

  err = cudaMalloc(reinterpret_cast<void **>(&d_positions), positions_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_token_to_req),
                   token_to_req_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_block_table), table_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_slot_mapping),
                   slot_mapping_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_global_decode),
                   global_decode_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_decode_lens),
                   decode_lens_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_prefill_local), prefill_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes = positions_bytes + token_to_req_bytes + table_bytes +
                            slot_mapping_bytes + global_decode_bytes +
                            decode_lens_bytes + prefill_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_global_decode),
                      global_decode_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_decode_lens),
                      decode_lens_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_prefill_local),
                      prefill_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes =
      global_decode_bytes + decode_lens_bytes + prefill_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(d_positions,
                        request->positions,
                        positions_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_token_to_req,
                        request->token_to_req_indices,
                        token_to_req_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_block_table,
                        request->block_table,
                        table_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_slot_mapping,
                        request->slot_mapping,
                        slot_mapping_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes =
      positions_bytes + token_to_req_bytes + table_bytes + slot_mapping_bytes;

  err = cudaMemsetAsync(d_global_decode, 0xff, global_decode_bytes, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemsetAsync(d_decode_lens, 0, decode_lens_bytes, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemsetAsync(d_prefill_local, 0xff, prefill_bytes, stream);
  if (err != cudaSuccess) goto cleanup;

  {
    constexpr uint32_t threads = 256;
    c128_topk_metadata_kernel<<<request->num_tokens, threads, 0, stream>>>(
        d_global_decode,
        d_decode_lens,
        d_prefill_local,
        d_positions,
        d_token_to_req,
        d_block_table,
        d_slot_mapping,
        request->num_tokens,
        request->num_decode_tokens,
        request->num_reqs,
        request->block_table_stride,
        request->block_size,
        request->compress_ratio,
        request->max_compressed_tokens);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_global_decode,
                        d_global_decode,
                        global_decode_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_decode_lens,
                        d_decode_lens,
                        decode_lens_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_prefill_local,
                        d_prefill_local,
                        prefill_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = global_decode_bytes + decode_lens_bytes + prefill_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->global_decode, h_global_decode, global_decode_bytes);
  memcpy(request->decode_lens, h_decode_lens, decode_lens_bytes);
  memcpy(request->prefill_local, h_prefill_local, prefill_bytes);
  for (uint32_t idx = 0; idx < request->num_decode_tokens; ++idx) {
    if (request->slot_mapping[idx] >= 0) {
      out->valid_decode_tokens += 1;
    }
    out->decode_entries += request->decode_lens[idx];
  }
  for (uint64_t idx = 0; idx < prefill_values; ++idx) {
    if (request->prefill_local[idx] >= 0) {
      out->prefill_entries += 1;
    }
  }
  out->output_hash = hash_bytes(
      reinterpret_cast<const uint8_t *>(request->global_decode),
      global_decode_bytes);
  out->output_hash ^= hash_bytes(
      reinterpret_cast<const uint8_t *>(request->decode_lens),
      decode_lens_bytes);
  out->output_hash *= 1099511628211ull;
  out->output_hash ^= hash_bytes(
      reinterpret_cast<const uint8_t *>(request->prefill_local),
      prefill_bytes);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_prefill_local != nullptr) cudaFreeHost(h_prefill_local);
  if (h_decode_lens != nullptr) cudaFreeHost(h_decode_lens);
  if (h_global_decode != nullptr) cudaFreeHost(h_global_decode);
  if (d_prefill_local != nullptr) cudaFree(d_prefill_local);
  if (d_decode_lens != nullptr) cudaFree(d_decode_lens);
  if (d_global_decode != nullptr) cudaFree(d_global_decode);
  if (d_slot_mapping != nullptr) cudaFree(d_slot_mapping);
  if (d_block_table != nullptr) cudaFree(d_block_table);
  if (d_token_to_req != nullptr) cudaFree(d_token_to_req);
  if (d_positions != nullptr) cudaFree(d_positions);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}
