#include "nerva_cuda_api.h"
#include "deepseek_quant.cuh"
#include "hf_decode_sequence/projection.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kFp8Rows = 3;
constexpr uint32_t kFp8Cols = 4;
constexpr uint32_t kFp8BlockRows = 2;
constexpr uint32_t kFp8BlockCols = 2;
constexpr uint32_t kFp8Values = kFp8Rows * kFp8Cols;
constexpr uint32_t kFp8Scales = 4;

constexpr uint32_t kMxfp4Rows = 2;
constexpr uint32_t kMxfp4PackedCols = 4;
constexpr uint32_t kMxfp4ScalePackedCols = 2;
constexpr uint32_t kMxfp4PackedValues = kMxfp4Rows * kMxfp4PackedCols;
constexpr uint32_t kMxfp4Values = kMxfp4PackedValues * 2;
constexpr uint32_t kMxfp4Scales = 4;

constexpr uint8_t kFp8Weights[kFp8Values] = {
    0x38, 0x40, 0x30, 0xb8,
    0x70, 0x77, 0x78, 0x7e,
    0x20, 0x28, 0x30, 0x38,
};
constexpr uint8_t kFp8ScaleBytes[kFp8Scales] = {
    0x7f, 0x80,
    0x7e, 0x81,
};
constexpr float kFp8Expected[kFp8Values] = {
    1.0f, 2.0f, 1.0f, -2.0f,
    128.0f, 240.0f, 512.0f, 896.0f,
    0.0625f, 0.125f, 2.0f, 4.0f,
};

constexpr uint8_t kMxfp4Packed[kMxfp4PackedValues] = {
    0x21, 0x76, 0xa9, 0xfe,
    0x10, 0x54, 0x98, 0xdc,
};
constexpr uint8_t kMxfp4ScaleBytes[kMxfp4Scales] = {
    0x7f, 0x80,
    0x7e, 0x81,
};
constexpr float kMxfp4Expected[kMxfp4Values] = {
    0.5f, 1.0f, 4.0f, 6.0f, -1.0f, -2.0f, -8.0f, -12.0f,
    0.0f, 0.25f, 1.0f, 1.5f, -0.0f, -2.0f, -8.0f, -12.0f,
};

__global__ void fp8_e4m3fn_block_dequant_kernel(
    const uint8_t *weights,
    const uint8_t *scales,
    float *output) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= kFp8Values) {
    return;
  }

  const uint32_t row = idx / kFp8Cols;
  const uint32_t col = idx - row * kFp8Cols;
  const uint32_t scale_cols = (kFp8Cols + kFp8BlockCols - 1) / kFp8BlockCols;
  const uint32_t scale_idx =
      (row / kFp8BlockRows) * scale_cols + (col / kFp8BlockCols);
  output[idx] =
      nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[idx]) *
      nerva::deepseek::e8m0_exponent_bits_to_f32(scales[scale_idx]);
}

__global__ void mxfp4_e2m1_block_dequant_kernel(
    const uint8_t *packed,
    const uint8_t *scales,
    float *output) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= kMxfp4PackedValues) {
    return;
  }

  const uint32_t row = idx / kMxfp4PackedCols;
  const uint32_t packed_col = idx - row * kMxfp4PackedCols;
  const uint32_t scale_cols =
      (kMxfp4PackedCols + kMxfp4ScalePackedCols - 1) /
      kMxfp4ScalePackedCols;
  const uint32_t scale_idx =
      row * scale_cols + packed_col / kMxfp4ScalePackedCols;
  const float scale =
      nerva::deepseek::e8m0_exponent_bits_to_f32(scales[scale_idx]);
  const uint8_t byte = packed[idx];
  const uint32_t out_idx = idx * 2;
  output[out_idx] =
      nerva::deepseek::mxfp4_e2m1_nibble_to_f32(byte & 0x0fu) * scale;
  output[out_idx + 1] =
      nerva::deepseek::mxfp4_e2m1_nibble_to_f32(byte >> 4) * scale;
}

__global__ void fp8_e4m3fn_block_dequant_dynamic_kernel(
    const uint8_t *weights,
    const uint8_t *scales,
    float *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t block_rows,
    uint32_t block_cols) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t values = rows * cols;
  if (idx >= values) {
    return;
  }
  const uint32_t row = idx / cols;
  const uint32_t col = idx - row * cols;
  const uint32_t scale_cols = (cols + block_cols - 1) / block_cols;
  const uint32_t scale_idx =
      (row / block_rows) * scale_cols + (col / block_cols);
  output[idx] =
      nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[idx]) *
      nerva::deepseek::e8m0_exponent_bits_to_f32(scales[scale_idx]);
}

__global__ void mxfp4_e2m1_block_dequant_dynamic_kernel(
    const uint8_t *packed,
    const uint8_t *scales,
    float *output,
    uint32_t rows,
    uint32_t packed_cols,
    uint32_t scale_packed_cols) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t values = rows * packed_cols;
  if (idx >= values) {
    return;
  }
  const uint32_t row = idx / packed_cols;
  const uint32_t packed_col = idx - row * packed_cols;
  const uint32_t scale_cols =
      (packed_cols + scale_packed_cols - 1) / scale_packed_cols;
  const uint32_t scale_idx = row * scale_cols + packed_col / scale_packed_cols;
  const float scale =
      nerva::deepseek::e8m0_exponent_bits_to_f32(scales[scale_idx]);
  const uint8_t byte = packed[idx];
  const uint32_t out_idx = idx * 2;
  output[out_idx] =
      nerva::deepseek::mxfp4_e2m1_nibble_to_f32(byte & 0x0fu) * scale;
  output[out_idx + 1] =
      nerva::deepseek::mxfp4_e2m1_nibble_to_f32(byte >> 4) * scale;
}

__device__ uint8_t f32_to_f8_e4m3fn_bits_nearest(float value) {
  return nerva::deepseek::f32_to_f8_e4m3fn_bits(value);
}

__global__ void fused_inv_rope_fp8_quant_kernel(
    const float *input,
    const int64_t *positions,
    const float *cos_sin_cache,
    uint8_t *fp8_output,
    float *scale_output,
    uint32_t *packed_scale_output,
    uint32_t num_tokens,
    uint32_t heads_per_group,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t quant_group_size,
    uint32_t cos_sin_stride,
    uint32_t scale_blocks,
    float fp8_max,
    float eps) {
  const uint32_t token_idx = blockIdx.x;
  const uint32_t global_head = blockIdx.y;
  if (token_idx >= num_tokens) {
    return;
  }

  const uint32_t chunks_per_head = head_dim / quant_group_size;
  const uint32_t nope_dim = head_dim - rope_dim;
  const uint32_t rope_start = nope_dim % quant_group_size;
  const uint32_t rope_abs_start =
      (chunks_per_head - 1) * quant_group_size + rope_start;
  const uint32_t half_rope = rope_dim / 2;
  const uint32_t group_idx = global_head / heads_per_group;
  const uint32_t head_in_group = global_head - group_idx * heads_per_group;
  const uint32_t num_heads = gridDim.y;
  const uint32_t n_groups = gridDim.y / heads_per_group;
  const uint64_t input_base =
      (static_cast<uint64_t>(token_idx) * num_heads + global_head) * head_dim;
  const uint32_t d = heads_per_group * head_dim;
  const uint64_t fp8_base =
      (static_cast<uint64_t>(group_idx) * num_tokens + token_idx) * d +
      static_cast<uint64_t>(head_in_group) * head_dim;
  const uint64_t scale_base =
      (static_cast<uint64_t>(group_idx) * num_tokens + token_idx) *
      scale_blocks;
  const uint64_t packed_base =
      (static_cast<uint64_t>(group_idx) * num_tokens + token_idx) *
      heads_per_group;
  const int64_t position = positions[token_idx];
  const uint64_t cache_base =
      static_cast<uint64_t>(position < 0 ? 0 : position) * cos_sin_stride;

  for (uint32_t chunk = 0; chunk < chunks_per_head; ++chunk) {
    float block_absmax = 0.0f;
    const uint32_t chunk_start = chunk * quant_group_size;
    for (uint32_t offset = 0; offset < quant_group_size; ++offset) {
      const uint32_t dim = chunk_start + offset;
      float x = input[input_base + dim];
      if (dim >= rope_abs_start && rope_dim > 0) {
        const uint32_t rope_local = dim - rope_abs_start;
        const uint32_t partner_dim = dim ^ 1u;
        const float partner = input[input_base + partner_dim];
        const uint32_t cs_idx = rope_local >> 1;
        const float cos_v = cos_sin_cache[cache_base + cs_idx];
        const float sin_v = cos_sin_cache[cache_base + half_rope + cs_idx];
        const bool is_even = (rope_local & 1u) == 0u;
        x = is_even ? x * cos_v + partner * sin_v
                    : x * cos_v - partner * sin_v;
      }
      block_absmax = fmaxf(block_absmax, fabsf(x));
    }

    const float scale_raw = fmaxf(block_absmax, eps) / fp8_max;
    const float scale = exp2f(ceilf(log2f(scale_raw)));
    const uint32_t scale_idx = head_in_group * chunks_per_head + chunk;
    scale_output[scale_base + scale_idx] = scale;

    const uint32_t scale_bits = __float_as_uint(scale);
    const uint32_t scale_byte = (scale_bits >> 23) & 0xffu;
    if (chunk == 0) {
      packed_scale_output[packed_base + head_in_group] = 0;
    }
    atomicOr(&packed_scale_output[packed_base + head_in_group],
             scale_byte << (chunk * 8u));

    for (uint32_t offset = 0; offset < quant_group_size; ++offset) {
      const uint32_t dim = chunk_start + offset;
      float x = input[input_base + dim];
      if (dim >= rope_abs_start && rope_dim > 0) {
        const uint32_t rope_local = dim - rope_abs_start;
        const uint32_t partner_dim = dim ^ 1u;
        const float partner = input[input_base + partner_dim];
        const uint32_t cs_idx = rope_local >> 1;
        const float cos_v = cos_sin_cache[cache_base + cs_idx];
        const float sin_v = cos_sin_cache[cache_base + half_rope + cs_idx];
        const bool is_even = (rope_local & 1u) == 0u;
        x = is_even ? x * cos_v + partner * sin_v
                    : x * cos_v - partner * sin_v;
      }
      const float scaled = fminf(fmaxf(x / scale, -fp8_max), fp8_max);
      fp8_output[fp8_base + dim] = f32_to_f8_e4m3fn_bits_nearest(scaled);
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
    uint32_t bits = f32_bits(values[i]);
    for (uint32_t byte = 0; byte < 4; ++byte) {
      hash ^= static_cast<uint8_t>((bits >> (byte * 8)) & 0xffu);
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

uint64_t hash_bytes(const uint8_t *values, uint64_t len) {
  uint64_t hash = 1469598103934665603ull;
  for (uint64_t i = 0; i < len; ++i) {
    hash ^= values[i];
    hash *= 1099511628211ull;
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
    if (f32_bits(actual[i]) != f32_bits(expected[i])) {
      *mismatches += 1;
    }
  }
}

void clear_result(NervaCudaDeepSeekQuantSmokeResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->fp8_rows = kFp8Rows;
  out->fp8_cols = kFp8Cols;
  out->fp8_block_rows = kFp8BlockRows;
  out->fp8_block_cols = kFp8BlockCols;
  out->mxfp4_rows = kMxfp4Rows;
  out->mxfp4_packed_cols = kMxfp4PackedCols;
  out->mxfp4_scale_packed_cols = kMxfp4ScalePackedCols;
}

int fail(NervaCudaDeepSeekQuantSmokeResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

void clear_dequant_result(NervaCudaDeepSeekQuantDequantResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

int fail_dequant(NervaCudaDeepSeekQuantDequantResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

void clear_inv_rope_result(NervaCudaDeepSeekFusedInvRopeFp8QuantResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

int fail_inv_rope(NervaCudaDeepSeekFusedInvRopeFp8QuantResult *out,
                  cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

bool validate_fp8_request(const NervaCudaDeepSeekQuantFp8DequantRequest *request) {
  return request != nullptr && request->weights != nullptr &&
         request->scales != nullptr && request->output != nullptr &&
         request->rows > 0 && request->cols > 0 && request->block_rows > 0 &&
         request->block_cols > 0;
}

bool validate_fp8_matvec_request(
    const NervaCudaDeepSeekQuantFp8F32ScaleMatvecRequest *request) {
  return request != nullptr && request->weights != nullptr &&
         request->scales != nullptr && request->input != nullptr &&
         request->output != nullptr && request->rows > 0 &&
         request->cols > 0 && request->block_rows > 0 &&
         request->block_cols > 0;
}

bool validate_fp8_encoded_matvec_request(
    const NervaCudaDeepSeekQuantFp8F32ScaleEncodedMatvecRequest *request) {
  return request != nullptr && request->weights != nullptr &&
         request->scales != nullptr && request->input != nullptr &&
         request->output != nullptr && request->rows > 0 &&
         request->cols > 0 && request->block_rows > 0 &&
         request->block_cols > 0 && request->input_dtype <= 1;
}

bool validate_fp8_encoded_gemm_tokens_request(
    const NervaCudaDeepSeekQuantFp8F32ScaleEncodedGemmTokensRequest *request) {
  return request != nullptr && request->weights != nullptr &&
         request->scales != nullptr && request->input != nullptr &&
         request->output != nullptr && request->rows > 0 &&
         request->cols > 0 && request->tokens > 0 &&
         request->block_rows > 0 && request->block_cols > 0 &&
         request->input_dtype <= 1;
}

bool validate_fp8_e8m0_encoded_gemm_tokens_request(
    const NervaCudaDeepSeekQuantFp8E8m0ScaleEncodedGemmTokensRequest *request) {
  return request != nullptr && request->weights != nullptr &&
         request->scales != nullptr && request->input != nullptr &&
         request->output != nullptr && request->rows > 0 &&
         request->cols > 0 && request->tokens > 0 &&
         request->block_rows > 0 && request->block_cols > 0 &&
         request->input_dtype <= 1;
}

bool validate_mxfp4_request(
    const NervaCudaDeepSeekQuantMxfp4DequantRequest *request) {
  return request != nullptr && request->packed != nullptr &&
         request->scales != nullptr && request->output != nullptr &&
         request->rows > 0 && request->packed_cols > 0 &&
         request->scale_packed_cols > 0;
}

bool validate_inv_rope_request(
    const NervaCudaDeepSeekFusedInvRopeFp8QuantRequest *request) {
  if (request == nullptr || request->input == nullptr ||
      request->positions == nullptr || request->cos_sin_cache == nullptr ||
      request->fp8_output == nullptr || request->scale_output == nullptr ||
      request->packed_scale_output == nullptr || request->num_tokens == 0 ||
      request->n_groups == 0 || request->heads_per_group == 0 ||
      request->head_dim == 0 || request->quant_group_size == 0 ||
      request->cos_sin_stride == 0 || !isfinite(request->fp8_max) ||
      !isfinite(request->eps) || request->fp8_max <= 0.0f ||
      request->eps <= 0.0f) {
    return false;
  }
  const uint32_t chunks = request->head_dim / request->quant_group_size;
  return request->head_dim % request->quant_group_size == 0 &&
         request->rope_dim <= request->head_dim && (request->rope_dim % 2) == 0 &&
         request->cos_sin_stride >= request->rope_dim && chunks > 0 &&
         chunks <= 4;
}

}  // namespace

extern "C" int nerva_cuda_deepseek_quant_smoke(
    NervaCudaDeepSeekQuantSmokeResult *out) {
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

  uint8_t *d_fp8_weights = nullptr;
  uint8_t *d_fp8_scales = nullptr;
  float *d_fp8_output = nullptr;
  uint8_t *d_mxfp4_packed = nullptr;
  uint8_t *d_mxfp4_scales = nullptr;
  float *d_mxfp4_output = nullptr;
  float *h_fp8_output = nullptr;
  float *h_mxfp4_output = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t fp8_weight_bytes = sizeof(kFp8Weights);
  const uint64_t fp8_scale_bytes = sizeof(kFp8ScaleBytes);
  const uint64_t fp8_output_bytes = sizeof(float) * kFp8Values;
  const uint64_t mxfp4_packed_bytes = sizeof(kMxfp4Packed);
  const uint64_t mxfp4_scale_bytes = sizeof(kMxfp4ScaleBytes);
  const uint64_t mxfp4_output_bytes = sizeof(float) * kMxfp4Values;

  err = cudaMalloc(reinterpret_cast<void **>(&d_fp8_weights), fp8_weight_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_fp8_scales), fp8_scale_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_fp8_output), fp8_output_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_mxfp4_packed), mxfp4_packed_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_mxfp4_scales), mxfp4_scale_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_mxfp4_output), mxfp4_output_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      fp8_weight_bytes + fp8_scale_bytes + fp8_output_bytes +
      mxfp4_packed_bytes + mxfp4_scale_bytes + mxfp4_output_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_fp8_output),
                      fp8_output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_mxfp4_output),
                      mxfp4_output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = fp8_output_bytes + mxfp4_output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(d_fp8_weights,
                        kFp8Weights,
                        fp8_weight_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_fp8_scales,
                        kFp8ScaleBytes,
                        fp8_scale_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_mxfp4_packed,
                        kMxfp4Packed,
                        mxfp4_packed_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_mxfp4_scales,
                        kMxfp4ScaleBytes,
                        mxfp4_scale_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = fp8_weight_bytes + fp8_scale_bytes +
                   mxfp4_packed_bytes + mxfp4_scale_bytes;

  fp8_e4m3fn_block_dequant_kernel<<<1, 32, 0, stream>>>(
      d_fp8_weights,
      d_fp8_scales,
      d_fp8_output);
  out->kernel_launches += 1;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  mxfp4_e2m1_block_dequant_kernel<<<1, 32, 0, stream>>>(
      d_mxfp4_packed,
      d_mxfp4_scales,
      d_mxfp4_output);
  out->kernel_launches += 1;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_fp8_output,
                        d_fp8_output,
                        fp8_output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_mxfp4_output,
                        d_mxfp4_output,
                        mxfp4_output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = fp8_output_bytes + mxfp4_output_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  out->fp8_output_hash = hash_f32_bits(h_fp8_output, kFp8Values);
  out->mxfp4_output_hash = hash_f32_bits(h_mxfp4_output, kMxfp4Values);
  compare_outputs(h_fp8_output,
                  kFp8Expected,
                  kFp8Values,
                  &out->fp8_mismatches,
                  &out->fp8_max_abs_diff);
  compare_outputs(h_mxfp4_output,
                  kMxfp4Expected,
                  kMxfp4Values,
                  &out->mxfp4_mismatches,
                  &out->mxfp4_max_abs_diff);

  out->status =
      (out->fp8_mismatches == 0 && out->mxfp4_mismatches == 0) ? 0 : -1;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_mxfp4_output != nullptr) cudaFreeHost(h_mxfp4_output);
  if (h_fp8_output != nullptr) cudaFreeHost(h_fp8_output);
  if (d_mxfp4_output != nullptr) cudaFree(d_mxfp4_output);
  if (d_mxfp4_scales != nullptr) cudaFree(d_mxfp4_scales);
  if (d_mxfp4_packed != nullptr) cudaFree(d_mxfp4_packed);
  if (d_fp8_output != nullptr) cudaFree(d_fp8_output);
  if (d_fp8_scales != nullptr) cudaFree(d_fp8_scales);
  if (d_fp8_weights != nullptr) cudaFree(d_fp8_weights);

  if (err != cudaSuccess) {
    return fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_quant_fp8_dequant(
    const NervaCudaDeepSeekQuantFp8DequantRequest *request,
    NervaCudaDeepSeekQuantDequantResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_dequant_result(out);
  if (!validate_fp8_request(request)) {
    return -1;
  }
  out->rows = request->rows;
  out->cols = request->cols;
  out->block_rows = request->block_rows;
  out->block_cols = request->block_cols;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }
  if (out->device_count <= 0) {
    return fail_dequant(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }

  uint8_t *d_weights = nullptr;
  uint8_t *d_scales = nullptr;
  float *d_output = nullptr;
  float *h_output = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t value_count =
      static_cast<uint64_t>(request->rows) * request->cols;
  const uint64_t scale_cols =
      (static_cast<uint64_t>(request->cols) + request->block_cols - 1) /
      request->block_cols;
  const uint64_t scale_rows =
      (static_cast<uint64_t>(request->rows) + request->block_rows - 1) /
      request->block_rows;
  const uint64_t weights_bytes = value_count;
  const uint64_t scales_bytes = scale_rows * scale_cols;
  const uint64_t output_bytes = sizeof(float) * value_count;

  err = cudaMalloc(reinterpret_cast<void **>(&d_weights), weights_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_scales), scales_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output), output_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes = weights_bytes + scales_bytes + output_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output),
                      output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_weights,
                        request->weights,
                        weights_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_scales,
                        request->scales,
                        scales_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = weights_bytes + scales_bytes;

  {
    constexpr uint32_t threads = 256;
    const uint32_t blocks =
        static_cast<uint32_t>((value_count + threads - 1) / threads);
    fp8_e4m3fn_block_dequant_dynamic_kernel<<<blocks, threads, 0, stream>>>(
        d_weights,
        d_scales,
        d_output,
        request->rows,
        request->cols,
        request->block_rows,
        request->block_cols);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_output,
                        d_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->output, h_output, output_bytes);
  out->output_hash =
      hash_f32_bits(request->output, static_cast<uint32_t>(value_count));
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output != nullptr) cudaFreeHost(h_output);
  if (d_output != nullptr) cudaFree(d_output);
  if (d_scales != nullptr) cudaFree(d_scales);
  if (d_weights != nullptr) cudaFree(d_weights);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_quant_fp8_f32_scale_matvec(
    const NervaCudaDeepSeekQuantFp8F32ScaleMatvecRequest *request,
    NervaCudaDeepSeekQuantDequantResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_dequant_result(out);
  if (!validate_fp8_matvec_request(request)) {
    return -1;
  }
  out->rows = request->rows;
  out->cols = request->cols;
  out->block_rows = request->block_rows;
  out->block_cols = request->block_cols;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }
  if (out->device_count <= 0) {
    return fail_dequant(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }

  uint8_t *d_weights = nullptr;
  float *d_scales = nullptr;
  float *d_input = nullptr;
  float *d_output = nullptr;
  float *h_output = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t value_count =
      static_cast<uint64_t>(request->rows) * request->cols;
  const uint64_t scale_cols =
      (static_cast<uint64_t>(request->cols) + request->block_cols - 1) /
      request->block_cols;
  const uint64_t scale_rows =
      (static_cast<uint64_t>(request->rows) + request->block_rows - 1) /
      request->block_rows;
  const uint64_t weights_bytes = value_count;
  const uint64_t scales_bytes = scale_rows * scale_cols * sizeof(float);
  const uint64_t input_bytes = static_cast<uint64_t>(request->cols) * sizeof(float);
  const uint64_t output_bytes = static_cast<uint64_t>(request->rows) * sizeof(float);

  err = cudaMalloc(reinterpret_cast<void **>(&d_weights), weights_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_scales), scales_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_input), input_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output), output_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      weights_bytes + scales_bytes + input_bytes + output_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output),
                      output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_weights,
                        request->weights,
                        weights_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_scales,
                        request->scales,
                        scales_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_input,
                        request->input,
                        input_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = weights_bytes + scales_bytes + input_bytes;

  {
    err = launch_deepseek_fp8_f32_scale_matvec(
        stream,
        d_weights,
        d_scales,
        d_input,
        request->rows,
        request->cols,
        request->block_rows,
        request->block_cols,
        d_output);
    out->kernel_launches += 1;
  }
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_output,
                        d_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->output, h_output, output_bytes);
  out->output_hash = hash_f32_bits(request->output, request->rows);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output != nullptr) cudaFreeHost(h_output);
  if (d_output != nullptr) cudaFree(d_output);
  if (d_input != nullptr) cudaFree(d_input);
  if (d_scales != nullptr) cudaFree(d_scales);
  if (d_weights != nullptr) cudaFree(d_weights);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_quant_fp8_f32_scale_encoded_matvec(
    const NervaCudaDeepSeekQuantFp8F32ScaleEncodedMatvecRequest *request,
    NervaCudaDeepSeekQuantDequantResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_dequant_result(out);
  if (!validate_fp8_encoded_matvec_request(request)) {
    return -1;
  }
  out->rows = request->rows;
  out->cols = request->cols;
  out->block_rows = request->block_rows;
  out->block_cols = request->block_cols;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }
  if (out->device_count <= 0) {
    return fail_dequant(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }

  uint8_t *d_weights = nullptr;
  float *d_scales = nullptr;
  uint16_t *d_input = nullptr;
  float *d_output = nullptr;
  float *h_output = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t value_count =
      static_cast<uint64_t>(request->rows) * request->cols;
  const uint64_t scale_cols =
      (static_cast<uint64_t>(request->cols) + request->block_cols - 1) /
      request->block_cols;
  const uint64_t scale_rows =
      (static_cast<uint64_t>(request->rows) + request->block_rows - 1) /
      request->block_rows;
  const uint64_t weights_bytes = value_count;
  const uint64_t scales_bytes = scale_rows * scale_cols * sizeof(float);
  const uint64_t input_bytes =
      static_cast<uint64_t>(request->cols) * sizeof(uint16_t);
  const uint64_t output_bytes =
      static_cast<uint64_t>(request->rows) * sizeof(float);

  err = cudaMalloc(reinterpret_cast<void **>(&d_weights), weights_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_scales), scales_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_input), input_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output), output_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      weights_bytes + scales_bytes + input_bytes + output_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output),
                      output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_weights,
                        request->weights,
                        weights_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_scales,
                        request->scales,
                        scales_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_input,
                        request->input,
                        input_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = weights_bytes + scales_bytes + input_bytes;

  {
    err = launch_deepseek_fp8_f32_scale_encoded_matvec(
        stream,
        d_weights,
        d_scales,
        d_input,
        request->input_dtype,
        request->rows,
        request->cols,
        request->block_rows,
        request->block_cols,
        d_output);
    out->kernel_launches += 1;
  }
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_output,
                        d_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->output, h_output, output_bytes);
  out->output_hash = hash_f32_bits(request->output, request->rows);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output != nullptr) cudaFreeHost(h_output);
  if (d_output != nullptr) cudaFree(d_output);
  if (d_input != nullptr) cudaFree(d_input);
  if (d_scales != nullptr) cudaFree(d_scales);
  if (d_weights != nullptr) cudaFree(d_weights);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_quant_fp8_f32_scale_encoded_gemm_tokens(
    const NervaCudaDeepSeekQuantFp8F32ScaleEncodedGemmTokensRequest *request,
    NervaCudaDeepSeekQuantDequantResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_dequant_result(out);
  if (!validate_fp8_encoded_gemm_tokens_request(request)) {
    return -1;
  }
  out->rows = request->rows;
  out->cols = request->cols;
  out->block_rows = request->block_rows;
  out->block_cols = request->block_cols;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }
  if (out->device_count <= 0) {
    return fail_dequant(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }

  uint8_t *d_weights = nullptr;
  float *d_scales = nullptr;
  uint16_t *d_input = nullptr;
  float *d_output = nullptr;
  float *h_output = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t value_count =
      static_cast<uint64_t>(request->rows) * request->cols;
  const uint64_t scale_cols =
      (static_cast<uint64_t>(request->cols) + request->block_cols - 1) /
      request->block_cols;
  const uint64_t scale_rows =
      (static_cast<uint64_t>(request->rows) + request->block_rows - 1) /
      request->block_rows;
  const uint64_t output_values =
      static_cast<uint64_t>(request->tokens) * request->rows;
  const uint64_t weights_bytes = value_count;
  const uint64_t scales_bytes = scale_rows * scale_cols * sizeof(float);
  const uint64_t input_bytes =
      static_cast<uint64_t>(request->tokens) * request->cols *
      sizeof(uint16_t);
  const uint64_t output_bytes = output_values * sizeof(float);

  err = cudaMalloc(reinterpret_cast<void **>(&d_weights), weights_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_scales), scales_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_input), input_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output), output_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      weights_bytes + scales_bytes + input_bytes + output_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output),
                      output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_weights,
                        request->weights,
                        weights_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_scales,
                        request->scales,
                        scales_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_input,
                        request->input,
                        input_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = weights_bytes + scales_bytes + input_bytes;

  {
    err = launch_deepseek_fp8_f32_scale_encoded_gemm_tokens(
        stream,
        d_weights,
        d_scales,
        d_input,
        request->input_dtype,
        request->rows,
        request->cols,
        request->tokens,
        request->block_rows,
        request->block_cols,
        d_output);
    out->kernel_launches += 1;
  }
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_output,
                        d_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->output, h_output, output_bytes);
  out->output_hash = hash_f32_bits(request->output, output_values);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output != nullptr) cudaFreeHost(h_output);
  if (d_output != nullptr) cudaFree(d_output);
  if (d_input != nullptr) cudaFree(d_input);
  if (d_scales != nullptr) cudaFree(d_scales);
  if (d_weights != nullptr) cudaFree(d_weights);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_quant_fp8_e8m0_scale_encoded_gemm_tokens(
    const NervaCudaDeepSeekQuantFp8E8m0ScaleEncodedGemmTokensRequest *request,
    NervaCudaDeepSeekQuantDequantResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_dequant_result(out);
  if (!validate_fp8_e8m0_encoded_gemm_tokens_request(request)) {
    return -1;
  }
  out->rows = request->rows;
  out->cols = request->cols;
  out->block_rows = request->block_rows;
  out->block_cols = request->block_cols;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }
  if (out->device_count <= 0) {
    return fail_dequant(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }

  uint8_t *d_weights = nullptr;
  uint8_t *d_scales = nullptr;
  uint16_t *d_input = nullptr;
  float *d_output = nullptr;
  float *h_output = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t value_count =
      static_cast<uint64_t>(request->rows) * request->cols;
  const uint64_t scale_cols =
      (static_cast<uint64_t>(request->cols) + request->block_cols - 1) /
      request->block_cols;
  const uint64_t scale_rows =
      (static_cast<uint64_t>(request->rows) + request->block_rows - 1) /
      request->block_rows;
  const uint64_t output_values =
      static_cast<uint64_t>(request->tokens) * request->rows;
  const uint64_t weights_bytes = value_count;
  const uint64_t scales_bytes = scale_rows * scale_cols;
  const uint64_t input_bytes =
      static_cast<uint64_t>(request->tokens) * request->cols *
      sizeof(uint16_t);
  const uint64_t output_bytes = output_values * sizeof(float);

  err = cudaMalloc(reinterpret_cast<void **>(&d_weights), weights_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_scales), scales_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_input), input_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output), output_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      weights_bytes + scales_bytes + input_bytes + output_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output),
                      output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_weights,
                        request->weights,
                        weights_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_scales,
                        request->scales,
                        scales_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_input,
                        request->input,
                        input_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = weights_bytes + scales_bytes + input_bytes;

  {
    err = launch_deepseek_fp8_e8m0_scale_encoded_gemm_tokens(
        stream,
        d_weights,
        d_scales,
        d_input,
        request->input_dtype,
        request->rows,
        request->cols,
        request->tokens,
        request->block_rows,
        request->block_cols,
        d_output);
    out->kernel_launches += 1;
  }
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_output,
                        d_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->output, h_output, output_bytes);
  out->output_hash = hash_f32_bits(request->output, output_values);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output != nullptr) cudaFreeHost(h_output);
  if (d_output != nullptr) cudaFree(d_output);
  if (d_input != nullptr) cudaFree(d_input);
  if (d_scales != nullptr) cudaFree(d_scales);
  if (d_weights != nullptr) cudaFree(d_weights);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_quant_mxfp4_dequant(
    const NervaCudaDeepSeekQuantMxfp4DequantRequest *request,
    NervaCudaDeepSeekQuantDequantResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_dequant_result(out);
  if (!validate_mxfp4_request(request)) {
    return -1;
  }
  out->rows = request->rows;
  out->cols = request->packed_cols * 2;
  out->block_rows = 1;
  out->block_cols = request->scale_packed_cols * 2;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }
  if (out->device_count <= 0) {
    return fail_dequant(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }

  uint8_t *d_packed = nullptr;
  uint8_t *d_scales = nullptr;
  float *d_output = nullptr;
  float *h_output = nullptr;
  cudaStream_t stream = nullptr;

  const uint64_t packed_count =
      static_cast<uint64_t>(request->rows) * request->packed_cols;
  const uint64_t scale_cols =
      (static_cast<uint64_t>(request->packed_cols) +
       request->scale_packed_cols - 1) /
      request->scale_packed_cols;
  const uint64_t scales_bytes = static_cast<uint64_t>(request->rows) * scale_cols;
  const uint64_t packed_bytes = packed_count;
  const uint64_t output_values = packed_count * 2;
  const uint64_t output_bytes = sizeof(float) * output_values;

  err = cudaMalloc(reinterpret_cast<void **>(&d_packed), packed_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_scales), scales_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_output), output_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes = packed_bytes + scales_bytes + output_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_output),
                      output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_packed,
                        request->packed,
                        packed_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_scales,
                        request->scales,
                        scales_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = packed_bytes + scales_bytes;

  {
    constexpr uint32_t threads = 256;
    const uint32_t blocks =
        static_cast<uint32_t>((packed_count + threads - 1) / threads);
    mxfp4_e2m1_block_dequant_dynamic_kernel<<<blocks, threads, 0, stream>>>(
        d_packed,
        d_scales,
        d_output,
        request->rows,
        request->packed_cols,
        request->scale_packed_cols);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_output,
                        d_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->output, h_output, output_bytes);
  out->output_hash =
      hash_f32_bits(request->output, static_cast<uint32_t>(output_values));
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_output != nullptr) cudaFreeHost(h_output);
  if (d_output != nullptr) cudaFree(d_output);
  if (d_scales != nullptr) cudaFree(d_scales);
  if (d_packed != nullptr) cudaFree(d_packed);
  if (err != cudaSuccess) {
    return fail_dequant(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_fused_inv_rope_fp8_quant(
    const NervaCudaDeepSeekFusedInvRopeFp8QuantRequest *request,
    NervaCudaDeepSeekFusedInvRopeFp8QuantResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_inv_rope_result(out);
  if (!validate_inv_rope_request(request)) {
    return -1;
  }

  out->num_tokens = request->num_tokens;
  out->n_groups = request->n_groups;
  out->heads_per_group = request->heads_per_group;
  out->head_dim = request->head_dim;
  out->rope_dim = request->rope_dim;
  out->quant_group_size = request->quant_group_size;
  out->scale_blocks =
      request->heads_per_group * (request->head_dim / request->quant_group_size);

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_inv_rope(out, err);
  }
  if (out->device_count <= 0) {
    return fail_inv_rope(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_inv_rope(out, err);
  }

  float *d_input = nullptr;
  int64_t *d_positions = nullptr;
  float *d_cos_sin = nullptr;
  uint8_t *d_fp8_output = nullptr;
  float *d_scale_output = nullptr;
  uint32_t *d_packed_scale_output = nullptr;
  uint8_t *h_fp8_output = nullptr;
  float *h_scale_output = nullptr;
  uint32_t *h_packed_scale_output = nullptr;
  cudaStream_t stream = nullptr;

  const uint32_t num_heads = request->n_groups * request->heads_per_group;
  const uint64_t input_values =
      static_cast<uint64_t>(request->num_tokens) * num_heads * request->head_dim;
  const uint64_t position_values = request->num_tokens;
  uint64_t max_position = 0;
  for (uint32_t idx = 0; idx < request->num_tokens; ++idx) {
    if (request->positions[idx] > static_cast<int64_t>(max_position)) {
      max_position = static_cast<uint64_t>(request->positions[idx]);
    }
  }
  const uint64_t cos_sin_values =
      (max_position + 1ull) * request->cos_sin_stride;
  const uint64_t fp8_values =
      static_cast<uint64_t>(request->n_groups) * request->num_tokens *
      request->heads_per_group * request->head_dim;
  const uint64_t scale_values =
      static_cast<uint64_t>(request->n_groups) * request->num_tokens *
      out->scale_blocks;
  const uint64_t packed_values =
      static_cast<uint64_t>(request->n_groups) * request->num_tokens *
      request->heads_per_group;
  if (input_values > UINT32_MAX || fp8_values > UINT32_MAX ||
      scale_values > UINT32_MAX || packed_values > UINT32_MAX) {
    return -1;
  }

  const uint64_t input_bytes = input_values * sizeof(float);
  const uint64_t position_bytes = position_values * sizeof(int64_t);
  const uint64_t cos_sin_bytes = cos_sin_values * sizeof(float);
  const uint64_t fp8_bytes = fp8_values;
  const uint64_t scale_bytes = scale_values * sizeof(float);
  const uint64_t packed_bytes = packed_values * sizeof(uint32_t);

  err = cudaMalloc(reinterpret_cast<void **>(&d_input), input_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_positions), position_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_cos_sin), cos_sin_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_fp8_output), fp8_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_scale_output), scale_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_packed_scale_output),
                   packed_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes = input_bytes + position_bytes + cos_sin_bytes +
                            fp8_bytes + scale_bytes + packed_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_fp8_output),
                      fp8_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_scale_output),
                      scale_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_packed_scale_output),
                      packed_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = fp8_bytes + scale_bytes + packed_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;
  err =
      cudaMemcpyAsync(d_input, request->input, input_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_positions,
                        request->positions,
                        position_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(d_cos_sin,
                        request->cos_sin_cache,
                        cos_sin_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes = input_bytes + position_bytes + cos_sin_bytes;

  err = cudaMemsetAsync(d_fp8_output, 0, fp8_bytes, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemsetAsync(d_scale_output, 0, scale_bytes, stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemsetAsync(d_packed_scale_output, 0, packed_bytes, stream);
  if (err != cudaSuccess) goto cleanup;

  {
    const dim3 grid(request->num_tokens, num_heads, 1);
    fused_inv_rope_fp8_quant_kernel<<<grid, 1, 0, stream>>>(
        d_input,
        d_positions,
        d_cos_sin,
        d_fp8_output,
        d_scale_output,
        d_packed_scale_output,
        request->num_tokens,
        request->heads_per_group,
        request->head_dim,
        request->rope_dim,
        request->quant_group_size,
        request->cos_sin_stride,
        out->scale_blocks,
        request->fp8_max,
        request->eps);
    out->kernel_launches += 1;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_fp8_output,
                        d_fp8_output,
                        fp8_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_scale_output,
                        d_scale_output,
                        scale_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_packed_scale_output,
                        d_packed_scale_output,
                        packed_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = fp8_bytes + scale_bytes + packed_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(request->fp8_output, h_fp8_output, fp8_bytes);
  memcpy(request->scale_output, h_scale_output, scale_bytes);
  memcpy(request->packed_scale_output, h_packed_scale_output, packed_bytes);
  out->fp8_output_hash =
      hash_bytes(reinterpret_cast<const uint8_t *>(request->fp8_output),
                 fp8_bytes);
  out->scale_output_hash = hash_f32_bits(
      request->scale_output, static_cast<uint32_t>(scale_values));
  out->packed_scale_output_hash = hash_bytes(
      reinterpret_cast<const uint8_t *>(request->packed_scale_output),
      packed_bytes);
  out->status = 0;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_packed_scale_output != nullptr) cudaFreeHost(h_packed_scale_output);
  if (h_scale_output != nullptr) cudaFreeHost(h_scale_output);
  if (h_fp8_output != nullptr) cudaFreeHost(h_fp8_output);
  if (d_packed_scale_output != nullptr) cudaFree(d_packed_scale_output);
  if (d_scale_output != nullptr) cudaFree(d_scale_output);
  if (d_fp8_output != nullptr) cudaFree(d_fp8_output);
  if (d_cos_sin != nullptr) cudaFree(d_cos_sin);
  if (d_positions != nullptr) cudaFree(d_positions);
  if (d_input != nullptr) cudaFree(d_input);
  if (err != cudaSuccess) {
    return fail_inv_rope(out, err);
  }
  return out->status == 0 ? 0 : -1;
}
