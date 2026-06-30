#include "nerva_cuda_api.h"
#include "deepseek_quant.cuh"

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
