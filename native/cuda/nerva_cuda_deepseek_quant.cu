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

void clear_dequant_result(NervaCudaDeepSeekQuantDequantResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
}

int fail_dequant(NervaCudaDeepSeekQuantDequantResult *out, cudaError_t err) {
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

bool validate_mxfp4_request(
    const NervaCudaDeepSeekQuantMxfp4DequantRequest *request) {
  return request != nullptr && request->packed != nullptr &&
         request->scales != nullptr && request->output != nullptr &&
         request->rows > 0 && request->packed_cols > 0 &&
         request->scale_packed_cols > 0;
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
