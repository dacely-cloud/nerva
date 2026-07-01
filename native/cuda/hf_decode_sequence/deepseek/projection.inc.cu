#include "../../deepseek_quant.cuh"

constexpr uint32_t kDeepSeekFp8ProjectionTokenTile = 4;
constexpr uint32_t kDeepSeekFp8ProjectionRowTile = 8;

__device__ __forceinline__ uint32_t deepseek_fp8_projection_scale_cols(
    uint32_t cols,
    uint32_t block_cols) {
  if (block_cols == 128u) {
    return (cols + 127u) >> 7u;
  }
  return (cols + block_cols - 1) / block_cols;
}

__device__ __forceinline__ uint32_t deepseek_fp8_projection_scale_idx(
    uint32_t row,
    uint32_t col,
    uint32_t scale_cols,
    uint32_t block_rows,
    uint32_t block_cols) {
  if (block_rows == 128u && block_cols == 128u) {
    return ((row >> 7u) * scale_cols) + (col >> 7u);
  }
  return (row / block_rows) * scale_cols + (col / block_cols);
}

__device__ __forceinline__ float deepseek_fp8_projection_f32_scale_slot(
    const uint16_t *scale_slots,
    uint32_t index) {
  const uint32_t lo = static_cast<uint32_t>(scale_slots[index * 2u]);
  const uint32_t hi = static_cast<uint32_t>(scale_slots[index * 2u + 1u]);
  return __uint_as_float(lo | (hi << 16u));
}

__global__ void deepseek_fp8_f32_scale_matvec_kernel(
    const uint8_t *weights,
    const float *scales,
    const float *input,
    float *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t block_rows,
    uint32_t block_cols) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ float partial[];
  float sum = 0.0f;
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  const uint64_t row_base = static_cast<uint64_t>(row) * cols;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint32_t scale_idx =
        deepseek_fp8_projection_scale_idx(row, col, scale_cols, block_rows,
                                          block_cols);
    const float weight =
        nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
        scales[scale_idx];
    sum += weight * input[col];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[row] = partial[0];
  }
}

__global__ void deepseek_fp8_f32_scale_encoded_matvec_kernel(
    const uint8_t *weights,
    const float *scales,
    const uint16_t *input,
    uint32_t input_dtype,
    float *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t block_rows,
    uint32_t block_cols) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ float partial[];
  float sum = 0.0f;
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  const uint64_t row_base = static_cast<uint64_t>(row) * cols;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint32_t scale_idx =
        deepseek_fp8_projection_scale_idx(row, col, scale_cols, block_rows,
                                          block_cols);
    const float weight =
        nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
        scales[scale_idx];
    sum += weight * encoded_input_to_f32(input[col], input_dtype);
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[row] = partial[0];
  }
}

__global__ void deepseek_fp8_f32_scale_slots_encoded_matvec_kernel(
    const uint8_t *weights,
    const uint16_t *scale_slots,
    const uint16_t *input,
    uint32_t input_dtype,
    float *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t block_rows,
    uint32_t block_cols) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ float partial[];
  float sum = 0.0f;
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  const uint64_t row_base = static_cast<uint64_t>(row) * cols;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint32_t scale_idx =
        deepseek_fp8_projection_scale_idx(row, col, scale_cols, block_rows,
                                          block_cols);
    const float weight =
        nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
        deepseek_fp8_projection_f32_scale_slot(scale_slots, scale_idx);
    sum += weight * encoded_input_to_f32(input[col], input_dtype);
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[row] = partial[0];
  }
}

__global__ void deepseek_fp8_f32_scale_encoded_gemm_tokens_kernel(
    const uint8_t *weights,
    const float *scales,
    const uint16_t *input,
    uint32_t input_dtype,
    float *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t tokens,
    uint32_t block_rows,
    uint32_t block_cols) {
  const uint32_t row_start = blockIdx.x * kDeepSeekFp8ProjectionRowTile;
  const uint32_t token_start = blockIdx.y * kDeepSeekFp8ProjectionTokenTile;
  if (row_start >= rows || token_start >= tokens) {
    return;
  }
  extern __shared__ float partial[];
  float sum[kDeepSeekFp8ProjectionRowTile][kDeepSeekFp8ProjectionTokenTile] = {};
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    float input_value[kDeepSeekFp8ProjectionTokenTile] = {};
    for (uint32_t tile = 0; tile < kDeepSeekFp8ProjectionTokenTile; ++tile) {
      const uint32_t token = token_start + tile;
      if (token < tokens) {
        const uint64_t input_base = static_cast<uint64_t>(token) * cols;
        input_value[tile] = encoded_input_to_f32(input[input_base + col],
                                                 input_dtype);
      }
    }
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= rows) {
        continue;
      }
      const uint32_t scale_idx =
          deepseek_fp8_projection_scale_idx(row, col, scale_cols, block_rows,
                                            block_cols);
      const uint64_t row_base = static_cast<uint64_t>(row) * cols;
      const float weight =
          nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
          scales[scale_idx];
      for (uint32_t tile = 0; tile < kDeepSeekFp8ProjectionTokenTile; ++tile) {
        sum[row_tile][tile] += weight * input_value[tile];
      }
    }
  }
  for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
       ++row_tile) {
    for (uint32_t tile = 0; tile < kDeepSeekFp8ProjectionTokenTile; ++tile) {
      partial[(row_tile * kDeepSeekFp8ProjectionTokenTile + tile) *
                  blockDim.x +
              threadIdx.x] = sum[row_tile][tile];
    }
  }
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
           ++row_tile) {
        for (uint32_t tile = 0; tile < kDeepSeekFp8ProjectionTokenTile;
             ++tile) {
          const uint32_t partial_offset =
              (row_tile * kDeepSeekFp8ProjectionTokenTile + tile) *
              blockDim.x;
          partial[partial_offset + threadIdx.x] +=
              partial[partial_offset + threadIdx.x + stride];
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= rows) {
        continue;
      }
      for (uint32_t tile = 0; tile < kDeepSeekFp8ProjectionTokenTile; ++tile) {
        const uint32_t token = token_start + tile;
        if (token < tokens) {
          const uint32_t partial_offset =
              (row_tile * kDeepSeekFp8ProjectionTokenTile + tile) *
              blockDim.x;
          output[static_cast<uint64_t>(token) * rows + row] =
              partial[partial_offset];
        }
      }
    }
  }
}

__global__ void deepseek_fp8_e8m0_scale_encoded_matvec_kernel(
    const uint8_t *weights,
    const uint8_t *scales,
    const uint16_t *input,
    uint32_t input_dtype,
    float *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t block_rows,
    uint32_t block_cols) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ float partial[];
  float sum = 0.0f;
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  const uint64_t row_base = static_cast<uint64_t>(row) * cols;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint32_t scale_idx =
        deepseek_fp8_projection_scale_idx(row, col, scale_cols, block_rows,
                                          block_cols);
    const float weight =
        nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
        nerva::deepseek::e8m0_exponent_bits_to_f32(scales[scale_idx]);
    sum += weight * encoded_input_to_f32(input[col], input_dtype);
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[row] = partial[0];
  }
}

__global__ void deepseek_fp8_e8m0_scale_encoded_gemm_tokens_kernel(
    const uint8_t *weights,
    const uint8_t *scales,
    const uint16_t *input,
    uint32_t input_dtype,
    float *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t tokens,
    uint32_t block_rows,
    uint32_t block_cols) {
  const uint32_t row_start = blockIdx.x * kDeepSeekFp8ProjectionRowTile;
  const uint32_t token_start = blockIdx.y * kDeepSeekFp8ProjectionTokenTile;
  if (row_start >= rows || token_start >= tokens) {
    return;
  }
  extern __shared__ float partial[];
  float sum[kDeepSeekFp8ProjectionRowTile][kDeepSeekFp8ProjectionTokenTile] = {};
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    float input_value[kDeepSeekFp8ProjectionTokenTile] = {};
    for (uint32_t tile = 0; tile < kDeepSeekFp8ProjectionTokenTile; ++tile) {
      const uint32_t token = token_start + tile;
      if (token < tokens) {
        const uint64_t input_base = static_cast<uint64_t>(token) * cols;
        input_value[tile] = encoded_input_to_f32(input[input_base + col],
                                                 input_dtype);
      }
    }
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= rows) {
        continue;
      }
      const uint32_t scale_idx =
          deepseek_fp8_projection_scale_idx(row, col, scale_cols, block_rows,
                                            block_cols);
      const uint64_t row_base = static_cast<uint64_t>(row) * cols;
      const float weight =
          nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
          nerva::deepseek::e8m0_exponent_bits_to_f32(scales[scale_idx]);
      for (uint32_t tile = 0; tile < kDeepSeekFp8ProjectionTokenTile; ++tile) {
        sum[row_tile][tile] += weight * input_value[tile];
      }
    }
  }
  for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
       ++row_tile) {
    for (uint32_t tile = 0; tile < kDeepSeekFp8ProjectionTokenTile; ++tile) {
      partial[(row_tile * kDeepSeekFp8ProjectionTokenTile + tile) *
                  blockDim.x +
              threadIdx.x] = sum[row_tile][tile];
    }
  }
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
           ++row_tile) {
        for (uint32_t tile = 0; tile < kDeepSeekFp8ProjectionTokenTile;
             ++tile) {
          const uint32_t partial_offset =
              (row_tile * kDeepSeekFp8ProjectionTokenTile + tile) *
              blockDim.x;
          partial[partial_offset + threadIdx.x] +=
              partial[partial_offset + threadIdx.x + stride];
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= rows) {
        continue;
      }
      for (uint32_t tile = 0; tile < kDeepSeekFp8ProjectionTokenTile; ++tile) {
        const uint32_t token = token_start + tile;
        if (token < tokens) {
          const uint32_t partial_offset =
              (row_tile * kDeepSeekFp8ProjectionTokenTile + tile) *
              blockDim.x;
          output[static_cast<uint64_t>(token) * rows + row] =
              partial[partial_offset];
        }
      }
    }
  }
}

__global__ void deepseek_fp8_e8m0_scale_matvec_kernel(
    const uint8_t *weights,
    const uint8_t *scales,
    const float *input,
    float *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t block_rows,
    uint32_t block_cols) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ float partial[];
  float sum = 0.0f;
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  const uint64_t row_base = static_cast<uint64_t>(row) * cols;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint32_t scale_idx =
        deepseek_fp8_projection_scale_idx(row, col, scale_cols, block_rows,
                                          block_cols);
    const float weight =
        nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
        nerva::deepseek::e8m0_exponent_bits_to_f32(scales[scale_idx]);
    sum += weight * input[col];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[row] = partial[0];
  }
}

__global__ void deepseek_fp8_e8m0_scale_matvec_row_offset_kernel(
    const uint8_t *weights,
    const uint8_t *scales,
    const float *input,
    float *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t global_row_offset,
    uint32_t block_rows,
    uint32_t block_cols) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ float partial[];
  float sum = 0.0f;
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  const uint32_t global_row = global_row_offset + row;
  const uint64_t row_base = static_cast<uint64_t>(row) * cols;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint32_t scale_idx =
        deepseek_fp8_projection_scale_idx(global_row, col, scale_cols,
                                          block_rows, block_cols);
    const float weight =
        nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
        nerva::deepseek::e8m0_exponent_bits_to_f32(scales[scale_idx]);
    sum += weight * input[col];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[row] = partial[0];
  }
}

cudaError_t launch_deepseek_fp8_f32_scale_matvec(
    cudaStream_t stream, const uint8_t *weights, const float *scales,
    const float *input, uint32_t rows, uint32_t cols, uint32_t block_rows,
    uint32_t block_cols, float *output) {
  if (weights == nullptr || scales == nullptr || input == nullptr ||
      output == nullptr || rows == 0 || cols == 0 || block_rows == 0 ||
      block_cols == 0) {
    return cudaErrorInvalidValue;
  }
  constexpr uint32_t threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  deepseek_fp8_f32_scale_matvec_kernel<<<rows, threads, shared_bytes, stream>>>(
      weights, scales, input, output, rows, cols, block_rows, block_cols);
  return cudaGetLastError();
}

cudaError_t launch_deepseek_fp8_e8m0_scale_matvec_row_offset(
    cudaStream_t stream, const uint8_t *weights, const uint8_t *scales,
    const float *input, uint32_t rows, uint32_t cols,
    uint32_t global_row_offset, uint32_t block_rows, uint32_t block_cols,
    float *output) {
  if (weights == nullptr || scales == nullptr || input == nullptr ||
      output == nullptr || rows == 0 || cols == 0 || block_rows == 0 ||
      block_cols == 0) {
    return cudaErrorInvalidValue;
  }
  constexpr uint32_t threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  deepseek_fp8_e8m0_scale_matvec_row_offset_kernel<<<rows,
                                                     threads,
                                                     shared_bytes,
                                                     stream>>>(
      weights, scales, input, output, rows, cols, global_row_offset,
      block_rows, block_cols);
  return cudaGetLastError();
}

cudaError_t launch_deepseek_fp8_f32_scale_encoded_matvec(
    cudaStream_t stream, const uint8_t *weights, const float *scales,
    const uint16_t *input, uint32_t input_dtype, uint32_t rows, uint32_t cols,
    uint32_t block_rows, uint32_t block_cols, float *output) {
  if (weights == nullptr || scales == nullptr || input == nullptr ||
      output == nullptr || rows == 0 || cols == 0 || block_rows == 0 ||
      block_cols == 0 || input_dtype > kDTypeBF16) {
    return cudaErrorInvalidValue;
  }
  constexpr uint32_t threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  deepseek_fp8_f32_scale_encoded_matvec_kernel<<<rows,
                                                 threads,
                                                 shared_bytes,
                                                 stream>>>(
      weights, scales, input, input_dtype, output, rows, cols, block_rows,
      block_cols);
  return cudaGetLastError();
}

cudaError_t launch_deepseek_fp8_f32_scale_slots_encoded_matvec(
    cudaStream_t stream, const uint8_t *weights, const uint16_t *scale_slots,
    const uint16_t *input, uint32_t input_dtype, uint32_t rows, uint32_t cols,
    uint32_t block_rows, uint32_t block_cols, float *output) {
  if (weights == nullptr || scale_slots == nullptr || input == nullptr ||
      output == nullptr || rows == 0 || cols == 0 || block_rows == 0 ||
      block_cols == 0 || input_dtype > kDTypeBF16) {
    return cudaErrorInvalidValue;
  }
  constexpr uint32_t threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  deepseek_fp8_f32_scale_slots_encoded_matvec_kernel<<<rows,
                                                       threads,
                                                       shared_bytes,
                                                       stream>>>(
      weights, scale_slots, input, input_dtype, output, rows, cols, block_rows,
      block_cols);
  return cudaGetLastError();
}

cudaError_t launch_deepseek_fp8_f32_scale_encoded_gemm_tokens(
    cudaStream_t stream, const uint8_t *weights, const float *scales,
    const uint16_t *input, uint32_t input_dtype, uint32_t rows, uint32_t cols,
    uint32_t tokens, uint32_t block_rows, uint32_t block_cols,
    float *output) {
  if (weights == nullptr || scales == nullptr || input == nullptr ||
      output == nullptr || rows == 0 || cols == 0 || tokens == 0 ||
      block_rows == 0 || block_cols == 0 || input_dtype > kDTypeBF16) {
    return cudaErrorInvalidValue;
  }
  constexpr uint32_t threads = 256;
  const size_t shared_bytes =
      threads * kDeepSeekFp8ProjectionRowTile *
      kDeepSeekFp8ProjectionTokenTile * sizeof(float);
  const dim3 grid((rows + kDeepSeekFp8ProjectionRowTile - 1) /
                      kDeepSeekFp8ProjectionRowTile,
                  (tokens + kDeepSeekFp8ProjectionTokenTile - 1) /
                      kDeepSeekFp8ProjectionTokenTile,
                  1);
  deepseek_fp8_f32_scale_encoded_gemm_tokens_kernel<<<grid,
                                                      threads,
                                                      shared_bytes,
                                                      stream>>>(
      weights, scales, input, input_dtype, output, rows, cols, tokens,
      block_rows, block_cols);
  return cudaGetLastError();
}

cudaError_t launch_deepseek_fp8_e8m0_scale_encoded_matvec(
    cudaStream_t stream, const uint8_t *weights, const uint8_t *scales,
    const uint16_t *input, uint32_t input_dtype, uint32_t rows, uint32_t cols,
    uint32_t block_rows, uint32_t block_cols, float *output) {
  if (weights == nullptr || scales == nullptr || input == nullptr ||
      output == nullptr || rows == 0 || cols == 0 || block_rows == 0 ||
      block_cols == 0 || input_dtype > kDTypeBF16) {
    return cudaErrorInvalidValue;
  }
  constexpr uint32_t threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  deepseek_fp8_e8m0_scale_encoded_matvec_kernel<<<rows,
                                                   threads,
                                                   shared_bytes,
                                                   stream>>>(
      weights, scales, input, input_dtype, output, rows, cols, block_rows,
      block_cols);
  return cudaGetLastError();
}

cudaError_t launch_deepseek_fp8_e8m0_scale_encoded_gemm_tokens(
    cudaStream_t stream, const uint8_t *weights, const uint8_t *scales,
    const uint16_t *input, uint32_t input_dtype, uint32_t rows, uint32_t cols,
    uint32_t tokens, uint32_t block_rows, uint32_t block_cols,
    float *output) {
  if (weights == nullptr || scales == nullptr || input == nullptr ||
      output == nullptr || rows == 0 || cols == 0 || tokens == 0 ||
      block_rows == 0 || block_cols == 0 || input_dtype > kDTypeBF16) {
    return cudaErrorInvalidValue;
  }
  constexpr uint32_t threads = 256;
  const size_t shared_bytes =
      threads * kDeepSeekFp8ProjectionRowTile *
      kDeepSeekFp8ProjectionTokenTile * sizeof(float);
  const dim3 grid((rows + kDeepSeekFp8ProjectionRowTile - 1) /
                      kDeepSeekFp8ProjectionRowTile,
                  (tokens + kDeepSeekFp8ProjectionTokenTile - 1) /
                      kDeepSeekFp8ProjectionTokenTile,
                  1);
  deepseek_fp8_e8m0_scale_encoded_gemm_tokens_kernel<<<grid,
                                                       threads,
                                                       shared_bytes,
                                                       stream>>>(
      weights, scales, input, input_dtype, output, rows, cols, tokens,
      block_rows, block_cols);
  return cudaGetLastError();
}

cudaError_t launch_deepseek_fp8_e8m0_scale_matvec(
    cudaStream_t stream, const uint8_t *weights, const uint8_t *scales,
    const float *input, uint32_t rows, uint32_t cols, uint32_t block_rows,
    uint32_t block_cols, float *output) {
  if (weights == nullptr || scales == nullptr || input == nullptr ||
      output == nullptr || rows == 0 || cols == 0 || block_rows == 0 ||
      block_cols == 0) {
    return cudaErrorInvalidValue;
  }
  constexpr uint32_t threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  deepseek_fp8_e8m0_scale_matvec_kernel<<<rows,
                                          threads,
                                          shared_bytes,
                                          stream>>>(
      weights, scales, input, output, rows, cols, block_rows, block_cols);
  return cudaGetLastError();
}
