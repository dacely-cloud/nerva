#include "../../deepseek_quant.cuh"

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
  const uint32_t scale_cols = (cols + block_cols - 1) / block_cols;
  const uint64_t row_base = static_cast<uint64_t>(row) * cols;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint32_t scale_idx =
        (row / block_rows) * scale_cols + (col / block_cols);
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
  const uint32_t scale_cols = (cols + block_cols - 1) / block_cols;
  const uint64_t row_base = static_cast<uint64_t>(row) * cols;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint32_t scale_idx =
        (row / block_rows) * scale_cols + (col / block_cols);
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
  const uint32_t row = blockIdx.x;
  const uint32_t token = blockIdx.y;
  if (row >= rows || token >= tokens) {
    return;
  }
  extern __shared__ float partial[];
  float sum = 0.0f;
  const uint32_t scale_cols = (cols + block_cols - 1) / block_cols;
  const uint64_t row_base = static_cast<uint64_t>(row) * cols;
  const uint64_t input_base = static_cast<uint64_t>(token) * cols;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint32_t scale_idx =
        (row / block_rows) * scale_cols + (col / block_cols);
    const float weight =
        nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
        scales[scale_idx];
    sum += weight * encoded_input_to_f32(input[input_base + col], input_dtype);
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
    output[static_cast<uint64_t>(token) * rows + row] = partial[0];
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
  const uint32_t scale_cols = (cols + block_cols - 1) / block_cols;
  const uint64_t row_base = static_cast<uint64_t>(row) * cols;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint32_t scale_idx =
        (row / block_rows) * scale_cols + (col / block_cols);
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
  const uint32_t row = blockIdx.x;
  const uint32_t token = blockIdx.y;
  if (row >= rows || token >= tokens) {
    return;
  }
  extern __shared__ float partial[];
  float sum = 0.0f;
  const uint32_t scale_cols = (cols + block_cols - 1) / block_cols;
  const uint64_t row_base = static_cast<uint64_t>(row) * cols;
  const uint64_t input_base = static_cast<uint64_t>(token) * cols;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint32_t scale_idx =
        (row / block_rows) * scale_cols + (col / block_cols);
    const float weight =
        nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
        nerva::deepseek::e8m0_exponent_bits_to_f32(scales[scale_idx]);
    sum += weight * encoded_input_to_f32(input[input_base + col], input_dtype);
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
    output[static_cast<uint64_t>(token) * rows + row] = partial[0];
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
  const uint32_t scale_cols = (cols + block_cols - 1) / block_cols;
  const uint64_t row_base = static_cast<uint64_t>(row) * cols;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint32_t scale_idx =
        (row / block_rows) * scale_cols + (col / block_cols);
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
  const size_t shared_bytes = threads * sizeof(float);
  const dim3 grid(rows, tokens, 1);
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
  const size_t shared_bytes = threads * sizeof(float);
  const dim3 grid(rows, tokens, 1);
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
