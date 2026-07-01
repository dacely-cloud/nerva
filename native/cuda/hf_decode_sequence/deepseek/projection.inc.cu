#include "../../deepseek/deepseek_quant.cuh"

constexpr uint32_t kDeepSeekFp8ProjectionTokenTile = 4;
constexpr uint32_t kDeepSeekFp8ProjectionRowTile = 8;

template <uint32_t Slots>
__device__ __forceinline__ void deepseek_fp8_projection_reduce_slots(
    float (&values)[Slots], float *partial) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5u;
  const uint32_t warp_count = (blockDim.x + 31u) >> 5u;

#pragma unroll
  for (uint32_t slot = 0; slot < Slots; ++slot) {
    float value = values[slot];
#pragma unroll
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
      value += __shfl_down_sync(0xffffffffu, value, static_cast<int>(offset));
    }
    if (lane == 0u) {
      partial[slot * blockDim.x + warp] = value;
    }
  }
  __syncthreads();

  if (warp == 0u) {
#pragma unroll
    for (uint32_t slot = 0; slot < Slots; ++slot) {
      float value =
          lane < warp_count ? partial[slot * blockDim.x + lane] : 0.0f;
#pragma unroll
      for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        value +=
            __shfl_down_sync(0xffffffffu, value, static_cast<int>(offset));
      }
      if (lane == 0u) {
        partial[slot * blockDim.x] = value;
      }
    }
  }
  __syncthreads();
}

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

__device__ __forceinline__ bool deepseek_fp8_projection_row_tile_shares_scale(
    uint32_t row_start,
    uint32_t block_rows) {
  if (block_rows < kDeepSeekFp8ProjectionRowTile) {
    return false;
  }
  const uint32_t row_end = row_start + kDeepSeekFp8ProjectionRowTile - 1u;
  return (row_start / block_rows) == (row_end / block_rows);
}

__device__ __forceinline__ float deepseek_fp8_projection_f32_scale_slot(
    const uint16_t *scale_slots,
    uint32_t index) {
  const uint32_t lo = static_cast<uint32_t>(scale_slots[index * 2u]);
  const uint32_t hi = static_cast<uint32_t>(scale_slots[index * 2u + 1u]);
  return __uint_as_float(lo | (hi << 16u));
}

struct DeepSeekFp8F32ScaleReader {
  const float *scales;
  __device__ __forceinline__ float get(uint32_t index) const {
    return scales[index];
  }
};

struct DeepSeekFp8E8M0ScaleReader {
  const uint8_t *scales;
  __device__ __forceinline__ float get(uint32_t index) const {
    return nerva::deepseek::e8m0_exponent_bits_to_f32(scales[index]);
  }
};

template <typename ScaleReader>
__device__ __forceinline__ void deepseek_fp8_dual_encoded_matvec_body(
    const uint8_t *weights_a,
    ScaleReader scales_a,
    const uint8_t *weights_b,
    ScaleReader scales_b,
    const uint16_t *input,
    uint32_t input_dtype,
    float *output_a,
    float *output_b,
    uint32_t rows_a,
    uint32_t rows_b,
    uint32_t cols,
    uint32_t block_rows,
    uint32_t block_cols) {
  const uint32_t row_start = blockIdx.x * kDeepSeekFp8ProjectionRowTile;
  const uint32_t row_limit = rows_a > rows_b ? rows_a : rows_b;
  if (row_start >= row_limit) {
    return;
  }
  extern __shared__ float partial[];
  float sum_a[kDeepSeekFp8ProjectionRowTile] = {};
  float sum_b[kDeepSeekFp8ProjectionRowTile] = {};
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  const bool row_tile_shared_scale =
      deepseek_fp8_projection_row_tile_shares_scale(row_start, block_rows);
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float input_value = encoded_input_to_f32(input[col], input_dtype);
    const uint32_t shared_scale_idx =
        row_tile_shared_scale
            ? deepseek_fp8_projection_scale_idx(
                  row_start, col, scale_cols, block_rows, block_cols)
            : 0u;
    const float shared_scale_a =
        row_tile_shared_scale ? scales_a.get(shared_scale_idx) : 0.0f;
    const float shared_scale_b =
        row_tile_shared_scale ? scales_b.get(shared_scale_idx) : 0.0f;
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= row_limit) {
        continue;
      }
      const uint32_t scale_idx =
          row_tile_shared_scale
              ? shared_scale_idx
              : deepseek_fp8_projection_scale_idx(
                    row, col, scale_cols, block_rows, block_cols);
      const uint64_t row_base = static_cast<uint64_t>(row) * cols;
      if (row < rows_a) {
        const float weight_a =
            nerva::deepseek::f8_e4m3fn_bits_to_f32(weights_a[row_base + col]) *
            (row_tile_shared_scale ? shared_scale_a : scales_a.get(scale_idx));
        sum_a[row_tile] += weight_a * input_value;
      }
      if (row < rows_b) {
        const float weight_b =
            nerva::deepseek::f8_e4m3fn_bits_to_f32(weights_b[row_base + col]) *
            (row_tile_shared_scale ? shared_scale_b : scales_b.get(scale_idx));
        sum_b[row_tile] += weight_b * input_value;
      }
    }
  }
  float sum[kDeepSeekFp8ProjectionRowTile * 2u] = {};
  for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
       ++row_tile) {
    sum[row_tile] = sum_a[row_tile];
    sum[kDeepSeekFp8ProjectionRowTile + row_tile] = sum_b[row_tile];
  }
  deepseek_fp8_projection_reduce_slots(sum, partial);
  if (threadIdx.x == 0) {
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row < rows_a) {
        output_a[row] = partial[row_tile * blockDim.x];
      }
      if (row < rows_b) {
        output_b[row] =
            partial[(kDeepSeekFp8ProjectionRowTile + row_tile) * blockDim.x];
      }
    }
  }
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
  float reduce[1] = {sum};
  deepseek_fp8_projection_reduce_slots(reduce, partial);
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
  const uint32_t row_start = blockIdx.x * kDeepSeekFp8ProjectionRowTile;
  if (row_start >= rows) {
    return;
  }
  extern __shared__ float partial[];
  float sum[kDeepSeekFp8ProjectionRowTile] = {};
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  const bool row_tile_shared_scale =
      deepseek_fp8_projection_row_tile_shares_scale(row_start, block_rows);
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float input_value = encoded_input_to_f32(input[col], input_dtype);
    const uint32_t shared_scale_idx =
        row_tile_shared_scale
            ? deepseek_fp8_projection_scale_idx(
                  row_start, col, scale_cols, block_rows, block_cols)
            : 0u;
    const float shared_scale =
        row_tile_shared_scale ? scales[shared_scale_idx] : 0.0f;
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= rows) {
        continue;
      }
      const uint32_t scale_idx =
          row_tile_shared_scale
              ? shared_scale_idx
              : deepseek_fp8_projection_scale_idx(
                    row, col, scale_cols, block_rows, block_cols);
      const uint64_t row_base = static_cast<uint64_t>(row) * cols;
      const float weight =
          nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
          (row_tile_shared_scale ? shared_scale : scales[scale_idx]);
      sum[row_tile] += weight * input_value;
    }
  }
  deepseek_fp8_projection_reduce_slots(sum, partial);
  if (threadIdx.x == 0) {
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row < rows) {
        output[row] = partial[row_tile * blockDim.x];
      }
    }
  }
}

__global__ void deepseek_fp8_f32_scale_dual_encoded_matvec_kernel(
    const uint8_t *weights_a,
    const float *scales_a,
    const uint8_t *weights_b,
    const float *scales_b,
    const uint16_t *input,
    uint32_t input_dtype,
    float *output_a,
    float *output_b,
    uint32_t rows_a,
    uint32_t rows_b,
    uint32_t cols,
    uint32_t block_rows,
    uint32_t block_cols) {
  deepseek_fp8_dual_encoded_matvec_body(
      weights_a, DeepSeekFp8F32ScaleReader{scales_a}, weights_b,
      DeepSeekFp8F32ScaleReader{scales_b}, input, input_dtype, output_a,
      output_b, rows_a, rows_b, cols, block_rows, block_cols);
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
  const uint32_t row_start = blockIdx.x * kDeepSeekFp8ProjectionRowTile;
  if (row_start >= rows) {
    return;
  }
  extern __shared__ float partial[];
  float sum[kDeepSeekFp8ProjectionRowTile] = {};
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  const bool row_tile_shared_scale =
      deepseek_fp8_projection_row_tile_shares_scale(row_start, block_rows);
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float input_value = encoded_input_to_f32(input[col], input_dtype);
    const uint32_t shared_scale_idx =
        row_tile_shared_scale
            ? deepseek_fp8_projection_scale_idx(
                  row_start, col, scale_cols, block_rows, block_cols)
            : 0u;
    const float shared_scale =
        row_tile_shared_scale
            ? deepseek_fp8_projection_f32_scale_slot(scale_slots,
                                                     shared_scale_idx)
            : 0.0f;
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= rows) {
        continue;
      }
      const uint32_t scale_idx =
          row_tile_shared_scale
              ? shared_scale_idx
              : deepseek_fp8_projection_scale_idx(
                    row, col, scale_cols, block_rows, block_cols);
      const uint64_t row_base = static_cast<uint64_t>(row) * cols;
      const float weight =
          nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
          (row_tile_shared_scale
               ? shared_scale
               : deepseek_fp8_projection_f32_scale_slot(scale_slots,
                                                        scale_idx));
      sum[row_tile] += weight * input_value;
    }
  }
  deepseek_fp8_projection_reduce_slots(sum, partial);
  if (threadIdx.x == 0) {
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row < rows) {
        output[row] = partial[row_tile * blockDim.x];
      }
    }
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
  float sum[kDeepSeekFp8ProjectionRowTile * kDeepSeekFp8ProjectionTokenTile] =
      {};
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  const bool row_tile_shared_scale =
      deepseek_fp8_projection_row_tile_shares_scale(row_start, block_rows);
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
    const uint32_t shared_scale_idx =
        row_tile_shared_scale
            ? deepseek_fp8_projection_scale_idx(
                  row_start, col, scale_cols, block_rows, block_cols)
            : 0u;
    const float shared_scale =
        row_tile_shared_scale ? scales[shared_scale_idx] : 0.0f;
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= rows) {
        continue;
      }
      const uint32_t scale_idx =
          row_tile_shared_scale
              ? shared_scale_idx
              : deepseek_fp8_projection_scale_idx(
                    row, col, scale_cols, block_rows, block_cols);
      const uint64_t row_base = static_cast<uint64_t>(row) * cols;
      const float weight =
          nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
          (row_tile_shared_scale ? shared_scale : scales[scale_idx]);
      for (uint32_t tile = 0; tile < kDeepSeekFp8ProjectionTokenTile; ++tile) {
        sum[row_tile * kDeepSeekFp8ProjectionTokenTile + tile] +=
            weight * input_value[tile];
      }
    }
  }
  deepseek_fp8_projection_reduce_slots(sum, partial);
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
  const uint32_t row_start = blockIdx.x * kDeepSeekFp8ProjectionRowTile;
  if (row_start >= rows) {
    return;
  }
  extern __shared__ float partial[];
  float sum[kDeepSeekFp8ProjectionRowTile] = {};
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  const bool row_tile_shared_scale =
      deepseek_fp8_projection_row_tile_shares_scale(row_start, block_rows);
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float input_value = encoded_input_to_f32(input[col], input_dtype);
    const uint32_t shared_scale_idx =
        row_tile_shared_scale
            ? deepseek_fp8_projection_scale_idx(
                  row_start, col, scale_cols, block_rows, block_cols)
            : 0u;
    const float shared_scale =
        row_tile_shared_scale
            ? nerva::deepseek::e8m0_exponent_bits_to_f32(
                  scales[shared_scale_idx])
            : 0.0f;
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= rows) {
        continue;
      }
      const uint32_t scale_idx =
          row_tile_shared_scale
              ? shared_scale_idx
              : deepseek_fp8_projection_scale_idx(
                    row, col, scale_cols, block_rows, block_cols);
      const uint64_t row_base = static_cast<uint64_t>(row) * cols;
      const float weight =
          nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
          (row_tile_shared_scale
               ? shared_scale
               : nerva::deepseek::e8m0_exponent_bits_to_f32(
                     scales[scale_idx]));
      sum[row_tile] += weight * input_value;
    }
  }
  deepseek_fp8_projection_reduce_slots(sum, partial);
  if (threadIdx.x == 0) {
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row < rows) {
        output[row] = partial[row_tile * blockDim.x];
      }
    }
  }
}

__global__ void deepseek_fp8_e8m0_scale_dual_encoded_matvec_kernel(
    const uint8_t *weights_a,
    const uint8_t *scales_a,
    const uint8_t *weights_b,
    const uint8_t *scales_b,
    const uint16_t *input,
    uint32_t input_dtype,
    float *output_a,
    float *output_b,
    uint32_t rows_a,
    uint32_t rows_b,
    uint32_t cols,
    uint32_t block_rows,
    uint32_t block_cols) {
  deepseek_fp8_dual_encoded_matvec_body(
      weights_a, DeepSeekFp8E8M0ScaleReader{scales_a}, weights_b,
      DeepSeekFp8E8M0ScaleReader{scales_b}, input, input_dtype, output_a,
      output_b, rows_a, rows_b, cols, block_rows, block_cols);
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
  float sum[kDeepSeekFp8ProjectionRowTile * kDeepSeekFp8ProjectionTokenTile] =
      {};
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  const bool row_tile_shared_scale =
      deepseek_fp8_projection_row_tile_shares_scale(row_start, block_rows);
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
    const uint32_t shared_scale_idx =
        row_tile_shared_scale
            ? deepseek_fp8_projection_scale_idx(
                  row_start, col, scale_cols, block_rows, block_cols)
            : 0u;
    const float shared_scale =
        row_tile_shared_scale
            ? nerva::deepseek::e8m0_exponent_bits_to_f32(
                  scales[shared_scale_idx])
            : 0.0f;
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= rows) {
        continue;
      }
      const uint32_t scale_idx =
          row_tile_shared_scale
              ? shared_scale_idx
              : deepseek_fp8_projection_scale_idx(
                    row, col, scale_cols, block_rows, block_cols);
      const uint64_t row_base = static_cast<uint64_t>(row) * cols;
      const float weight =
          nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
          (row_tile_shared_scale
               ? shared_scale
               : nerva::deepseek::e8m0_exponent_bits_to_f32(
                     scales[scale_idx]));
      for (uint32_t tile = 0; tile < kDeepSeekFp8ProjectionTokenTile; ++tile) {
        sum[row_tile * kDeepSeekFp8ProjectionTokenTile + tile] +=
            weight * input_value[tile];
      }
    }
  }
  deepseek_fp8_projection_reduce_slots(sum, partial);
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
  const uint32_t row_start = blockIdx.x * kDeepSeekFp8ProjectionRowTile;
  if (row_start >= rows) {
    return;
  }
  extern __shared__ float partial[];
  float sum[kDeepSeekFp8ProjectionRowTile] = {};
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);
  const bool row_tile_shared_scale =
      deepseek_fp8_projection_row_tile_shares_scale(row_start, block_rows);
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float input_value = input[col];
    const uint32_t shared_scale_idx =
        row_tile_shared_scale
            ? deepseek_fp8_projection_scale_idx(
                  row_start, col, scale_cols, block_rows, block_cols)
            : 0u;
    const float shared_scale =
        row_tile_shared_scale
            ? nerva::deepseek::e8m0_exponent_bits_to_f32(
                  scales[shared_scale_idx])
            : 0.0f;
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= rows) {
        continue;
      }
      const uint32_t scale_idx =
          row_tile_shared_scale
              ? shared_scale_idx
              : deepseek_fp8_projection_scale_idx(
                    row, col, scale_cols, block_rows, block_cols);
      const uint64_t row_base = static_cast<uint64_t>(row) * cols;
      const float weight =
          nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
          (row_tile_shared_scale
               ? shared_scale
               : nerva::deepseek::e8m0_exponent_bits_to_f32(
                     scales[scale_idx]));
      sum[row_tile] += weight * input_value;
    }
  }
  deepseek_fp8_projection_reduce_slots(sum, partial);
  if (threadIdx.x == 0) {
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row < rows) {
        output[row] = partial[row_tile * blockDim.x];
      }
    }
  }
}

__global__ void deepseek_fp8_e8m0_scale_grouped_matvec_kernel(
    const uint8_t *weights,
    const uint8_t *scales,
    const float *input,
    float *output,
    uint32_t groups,
    uint32_t rows_per_group,
    uint32_t cols_per_group,
    uint32_t block_rows,
    uint32_t block_cols) {
  const uint32_t row_start = blockIdx.x * kDeepSeekFp8ProjectionRowTile;
  const uint32_t total_rows = groups * rows_per_group;
  if (row_start >= total_rows) {
    return;
  }
  extern __shared__ float partial[];
  float sum[kDeepSeekFp8ProjectionRowTile] = {};
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols_per_group, block_cols);
  const bool row_tile_shared_scale =
      deepseek_fp8_projection_row_tile_shares_scale(row_start, block_rows);
  for (uint32_t col = threadIdx.x; col < cols_per_group; col += blockDim.x) {
    const uint32_t shared_scale_idx =
        row_tile_shared_scale
            ? deepseek_fp8_projection_scale_idx(
                  row_start, col, scale_cols, block_rows, block_cols)
            : 0u;
    const float shared_scale =
        row_tile_shared_scale
            ? nerva::deepseek::e8m0_exponent_bits_to_f32(
                  scales[shared_scale_idx])
            : 0.0f;
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= total_rows) {
        continue;
      }
      const uint32_t group = row / rows_per_group;
      const uint64_t input_base =
          static_cast<uint64_t>(group) * cols_per_group;
      const uint64_t row_base = static_cast<uint64_t>(row) * cols_per_group;
      const uint32_t scale_idx =
          row_tile_shared_scale
              ? shared_scale_idx
              : deepseek_fp8_projection_scale_idx(
                    row, col, scale_cols, block_rows, block_cols);
      const float weight =
          nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[row_base + col]) *
          (row_tile_shared_scale
               ? shared_scale
               : nerva::deepseek::e8m0_exponent_bits_to_f32(
                     scales[scale_idx]));
      sum[row_tile] += weight * input[input_base + col];
    }
  }
  deepseek_fp8_projection_reduce_slots(sum, partial);
  if (threadIdx.x == 0) {
    for (uint32_t row_tile = 0; row_tile < kDeepSeekFp8ProjectionRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row < total_rows) {
        output[row] = partial[row_tile * blockDim.x];
      }
    }
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

cudaError_t launch_deepseek_fp8_e8m0_scale_grouped_matvec(
    cudaStream_t stream, const uint8_t *weights, const uint8_t *scales,
    const float *input, uint32_t groups, uint32_t rows_per_group,
    uint32_t cols_per_group, uint32_t block_rows, uint32_t block_cols,
    float *output) {
  if (weights == nullptr || scales == nullptr || input == nullptr ||
      output == nullptr || groups == 0 || rows_per_group == 0 ||
      cols_per_group == 0 || block_rows == 0 || block_cols == 0 ||
      groups > (0xffffffffu / rows_per_group)) {
    return cudaErrorInvalidValue;
  }
  constexpr uint32_t threads = 256;
  const uint32_t total_rows = groups * rows_per_group;
  const size_t shared_bytes =
      threads * kDeepSeekFp8ProjectionRowTile * sizeof(float);
  const uint32_t blocks =
      (total_rows + kDeepSeekFp8ProjectionRowTile - 1) /
      kDeepSeekFp8ProjectionRowTile;
  deepseek_fp8_e8m0_scale_grouped_matvec_kernel<<<blocks,
                                                   threads,
                                                   shared_bytes,
                                                   stream>>>(
      weights, scales, input, output, groups, rows_per_group, cols_per_group,
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
  const size_t shared_bytes =
      threads * kDeepSeekFp8ProjectionRowTile * sizeof(float);
  const uint32_t blocks =
      (rows + kDeepSeekFp8ProjectionRowTile - 1) /
      kDeepSeekFp8ProjectionRowTile;
  deepseek_fp8_f32_scale_encoded_matvec_kernel<<<blocks,
                                                 threads,
                                                 shared_bytes,
                                                 stream>>>(
      weights, scales, input, input_dtype, output, rows, cols, block_rows,
      block_cols);
  return cudaGetLastError();
}

cudaError_t launch_deepseek_fp8_f32_scale_dual_encoded_matvec(
    cudaStream_t stream, const uint8_t *weights_a, const float *scales_a,
    const uint8_t *weights_b, const float *scales_b, const uint16_t *input,
    uint32_t input_dtype, uint32_t rows, uint32_t cols, uint32_t block_rows,
    uint32_t block_cols, float *output_a, float *output_b) {
  return launch_deepseek_fp8_f32_scale_dual_encoded_matvec_varrows(
      stream, weights_a, scales_a, weights_b, scales_b, input, input_dtype,
      rows, rows, cols, block_rows, block_cols, output_a, output_b);
}

cudaError_t launch_deepseek_fp8_f32_scale_dual_encoded_matvec_varrows(
    cudaStream_t stream, const uint8_t *weights_a, const float *scales_a,
    const uint8_t *weights_b, const float *scales_b, const uint16_t *input,
    uint32_t input_dtype, uint32_t rows_a, uint32_t rows_b, uint32_t cols,
    uint32_t block_rows, uint32_t block_cols, float *output_a,
    float *output_b) {
  if (weights_a == nullptr || scales_a == nullptr || weights_b == nullptr ||
      scales_b == nullptr || input == nullptr || output_a == nullptr ||
      output_b == nullptr || rows_a == 0 || rows_b == 0 || cols == 0 ||
      block_rows == 0 || block_cols == 0 || input_dtype > kDTypeBF16) {
    return cudaErrorInvalidValue;
  }
  constexpr uint32_t threads = 256;
  const uint32_t row_limit = rows_a > rows_b ? rows_a : rows_b;
  const size_t shared_bytes =
      threads * kDeepSeekFp8ProjectionRowTile * 2u * sizeof(float);
  const uint32_t blocks =
      (row_limit + kDeepSeekFp8ProjectionRowTile - 1) /
      kDeepSeekFp8ProjectionRowTile;
  deepseek_fp8_f32_scale_dual_encoded_matvec_kernel<<<blocks,
                                                      threads,
                                                      shared_bytes,
                                                      stream>>>(
      weights_a, scales_a, weights_b, scales_b, input, input_dtype, output_a,
      output_b, rows_a, rows_b, cols, block_rows, block_cols);
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
  const size_t shared_bytes =
      threads * kDeepSeekFp8ProjectionRowTile * sizeof(float);
  const uint32_t blocks =
      (rows + kDeepSeekFp8ProjectionRowTile - 1) /
      kDeepSeekFp8ProjectionRowTile;
  deepseek_fp8_f32_scale_slots_encoded_matvec_kernel<<<blocks,
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
  const size_t shared_bytes =
      threads * kDeepSeekFp8ProjectionRowTile * sizeof(float);
  const uint32_t blocks =
      (rows + kDeepSeekFp8ProjectionRowTile - 1) /
      kDeepSeekFp8ProjectionRowTile;
  deepseek_fp8_e8m0_scale_encoded_matvec_kernel<<<blocks,
                                                   threads,
                                                   shared_bytes,
                                                   stream>>>(
      weights, scales, input, input_dtype, output, rows, cols, block_rows,
      block_cols);
  return cudaGetLastError();
}

cudaError_t launch_deepseek_fp8_e8m0_scale_dual_encoded_matvec(
    cudaStream_t stream, const uint8_t *weights_a, const uint8_t *scales_a,
    const uint8_t *weights_b, const uint8_t *scales_b, const uint16_t *input,
    uint32_t input_dtype, uint32_t rows, uint32_t cols, uint32_t block_rows,
    uint32_t block_cols, float *output_a, float *output_b) {
  return launch_deepseek_fp8_e8m0_scale_dual_encoded_matvec_varrows(
      stream, weights_a, scales_a, weights_b, scales_b, input, input_dtype,
      rows, rows, cols, block_rows, block_cols, output_a, output_b);
}

cudaError_t launch_deepseek_fp8_e8m0_scale_dual_encoded_matvec_varrows(
    cudaStream_t stream, const uint8_t *weights_a, const uint8_t *scales_a,
    const uint8_t *weights_b, const uint8_t *scales_b, const uint16_t *input,
    uint32_t input_dtype, uint32_t rows_a, uint32_t rows_b, uint32_t cols,
    uint32_t block_rows, uint32_t block_cols, float *output_a,
    float *output_b) {
  if (weights_a == nullptr || scales_a == nullptr || weights_b == nullptr ||
      scales_b == nullptr || input == nullptr || output_a == nullptr ||
      output_b == nullptr || rows_a == 0 || rows_b == 0 || cols == 0 ||
      block_rows == 0 || block_cols == 0 || input_dtype > kDTypeBF16) {
    return cudaErrorInvalidValue;
  }
  constexpr uint32_t threads = 256;
  const uint32_t row_limit = rows_a > rows_b ? rows_a : rows_b;
  const size_t shared_bytes =
      threads * kDeepSeekFp8ProjectionRowTile * 2u * sizeof(float);
  const uint32_t blocks =
      (row_limit + kDeepSeekFp8ProjectionRowTile - 1) /
      kDeepSeekFp8ProjectionRowTile;
  deepseek_fp8_e8m0_scale_dual_encoded_matvec_kernel<<<blocks,
                                                       threads,
                                                       shared_bytes,
                                                       stream>>>(
      weights_a, scales_a, weights_b, scales_b, input, input_dtype, output_a,
      output_b, rows_a, rows_b, cols, block_rows, block_cols);
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
  const size_t shared_bytes =
      threads * kDeepSeekFp8ProjectionRowTile * sizeof(float);
  const uint32_t blocks =
      (rows + kDeepSeekFp8ProjectionRowTile - 1) /
      kDeepSeekFp8ProjectionRowTile;
  deepseek_fp8_e8m0_scale_matvec_kernel<<<blocks,
                                          threads,
                                          shared_bytes,
                                          stream>>>(
      weights, scales, input, output, rows, cols, block_rows, block_cols);
  return cudaGetLastError();
}
