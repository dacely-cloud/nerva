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

constexpr uint32_t kDeepSeekFp8GemmTileM = 64;   // tokens per block
constexpr uint32_t kDeepSeekFp8GemmTileN = 64;   // weight rows per block
constexpr uint32_t kDeepSeekFp8GemmTileK = 64;   // staged K slice
constexpr uint32_t kDeepSeekFp8GemmThreads = 256;

// Shared-memory tiled GEMM: output[token][row] = sum_col dequant(W[row][col])
// * X[token][col]. Each 256-thread block computes a 64(token) x 64(row) tile
// with 4x4 register blocking per thread; fp8 weights are dequantized (scale
// applied, identical dequant semantics) while being staged into shared
// memory. The K-summation order is reassociated relative to the strided
// scalar kernels; accumulation stays in f32.
template <typename ScaleReader>
__device__ __forceinline__ void deepseek_fp8_gemm_tokens_body(
    const uint8_t *weights,
    ScaleReader scales,
    const uint16_t *input,
    uint32_t input_dtype,
    float *output,
    uint32_t rows,
    uint32_t cols,
    uint32_t tokens,
    uint32_t block_rows,
    uint32_t block_cols) {
  constexpr uint32_t TM = kDeepSeekFp8GemmTileM;
  constexpr uint32_t TN = kDeepSeekFp8GemmTileN;
  constexpr uint32_t TK = kDeepSeekFp8GemmTileK;
  __shared__ float x_tile[TK][TM + 1];
  __shared__ float w_tile[TK][TN + 1];

  const uint32_t row_block = blockIdx.x * TN;
  const uint32_t token_block = blockIdx.y * TM;
  if (row_block >= rows || token_block >= tokens) {
    return;
  }
  const uint32_t tid = threadIdx.x;
  const uint32_t load_m = tid >> 2u;         // 0..63: token/row within tile
  const uint32_t load_k = (tid & 3u) * 16u;  // 0,16,32,48: K strip base
  const uint32_t thread_token = tid >> 4u;   // 0..15
  const uint32_t thread_row = tid & 15u;     // 0..15
  const uint32_t scale_cols =
      deepseek_fp8_projection_scale_cols(cols, block_cols);

  float acc[4][4] = {};
  for (uint32_t k0 = 0; k0 < cols; k0 += TK) {
    // Stage the activation strip (16 consecutive encoded values).
    {
      const uint32_t token = token_block + load_m;
      const uint32_t kb = k0 + load_k;
      float staged[16];
      if (token < tokens && kb + 16u <= cols) {
        const uint16_t *src = input + static_cast<uint64_t>(token) * cols + kb;
        if ((reinterpret_cast<uintptr_t>(src) & 0xfu) == 0u) {
          const uint4 *vec = reinterpret_cast<const uint4 *>(src);
          const uint4 lo = vec[0];
          const uint4 hi = vec[1];
          const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w,
                                     hi.x, hi.y, hi.z, hi.w};
#pragma unroll
          for (uint32_t word = 0; word < 8u; ++word) {
            staged[word * 2u] = encoded_input_to_f32(
                static_cast<uint16_t>(words[word] & 0xffffu), input_dtype);
            staged[word * 2u + 1u] = encoded_input_to_f32(
                static_cast<uint16_t>(words[word] >> 16u), input_dtype);
          }
        } else {
#pragma unroll
          for (uint32_t i = 0; i < 16u; ++i) {
            staged[i] = encoded_input_to_f32(src[i], input_dtype);
          }
        }
      } else {
#pragma unroll
        for (uint32_t i = 0; i < 16u; ++i) {
          const uint32_t k = kb + i;
          staged[i] =
              (token < tokens && k < cols)
                  ? encoded_input_to_f32(
                        input[static_cast<uint64_t>(token) * cols + k],
                        input_dtype)
                  : 0.0f;
        }
      }
#pragma unroll
      for (uint32_t i = 0; i < 16u; ++i) {
        x_tile[load_k + i][load_m] = staged[i];
      }
    }
    // Stage the dequantized weight strip (16 consecutive fp8 values).
    {
      const uint32_t row = row_block + load_m;
      const uint32_t kb = k0 + load_k;
      float staged[16];
      if (row < rows && kb + 16u <= cols) {
        const uint8_t *src = weights + static_cast<uint64_t>(row) * cols + kb;
        if (kb / block_cols == (kb + 15u) / block_cols) {
          const float scale = scales.get(deepseek_fp8_projection_scale_idx(
              row, kb, scale_cols, block_rows, block_cols));
          if ((reinterpret_cast<uintptr_t>(src) & 0xfu) == 0u) {
            const uint4 raw = *reinterpret_cast<const uint4 *>(src);
            const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
            for (uint32_t word = 0; word < 4u; ++word) {
#pragma unroll
              for (uint32_t byte = 0; byte < 4u; ++byte) {
                staged[word * 4u + byte] =
                    nerva::deepseek::f8_e4m3fn_bits_to_f32(static_cast<uint8_t>(
                        (words[word] >> (byte * 8u)) & 0xffu)) *
                    scale;
              }
            }
          } else {
#pragma unroll
            for (uint32_t i = 0; i < 16u; ++i) {
              staged[i] =
                  nerva::deepseek::f8_e4m3fn_bits_to_f32(src[i]) * scale;
            }
          }
        } else {
#pragma unroll
          for (uint32_t i = 0; i < 16u; ++i) {
            staged[i] = nerva::deepseek::f8_e4m3fn_bits_to_f32(src[i]) *
                        scales.get(deepseek_fp8_projection_scale_idx(
                            row, kb + i, scale_cols, block_rows, block_cols));
          }
        }
      } else {
#pragma unroll
        for (uint32_t i = 0; i < 16u; ++i) {
          const uint32_t k = kb + i;
          staged[i] =
              (row < rows && k < cols)
                  ? nerva::deepseek::f8_e4m3fn_bits_to_f32(
                        weights[static_cast<uint64_t>(row) * cols + k]) *
                        scales.get(deepseek_fp8_projection_scale_idx(
                            row, k, scale_cols, block_rows, block_cols))
                  : 0.0f;
        }
      }
#pragma unroll
      for (uint32_t i = 0; i < 16u; ++i) {
        w_tile[load_k + i][load_m] = staged[i];
      }
    }
    __syncthreads();
#pragma unroll 8
    for (uint32_t k = 0; k < TK; ++k) {
      float a[4];
      float b[4];
#pragma unroll
      for (uint32_t i = 0; i < 4u; ++i) {
        a[i] = x_tile[k][thread_token * 4u + i];
        b[i] = w_tile[k][thread_row * 4u + i];
      }
#pragma unroll
      for (uint32_t i = 0; i < 4u; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
          acc[i][j] += a[i] * b[j];
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t i = 0; i < 4u; ++i) {
    const uint32_t token = token_block + thread_token * 4u + i;
    if (token >= tokens) continue;
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
      const uint32_t row = row_block + thread_row * 4u + j;
      if (row < rows) {
        output[static_cast<uint64_t>(token) * rows + row] = acc[i][j];
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
  deepseek_fp8_gemm_tokens_body(weights, DeepSeekFp8F32ScaleReader{scales},
                                input, input_dtype, output, rows, cols,
                                tokens, block_rows, block_cols);
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
  deepseek_fp8_gemm_tokens_body(weights, DeepSeekFp8E8M0ScaleReader{scales},
                                input, input_dtype, output, rows, cols,
                                tokens, block_rows, block_cols);
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

__global__ void deepseek_bf16_grouped_matvec_kernel(
    const uint16_t *weights, const float *input, float *output,
    uint32_t groups, uint32_t rows_per_group, uint32_t cols_per_group) {
  const uint32_t row_start = blockIdx.x * kDeepSeekFp8ProjectionRowTile;
  const uint32_t total_rows = groups * rows_per_group;
  if (row_start >= total_rows) {
    return;
  }
  extern __shared__ float partial[];
  float sum[kDeepSeekFp8ProjectionRowTile] = {};
  for (uint32_t col = threadIdx.x; col < cols_per_group; col += blockDim.x) {
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
      const float weight = encoded_input_to_f32(weights[row_base + col],
                                                kDTypeBF16);
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

cudaError_t launch_deepseek_bf16_grouped_matvec(
    cudaStream_t stream, const uint16_t *weights, const float *input,
    uint32_t groups, uint32_t rows_per_group, uint32_t cols_per_group,
    float *output) {
  if (weights == nullptr || input == nullptr || output == nullptr ||
      groups == 0 || rows_per_group == 0 || cols_per_group == 0 ||
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
  deepseek_bf16_grouped_matvec_kernel<<<blocks, threads, shared_bytes,
                                        stream>>>(
      weights, input, output, groups, rows_per_group, cols_per_group);
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
  const dim3 grid(
      (rows + kDeepSeekFp8GemmTileN - 1) / kDeepSeekFp8GemmTileN,
      (tokens + kDeepSeekFp8GemmTileM - 1) / kDeepSeekFp8GemmTileM, 1);
  deepseek_fp8_f32_scale_encoded_gemm_tokens_kernel<<<grid,
                                                      kDeepSeekFp8GemmThreads,
                                                      0,
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
  const dim3 grid(
      (rows + kDeepSeekFp8GemmTileN - 1) / kDeepSeekFp8GemmTileN,
      (tokens + kDeepSeekFp8GemmTileM - 1) / kDeepSeekFp8GemmTileM, 1);
  deepseek_fp8_e8m0_scale_encoded_gemm_tokens_kernel<<<
      grid, kDeepSeekFp8GemmThreads, 0, stream>>>(
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
