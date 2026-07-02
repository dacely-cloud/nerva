#include "../../deepseek/deepseek_quant.cuh"

#include <cuda_bf16.h>
#include <mma.h>

// ---------------------------------------------------------------------------
// Unified DeepSeek session projection GEMM (tensor-core bf16 MMA).
//
// One kernel family computes C[M x N] = A[M x K] * W[N x K]^T for every
// session-path fp8/bf16 projection, serving both decode (M == 1) and batched
// prefill (M == chunk_tokens) so that prefill and decode numerics are
// identical by construction.
//
// Numeric spec (identical for every M):
//   A[m][k]  activation staged as bf16:
//              - bf16-encoded inputs are used bit-exactly,
//              - fp16-encoded and f32 inputs are converted with
//                __float2bfloat16_rn.
//   W[n][k]  fp8-e4m3fn weight dequantized and staged as
//              __float2bfloat16_rn(f8_e4m3fn_bits_to_f32(bits) * scale(n, k))
//            where scale comes from the f32 / f32-slot / e8m0 block-scale
//            table (bf16 weights are used bit-exactly, no scale).
//   C[m][n]  f32, accumulated by bf16 HMMA (wmma 16x16x16, f32 accumulator).
//
// K reduction order (part of the numeric spec, never depends on M, N tile
// size, or grid shape):
//   - K is consumed in "super-chunks" of kDeepSeekGemmSuperK (256) columns,
//     ascending; the tail is zero-padded (adding +0.0f is exact).
//   - Each super-chunk is divided into kDeepSeekGemmSplitK (4) contiguous
//     sub-chunks of 64. Sub-chunk s of every super-chunk accumulates into
//     partial P_s: within a sub-chunk the four 16-wide HMMA steps run in
//     ascending k order into a per-warp register fragment accumulator.
//   - After the K loop the four partials are combined in the fixed order
//     ((P_0 + P_1) + P_2) + P_3.
//   There is no cross-block K split and no atomics; every output element is
//   produced by exactly one block, and each partial by exactly one warp's
//   register accumulator.
//
// M-invariance argument: each HMMA output element is an independent dot
// product of one A row-fragment and one W column with a fixed internal
// accumulation order per instruction; rows never mix. Zero-padded token rows
// (M-tile padding) therefore cannot perturb real rows, and a token's row
// index inside the 16-row fragment does not change its result. Since the
// k-tile sequence, sub-chunk split, and combine order above are compile-time
// constants, C[m][n] depends only on A[m][*] and W[n][*] -- identical for
// decode (M = 1) and prefill (M = chunk) by construction.
// ---------------------------------------------------------------------------

constexpr uint32_t kDeepSeekGemmTileM = 16;    // token rows per output tile
constexpr uint32_t kDeepSeekGemmSuperK = 256;  // staged K per iteration (spec)
constexpr uint32_t kDeepSeekGemmSplitK = 4;    // fixed warp K split (spec)
constexpr uint32_t kDeepSeekGemmSubK =
    kDeepSeekGemmSuperK / kDeepSeekGemmSplitK;  // 64
constexpr uint32_t kDeepSeekGemmMmaK = 16;
constexpr uint32_t kDeepSeekGemmStageStride =
    kDeepSeekGemmSuperK + 8u;  // padded bf16 row stride (multiple of 8)
constexpr uint32_t kDeepSeekGemmInputF32 = 2;  // input_kind: raw f32 input

__device__ __forceinline__ __nv_bfloat16 deepseek_gemm_zero_bf16() {
  return __ushort_as_bfloat16(static_cast<unsigned short>(0));
}

__device__ __forceinline__ __nv_bfloat16 deepseek_gemm_load_activation(
    const void *input, uint64_t index, uint32_t input_kind) {
  if (input_kind == kDTypeBF16) {
    // bf16-encoded activations are used bit-exactly.
    return __ushort_as_bfloat16(static_cast<const uint16_t *>(input)[index]);
  }
  if (input_kind == kDeepSeekGemmInputF32) {
    return __float2bfloat16_rn(static_cast<const float *>(input)[index]);
  }
  return __float2bfloat16_rn(encoded_input_to_f32(
      static_cast<const uint16_t *>(input)[index], input_kind));
}

struct DeepSeekFp8F32ScaleReader {
  const float *scales;
  __device__ __forceinline__ float get(uint32_t index) const {
    return scales[index];
  }
};

// f32 scales stored as unaligned pairs of u16 arena slots.
struct DeepSeekFp8SlotScaleReader {
  const uint16_t *slots;
  __device__ __forceinline__ float get(uint32_t index) const {
    const uint32_t lo = static_cast<uint32_t>(slots[index * 2u]);
    const uint32_t hi = static_cast<uint32_t>(slots[index * 2u + 1u]);
    return __uint_as_float(lo | (hi << 16u));
  }
};

struct DeepSeekFp8E8M0ScaleReader {
  const uint8_t *scales;
  __device__ __forceinline__ float get(uint32_t index) const {
    return nerva::deepseek::e8m0_exponent_bits_to_f32(scales[index]);
  }
};

// Raw register buffer for one prefetched 16-wide weight strip. Weight
// streaming is register double-buffered: the next super-chunk's raw bytes are
// fetched (pure loads) before the current chunk's MMA work so the DRAM
// latency overlaps compute. Prefetch order does not affect numerics; the
// committed values follow the exact dequant spec above.
union DeepSeekGemmRawStrip {
  uint4 vec[2];
  uint8_t bytes[32];
  uint16_t halves[16];
};

// Stages one 16-wide k strip of one weight row: fetch() issues the global
// loads into registers, commit() dequantizes into shared memory as bf16.
// Zero-fills past the end of the row.
template <typename ScaleReader>
struct DeepSeekFp8WeightStager {
  const uint8_t *weights;
  ScaleReader scales;
  uint32_t block_rows;
  uint32_t block_cols;
  uint32_t scale_cols;

  __device__ __forceinline__ void fetch(DeepSeekGemmRawStrip &raw,
                                        uint32_t row, uint32_t k,
                                        uint32_t cols) const {
    const uint8_t *src = weights + static_cast<uint64_t>(row) * cols + k;
    if (k + 16u <= cols &&
        (reinterpret_cast<uintptr_t>(src) & 0xfu) == 0u) {
      raw.vec[0] = *reinterpret_cast<const uint4 *>(src);
      return;
    }
#pragma unroll
    for (uint32_t i = 0; i < 16u; ++i) {
      raw.bytes[i] = k + i < cols ? src[i] : static_cast<uint8_t>(0);
    }
  }

  __device__ __forceinline__ void commit(const DeepSeekGemmRawStrip &raw,
                                         __nv_bfloat16 *dst, uint32_t row,
                                         uint32_t k, uint32_t cols) const {
    const uint32_t scale_row = (row / block_rows) * scale_cols;
    if (k + 16u <= cols && (k / block_cols) == ((k + 15u) / block_cols)) {
      const float scale = scales.get(scale_row + k / block_cols);
#pragma unroll
      for (uint32_t i = 0; i < 16u; ++i) {
        dst[i] = __float2bfloat16_rn(
            nerva::deepseek::f8_e4m3fn_bits_to_f32(raw.bytes[i]) * scale);
      }
      return;
    }
#pragma unroll
    for (uint32_t i = 0; i < 16u; ++i) {
      const uint32_t kk = k + i;
      dst[i] = kk < cols
                   ? __float2bfloat16_rn(
                         nerva::deepseek::f8_e4m3fn_bits_to_f32(raw.bytes[i]) *
                         scales.get(scale_row + kk / block_cols))
                   : deepseek_gemm_zero_bf16();
    }
  }
};

// bf16 weights are staged bit-exactly.
struct DeepSeekBf16WeightStager {
  const uint16_t *weights;

  __device__ __forceinline__ void fetch(DeepSeekGemmRawStrip &raw,
                                        uint32_t row, uint32_t k,
                                        uint32_t cols) const {
    const uint16_t *src = weights + static_cast<uint64_t>(row) * cols + k;
    if (k + 16u <= cols &&
        (reinterpret_cast<uintptr_t>(src) & 0xfu) == 0u) {
      raw.vec[0] = reinterpret_cast<const uint4 *>(src)[0];
      raw.vec[1] = reinterpret_cast<const uint4 *>(src)[1];
      return;
    }
#pragma unroll
    for (uint32_t i = 0; i < 16u; ++i) {
      raw.halves[i] = k + i < cols ? src[i] : static_cast<uint16_t>(0);
    }
  }

  __device__ __forceinline__ void commit(const DeepSeekGemmRawStrip &raw,
                                         __nv_bfloat16 *dst, uint32_t row,
                                         uint32_t k, uint32_t cols) const {
    (void)row;
    (void)k;
    (void)cols;
#pragma unroll
    for (uint32_t i = 0; i < 16u; ++i) {
      dst[i] = __ushort_as_bfloat16(raw.halves[i]);
    }
  }
};

// grid: x = M tiles (fastest varying, so blocks sharing a W tile are
//           scheduled together and W stays in L2 during prefill),
//       y = N tiles (tile_n = warps_n * 16),
//       z = weight groups (grouped o_proj); group g uses weight rows
//           [g*rows, (g+1)*rows), input segment g*cols, output segment
//           g*rows. Non-grouped launches use gridDim.z == 1.
template <typename WeightStager>
__global__ void deepseek_session_gemm_kernel(WeightStager weights,
                                             const void *input,
                                             uint32_t input_kind,
                                             float *output,
                                             uint32_t rows,
                                             uint32_t cols,
                                             uint32_t tokens,
                                             uint32_t warps_n) {
  // blockDim.x may exceed the 32 * kDeepSeekGemmSplitK * warps_n compute
  // threads; the extra warps only help stage tiles (more loads in flight for
  // narrow-N launches). Thread role does not affect numerics.
  const uint32_t tile_n = warps_n * 16u;
  const uint32_t token_base = blockIdx.x * kDeepSeekGemmTileM;
  const uint32_t row_base = blockIdx.y * tile_n;
  const uint32_t group = blockIdx.z;
  const uint32_t groups = gridDim.z;
  if (token_base >= tokens || row_base >= rows) {
    return;
  }

  extern __shared__ unsigned char deepseek_gemm_smem[];
  __nv_bfloat16 *a_tile =
      reinterpret_cast<__nv_bfloat16 *>(deepseek_gemm_smem);
  __nv_bfloat16 *w_tile =
      a_tile + kDeepSeekGemmTileM * kDeepSeekGemmStageStride;
  const uint32_t tid = threadIdx.x;
  const uint32_t group_row_base = group * rows;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  namespace wmma = nvcuda::wmma;
  const uint32_t warp = tid >> 5u;
  const uint32_t warp_k = warp % kDeepSeekGemmSplitK;
  const uint32_t warp_n = warp / kDeepSeekGemmSplitK;
  const bool compute_warp = warp < kDeepSeekGemmSplitK * warps_n;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);
#else
  // Pre-sm_80 scalar fallback: same staging, same k-chunk/split/combine
  // structure (self-consistent across M on that arch), FFMA instead of HMMA.
  float acc_scalar[2][kDeepSeekGemmSplitK] = {};
#endif

  // Each thread owns one or two weight strips per super-chunk
  // (tile_n * 16 strips spread over blockDim.x threads); their raw bytes are
  // prefetched one super-chunk ahead into registers.
  const uint32_t w_strips_per_thread =
      (tile_n * 16u + blockDim.x - 1u) / blockDim.x;
  DeepSeekGemmRawStrip w_raw[2];
  const uint32_t w_strip[2] = {tid, tid + blockDim.x};
  auto fetch_weights = [&](uint32_t k0) {
#pragma unroll
    for (uint32_t s = 0; s < 2u; ++s) {
      if (s >= w_strips_per_thread || w_strip[s] >= tile_n * 16u) {
        break;
      }
      const uint32_t n = w_strip[s] >> 4u;
      const uint32_t k = k0 + (w_strip[s] & 15u) * 16u;
      const uint32_t row = row_base + n;
      if (row < rows && k < cols) {
        weights.fetch(w_raw[s], group_row_base + row, k, cols);
      }
    }
  };
  fetch_weights(0);

  for (uint32_t k0 = 0; k0 < cols; k0 += kDeepSeekGemmSuperK) {
    // Stage activations: kDeepSeekGemmTileM x kDeepSeekGemmSuperK bf16.
    constexpr uint32_t kAStrips = kDeepSeekGemmTileM * (kDeepSeekGemmSuperK / 16u);
    for (uint32_t strip = tid; strip < kAStrips; strip += blockDim.x) {
      const uint32_t m = strip >> 4u;
      const uint32_t kslot = strip & 15u;
      const uint32_t k = k0 + kslot * 16u;
      __nv_bfloat16 *dst =
          a_tile + m * kDeepSeekGemmStageStride + kslot * 16u;
      const uint32_t token = token_base + m;
      if (token >= tokens || k >= cols) {
#pragma unroll
        for (uint32_t i = 0; i < 16u; ++i) {
          dst[i] = deepseek_gemm_zero_bf16();
        }
        continue;
      }
      const uint64_t src_base =
          (static_cast<uint64_t>(token) * groups + group) * cols + k;
      if (k + 16u <= cols) {
        if (input_kind == kDTypeBF16) {
          const uint16_t *src =
              static_cast<const uint16_t *>(input) + src_base;
          if ((reinterpret_cast<uintptr_t>(src) & 0xfu) == 0u) {
            const uint4 lo = reinterpret_cast<const uint4 *>(src)[0];
            const uint4 hi = reinterpret_cast<const uint4 *>(src)[1];
            *reinterpret_cast<uint4 *>(dst) = lo;
            *reinterpret_cast<uint4 *>(dst + 8) = hi;
          } else {
#pragma unroll
            for (uint32_t i = 0; i < 16u; ++i) {
              dst[i] = __ushort_as_bfloat16(src[i]);
            }
          }
        } else {
#pragma unroll
          for (uint32_t i = 0; i < 16u; ++i) {
            dst[i] =
                deepseek_gemm_load_activation(input, src_base + i, input_kind);
          }
        }
      } else {
#pragma unroll
        for (uint32_t i = 0; i < 16u; ++i) {
          dst[i] = k + i < cols ? deepseek_gemm_load_activation(
                                      input, src_base + i, input_kind)
                                : deepseek_gemm_zero_bf16();
        }
      }
    }
    // Commit the prefetched weight strips: tile_n x kDeepSeekGemmSuperK bf16.
#pragma unroll
    for (uint32_t s = 0; s < 2u; ++s) {
      if (s >= w_strips_per_thread || w_strip[s] >= tile_n * 16u) {
        break;
      }
      const uint32_t n = w_strip[s] >> 4u;
      const uint32_t kslot = w_strip[s] & 15u;
      const uint32_t k = k0 + kslot * 16u;
      __nv_bfloat16 *dst =
          w_tile + n * kDeepSeekGemmStageStride + kslot * 16u;
      const uint32_t row = row_base + n;
      if (row >= rows || k >= cols) {
#pragma unroll
        for (uint32_t i = 0; i < 16u; ++i) {
          dst[i] = deepseek_gemm_zero_bf16();
        }
        continue;
      }
      weights.commit(w_raw[s], dst, group_row_base + row, k, cols);
    }
    __syncthreads();
    // Prefetch the next super-chunk before the MMA work so the loads are in
    // flight while tensor cores consume the staged tiles.
    if (k0 + kDeepSeekGemmSuperK < cols) {
      fetch_weights(k0 + kDeepSeekGemmSuperK);
    }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    if (compute_warp) {
#pragma unroll
      for (uint32_t step = 0; step < kDeepSeekGemmSubK / kDeepSeekGemmMmaK;
           ++step) {
        const uint32_t kk =
            warp_k * kDeepSeekGemmSubK + step * kDeepSeekGemmMmaK;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major>
            a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                       wmma::col_major>
            b_frag;
        wmma::load_matrix_sync(a_frag, a_tile + kk, kDeepSeekGemmStageStride);
        wmma::load_matrix_sync(
            b_frag, w_tile + (warp_n * 16u) * kDeepSeekGemmStageStride + kk,
            kDeepSeekGemmStageStride);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
      }
    }
#else
    for (uint32_t part = 0; part < 2u; ++part) {
      const uint32_t idx = tid + part * blockDim.x;
      if (idx >= kDeepSeekGemmTileM * tile_n) continue;
      const uint32_t m = idx / tile_n;
      const uint32_t n = idx - m * tile_n;
      for (uint32_t split = 0; split < kDeepSeekGemmSplitK; ++split) {
        float sum = acc_scalar[part][split];
        for (uint32_t kk = 0; kk < kDeepSeekGemmSubK; ++kk) {
          const uint32_t k = split * kDeepSeekGemmSubK + kk;
          sum = fmaf(
              __bfloat162float(a_tile[m * kDeepSeekGemmStageStride + k]),
              __bfloat162float(w_tile[n * kDeepSeekGemmStageStride + k]), sum);
        }
        acc_scalar[part][split] = sum;
      }
    }
#endif
    __syncthreads();
  }

  // Combine the fixed K-split partials and write C.
  float *c_tile = reinterpret_cast<float *>(deepseek_gemm_smem);
  const uint32_t c_stride = tile_n + 4u;
  const uint32_t c_plane = kDeepSeekGemmTileM * c_stride;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  if (compute_warp) {
    wmma::store_matrix_sync(c_tile + warp_k * c_plane + warp_n * 16u, acc,
                            c_stride, wmma::mem_row_major);
  }
#else
  for (uint32_t part = 0; part < 2u; ++part) {
    const uint32_t idx = tid + part * blockDim.x;
    if (idx >= kDeepSeekGemmTileM * tile_n) continue;
    const uint32_t m = idx / tile_n;
    const uint32_t n = idx - m * tile_n;
    for (uint32_t split = 0; split < kDeepSeekGemmSplitK; ++split) {
      c_tile[split * c_plane + m * c_stride + n] = acc_scalar[part][split];
    }
  }
#endif
  __syncthreads();
  for (uint32_t idx = tid; idx < kDeepSeekGemmTileM * tile_n;
       idx += blockDim.x) {
    const uint32_t m = idx / tile_n;
    const uint32_t n = idx - m * tile_n;
    const uint32_t token = token_base + m;
    const uint32_t row = row_base + n;
    if (token >= tokens || row >= rows) {
      continue;
    }
    const float *c = c_tile + m * c_stride + n;
    // Fixed combine order over the K-split partials (part of the spec).
    const float value =
        ((c[0] + c[c_plane]) + c[2u * c_plane]) + c[3u * c_plane];
    output[(static_cast<uint64_t>(token) * groups + group) * rows + row] =
        value;
  }
}

// Picks the N tile width (does not affect numerics): wide tiles for large
// projections keep staging efficient, narrow tiles keep enough blocks in
// flight to stream weights at M == 1 for small N.
static uint32_t deepseek_gemm_warps_n(uint32_t rows) {
  if (rows >= 16384u) return 4u;
  if (rows >= 8192u) return 2u;
  return 1u;
}

template <typename WeightStager>
static cudaError_t launch_deepseek_session_gemm(
    cudaStream_t stream, const WeightStager &weights, const void *input,
    uint32_t input_kind, uint32_t rows, uint32_t cols, uint32_t tokens,
    uint32_t groups, float *output) {
  const uint32_t warps_n = deepseek_gemm_warps_n(rows);
  const uint32_t tile_n = warps_n * 16u;
  // At least 256 threads per block: narrow-N launches get staging-only warps
  // so enough weight loads stay in flight to stream DRAM at M == 1.
  const uint32_t threads =
      warps_n * 128u < 256u ? 256u : warps_n * 128u;
  const dim3 grid((tokens + kDeepSeekGemmTileM - 1u) / kDeepSeekGemmTileM,
                  (rows + tile_n - 1u) / tile_n, groups);
  const size_t stage_bytes = static_cast<size_t>(kDeepSeekGemmTileM + tile_n) *
                             kDeepSeekGemmStageStride * sizeof(__nv_bfloat16);
  const size_t combine_bytes = static_cast<size_t>(kDeepSeekGemmSplitK) *
                               kDeepSeekGemmTileM * (tile_n + 4u) *
                               sizeof(float);
  const size_t shared_bytes =
      stage_bytes > combine_bytes ? stage_bytes : combine_bytes;
  deepseek_session_gemm_kernel<<<grid, threads, shared_bytes, stream>>>(
      weights, input, input_kind, output, rows, cols, tokens, warps_n);
  return cudaGetLastError();
}

static uint32_t deepseek_gemm_scale_cols(uint32_t cols, uint32_t block_cols) {
  return (cols + block_cols - 1u) / block_cols;
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
  const DeepSeekFp8WeightStager<DeepSeekFp8F32ScaleReader> stager{
      weights, DeepSeekFp8F32ScaleReader{scales}, block_rows, block_cols,
      deepseek_gemm_scale_cols(cols, block_cols)};
  return launch_deepseek_session_gemm(stream, stager, input,
                                      kDeepSeekGemmInputF32, rows, cols, 1u,
                                      1u, output);
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
  const DeepSeekFp8WeightStager<DeepSeekFp8F32ScaleReader> stager{
      weights, DeepSeekFp8F32ScaleReader{scales}, block_rows, block_cols,
      deepseek_gemm_scale_cols(cols, block_cols)};
  return launch_deepseek_session_gemm(stream, stager, input, input_dtype,
                                      rows, cols, tokens, 1u, output);
}

cudaError_t launch_deepseek_fp8_f32_scale_encoded_matvec(
    cudaStream_t stream, const uint8_t *weights, const float *scales,
    const uint16_t *input, uint32_t input_dtype, uint32_t rows, uint32_t cols,
    uint32_t block_rows, uint32_t block_cols, float *output) {
  return launch_deepseek_fp8_f32_scale_encoded_gemm_tokens(
      stream, weights, scales, input, input_dtype, rows, cols, 1u, block_rows,
      block_cols, output);
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
  const DeepSeekFp8WeightStager<DeepSeekFp8SlotScaleReader> stager{
      weights, DeepSeekFp8SlotScaleReader{scale_slots}, block_rows,
      block_cols, deepseek_gemm_scale_cols(cols, block_cols)};
  return launch_deepseek_session_gemm(stream, stager, input, input_dtype,
                                      rows, cols, 1u, 1u, output);
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
  const DeepSeekFp8WeightStager<DeepSeekFp8E8M0ScaleReader> stager{
      weights, DeepSeekFp8E8M0ScaleReader{scales}, block_rows, block_cols,
      deepseek_gemm_scale_cols(cols, block_cols)};
  return launch_deepseek_session_gemm(stream, stager, input, input_dtype,
                                      rows, cols, tokens, 1u, output);
}

cudaError_t launch_deepseek_fp8_e8m0_scale_encoded_matvec(
    cudaStream_t stream, const uint8_t *weights, const uint8_t *scales,
    const uint16_t *input, uint32_t input_dtype, uint32_t rows, uint32_t cols,
    uint32_t block_rows, uint32_t block_cols, float *output) {
  return launch_deepseek_fp8_e8m0_scale_encoded_gemm_tokens(
      stream, weights, scales, input, input_dtype, rows, cols, 1u, block_rows,
      block_cols, output);
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
  const DeepSeekFp8WeightStager<DeepSeekFp8E8M0ScaleReader> stager{
      weights, DeepSeekFp8E8M0ScaleReader{scales}, block_rows, block_cols,
      deepseek_gemm_scale_cols(cols, block_cols)};
  return launch_deepseek_session_gemm(stream, stager, input,
                                      kDeepSeekGemmInputF32, rows, cols, 1u,
                                      1u, output);
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
  const DeepSeekFp8WeightStager<DeepSeekFp8E8M0ScaleReader> stager{
      weights, DeepSeekFp8E8M0ScaleReader{scales}, block_rows, block_cols,
      deepseek_gemm_scale_cols(cols_per_group, block_cols)};
  return launch_deepseek_session_gemm(stream, stager, input,
                                      kDeepSeekGemmInputF32, rows_per_group,
                                      cols_per_group, 1u, groups, output);
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
  const DeepSeekBf16WeightStager stager{weights};
  return launch_deepseek_session_gemm(stream, stager, input,
                                      kDeepSeekGemmInputF32, rows_per_group,
                                      cols_per_group, 1u, groups, output);
}
