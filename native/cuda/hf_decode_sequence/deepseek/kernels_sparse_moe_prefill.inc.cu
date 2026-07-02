__device__ __forceinline__ uint32_t *deepseek_prefill_moe_route_ids(
    uint16_t *route_scratch) {
  return reinterpret_cast<uint32_t *>(route_scratch);
}

__device__ __forceinline__ float *deepseek_prefill_moe_route_weights(
    uint16_t *route_scratch, uint32_t chunk_tokens, uint32_t top_k) {
  return reinterpret_cast<float *>(
      deepseek_prefill_moe_route_ids(route_scratch) +
      static_cast<uint64_t>(chunk_tokens) * top_k);
}

__device__ __forceinline__ float *deepseek_prefill_moe_rank_ff(
    float *gate_up_tmp, uint32_t token, uint32_t rank_ff_stride,
    uint32_t moe_intermediate, uint32_t rank) {
  return gate_up_tmp +
         static_cast<uint64_t>(token) * rank_ff_stride +
         static_cast<uint64_t>(rank) * moe_intermediate;
}

__device__ __forceinline__ float *deepseek_prefill_moe_shared_ff(
    float *gate_up_tmp, uint32_t token, uint32_t rank_ff_stride, uint32_t top_k,
    uint32_t moe_intermediate) {
  return gate_up_tmp + static_cast<uint64_t>(token) * rank_ff_stride +
         static_cast<uint64_t>(top_k) * moe_intermediate;
}

// Router logits with the same per-expert column partition and block
// reduction as the decode-path hf_deepseek_v3_sparse_moe_router_logits_kernel
// so prefill and decode select identical experts on near-tie logits.
__global__ void hf_deepseek_prefill_sparse_moe_router_logits_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t chunk_tokens, const uint16_t *norm_in,
    float *router_logits_tokens) {
  const uint32_t expert = blockIdx.x;
  const uint32_t token = blockIdx.y;
  const uint32_t num_experts = layout.num_experts;
  if (token >= chunk_tokens || arena == nullptr || norm_in == nullptr ||
      router_logits_tokens == nullptr ||
      layout.w_router == kMissingOffset || num_experts == 0 ||
      expert >= num_experts) {
    return;
  }
  const uint16_t *token_norm = norm_in + static_cast<uint64_t>(token) * hidden;
  const uint64_t row =
      layout.w_router + static_cast<uint64_t>(expert) * hidden;
  float sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    sum += encoded_to_f32(arena[row + col], kDTypeBF16) *
           encoded_to_f32(token_norm[col], dtype);
  }
  sum = block_sum(sum);
  if (threadIdx.x == 0) {
    router_logits_tokens[static_cast<uint64_t>(token) * num_experts + expert] =
        sum;
  }
}

__global__ void hf_deepseek_prefill_sparse_moe_route_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t intermediate, uint32_t chunk_tokens,
    const float *router_logits_tokens, uint16_t *route_scratch,
    uint64_t *deepseek_runtime_counters) {
  const uint32_t token = blockIdx.x;
  if (token >= chunk_tokens || arena == nullptr ||
      router_logits_tokens == nullptr || route_scratch == nullptr) {
    return;
  }

  __shared__ float router_logits[kSparseMoeExpertsMax];
  __shared__ float correction_bias[kSparseMoeExpertsMax];
  __shared__ uint32_t selected_experts[kSparseMoeTopKMax];
  __shared__ float selected_weights[kSparseMoeTopKMax];
  __shared__ int route_status;

  const uint32_t num_experts = layout.num_experts;
  const uint32_t top_k = layout.experts_per_token;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  uint32_t *route_ids = deepseek_prefill_moe_route_ids(route_scratch);
  float *route_weights =
      deepseek_prefill_moe_route_weights(route_scratch, chunk_tokens, top_k);
  uint32_t *token_route_ids = route_ids + static_cast<uint64_t>(token) * top_k;
  float *token_route_weights =
      route_weights + static_cast<uint64_t>(token) * top_k;

  if (layout.w_router == kMissingOffset || num_experts == 0 ||
      num_experts > kSparseMoeExpertsMax || top_k == 0 ||
      top_k > kSparseMoeTopKMax || top_k > num_experts ||
      moe_intermediate == 0 || moe_intermediate > intermediate) {
    if (threadIdx.x == 0 && top_k != 0) {
      for (uint32_t rank = 0; rank < top_k; ++rank) {
        token_route_ids[rank] = 0xffffffffu;
        token_route_weights[rank] = 0.0f;
      }
    }
    return;
  }

  for (uint32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    router_logits[expert] =
        router_logits_tokens[static_cast<uint64_t>(token) * num_experts +
                             expert];
  }

  const bool has_router_bias =
      (layout.deepseek_flags & kDeepSeekFlagRouterBias) != 0;
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  const uint64_t router_bias_offset =
      layout.w_router + static_cast<uint64_t>(num_experts) * hidden;
  for (uint32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    correction_bias[expert] =
        !has_router_bias
            ? 0.0f
            : (bf16_storage
                   ? encoded_to_f32(arena[router_bias_offset + expert],
                                    kDTypeBF16)
                   : f32_from_u16_slots(arena + router_bias_offset, expert));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const uint32_t router_groups =
        layout.deepseek_router_num_groups == 0 ? 1u
                                               : layout.deepseek_router_num_groups;
    const uint32_t router_topk_groups =
        layout.deepseek_router_topk_groups == 0
            ? 1u
            : layout.deepseek_router_topk_groups;
    const float routed_scale =
        isfinite(layout.deepseek_routed_scaling_factor) &&
                layout.deepseek_routed_scaling_factor != 0.0f
            ? layout.deepseek_routed_scaling_factor
            : 1.0f;
    route_status = nerva::deepseek::router::route_v3_grouped_sigmoid(
        router_logits, has_router_bias ? correction_bias : nullptr,
        num_experts, router_groups, router_topk_groups, top_k,
        layout.norm_topk_prob, routed_scale, selected_experts,
        selected_weights);
    if (route_status == 0 && deepseek_runtime_counters != nullptr) {
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterV3GroupedRouterSelections),
          1ull);
    }
    for (uint32_t rank = 0; rank < top_k; ++rank) {
      token_route_ids[rank] =
          route_status == 0 ? selected_experts[rank] : 0xffffffffu;
      token_route_weights[rank] =
          route_status == 0 ? selected_weights[rank] : 0.0f;
    }
  }
}

// ---------------------------------------------------------------------------
// Expert-grouped tiled sparse-MoE prefill path.
//
// The route kernel output (route ids/weights at the start of the route
// scratch) is kept unchanged; a pair list grouped by expert plus tile
// metadata is appended after it (see deepseek_prefill_moe_tile_scratch in
// kernels.cuh). Each (token, rank) pair keeps writing the same rank_ff /
// shared_ff slices of gate_up_tmp and the down output keeps its layout; only
// the K-summation order is reassociated (f32 accumulation preserved).
// ---------------------------------------------------------------------------

constexpr uint32_t kDeepSeekPrefillMoeRowTileV2 = 64;
constexpr uint32_t kDeepSeekPrefillMoeKTile = 64;
constexpr uint32_t kDeepSeekPrefillMoeInvalidPair = 0xffffffffu;
constexpr uint32_t kDeepSeekPrefillMoeSharedTile = 0xffffffffu;

// Dequantize 16 consecutive fp8 weights that share one 128-column scale
// block, preserving the exact f8->f32 * scale semantics.
__device__ __forceinline__ void deepseek_prefill_moe_dequant16(
    const uint8_t *src, float scale, float (&out)[16]) {
  if ((reinterpret_cast<uintptr_t>(src) & 0xfu) == 0u) {
    const uint4 raw = *reinterpret_cast<const uint4 *>(src);
    const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (uint32_t word = 0; word < 4u; ++word) {
#pragma unroll
      for (uint32_t byte = 0; byte < 4u; ++byte) {
        out[word * 4u + byte] =
            nerva::deepseek::f8_e4m3fn_bits_to_f32(static_cast<uint8_t>(
                (words[word] >> (byte * 8u)) & 0xffu)) *
            scale;
      }
    }
  } else {
#pragma unroll
    for (uint32_t i = 0; i < 16u; ++i) {
      out[i] = nerva::deepseek::f8_e4m3fn_bits_to_f32(src[i]) * scale;
    }
  }
}

// Load a 16-wide K strip (row, kb..kb+15) of a rows x cols expert matrix in
// either fp8+f32-block-scale or bf16 storage. kb must be 16-aligned so the
// strip never crosses a 128-column scale block. Out-of-range rows/columns
// yield zeros.
__device__ __forceinline__ void deepseek_prefill_moe_load_weight_strip(
    const uint16_t *arena, uint64_t weight_offset, uint64_t scale_offset,
    bool bf16_storage, uint64_t elem_base, uint32_t scale_base,
    uint32_t scale_cols, uint32_t rows, uint32_t cols, uint32_t row,
    uint32_t kb, float (&out)[16]) {
  if (row >= rows || kb >= cols) {
#pragma unroll
    for (uint32_t i = 0; i < 16u; ++i) out[i] = 0.0f;
    return;
  }
  if (bf16_storage) {
    const uint16_t *src = arena + weight_offset + elem_base +
                          static_cast<uint64_t>(row) * cols + kb;
    if (kb + 16u <= cols) {
#pragma unroll
      for (uint32_t i = 0; i < 16u; ++i) {
        out[i] = encoded_to_f32(src[i], kDTypeBF16);
      }
    } else {
#pragma unroll
      for (uint32_t i = 0; i < 16u; ++i) {
        out[i] = kb + i < cols ? encoded_to_f32(src[i], kDTypeBF16) : 0.0f;
      }
    }
    return;
  }
  const uint8_t *base =
      reinterpret_cast<const uint8_t *>(arena + weight_offset);
  const uint8_t *src =
      base + elem_base + static_cast<uint64_t>(row) * cols + kb;
  const uint32_t scale_idx =
      scale_base + (row >> 7u) * scale_cols + (kb >> 7u);
  const float scale = f32_from_u16_slots(arena + scale_offset, scale_idx);
  if (kb + 16u <= cols) {
    deepseek_prefill_moe_dequant16(src, scale, out);
  } else {
#pragma unroll
    for (uint32_t i = 0; i < 16u; ++i) {
      out[i] = kb + i < cols
                   ? nerva::deepseek::f8_e4m3fn_bits_to_f32(src[i]) * scale
                   : 0.0f;
    }
  }
}

// Load a 16-wide strip of encoded (bf16/f16) activations; null src or
// out-of-range columns yield zeros.
__device__ __forceinline__ void deepseek_prefill_moe_load_encoded_strip(
    const uint16_t *src, uint32_t dtype, uint32_t kb, uint32_t cols,
    float (&out)[16]) {
  if (src == nullptr || kb >= cols) {
#pragma unroll
    for (uint32_t i = 0; i < 16u; ++i) out[i] = 0.0f;
    return;
  }
  const uint16_t *strip = src + kb;
  if (kb + 16u <= cols) {
    if ((reinterpret_cast<uintptr_t>(strip) & 0xfu) == 0u) {
      const uint4 *vec = reinterpret_cast<const uint4 *>(strip);
      const uint4 lo = vec[0];
      const uint4 hi = vec[1];
      const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w,
                                 hi.x, hi.y, hi.z, hi.w};
#pragma unroll
      for (uint32_t word = 0; word < 8u; ++word) {
        out[word * 2u] = encoded_to_f32(
            static_cast<uint16_t>(words[word] & 0xffffu), dtype);
        out[word * 2u + 1u] = encoded_to_f32(
            static_cast<uint16_t>(words[word] >> 16u), dtype);
      }
    } else {
#pragma unroll
      for (uint32_t i = 0; i < 16u; ++i) {
        out[i] = encoded_to_f32(strip[i], dtype);
      }
    }
  } else {
#pragma unroll
    for (uint32_t i = 0; i < 16u; ++i) {
      out[i] = kb + i < cols ? encoded_to_f32(strip[i], dtype) : 0.0f;
    }
  }
}

// Load a 16-wide strip of f32 activations; null src or out-of-range columns
// yield zeros.
__device__ __forceinline__ void deepseek_prefill_moe_load_f32_strip(
    const float *src, uint32_t kb, uint32_t cols, float (&out)[16]) {
  if (src == nullptr || kb >= cols) {
#pragma unroll
    for (uint32_t i = 0; i < 16u; ++i) out[i] = 0.0f;
    return;
  }
  const float *strip = src + kb;
  if (kb + 16u <= cols) {
    if ((reinterpret_cast<uintptr_t>(strip) & 0xfu) == 0u) {
      const float4 *vec = reinterpret_cast<const float4 *>(strip);
#pragma unroll
      for (uint32_t part = 0; part < 4u; ++part) {
        const float4 value = vec[part];
        out[part * 4u] = value.x;
        out[part * 4u + 1u] = value.y;
        out[part * 4u + 2u] = value.z;
        out[part * 4u + 3u] = value.w;
      }
    } else {
#pragma unroll
      for (uint32_t i = 0; i < 16u; ++i) out[i] = strip[i];
    }
  } else {
#pragma unroll
    for (uint32_t i = 0; i < 16u; ++i) {
      out[i] = kb + i < cols ? strip[i] : 0.0f;
    }
  }
}

// Single-block kernel that groups the (token, rank) pairs emitted by the
// route kernel by expert, padding each expert segment to whole tiles, and
// emits tile metadata. Pair order within an expert is not deterministic
// (atomic scatter) but every pair owns a distinct output slice, so results
// are unaffected.
__global__ void hf_deepseek_prefill_sparse_moe_build_pairs_kernel(
    SequenceLayerLayout layout, uint32_t chunk_tokens, uint32_t has_shared,
    uint16_t *route_scratch) {
  const uint32_t num_experts = layout.num_experts;
  const uint32_t top_k = layout.experts_per_token;
  if (route_scratch == nullptr || num_experts == 0 ||
      num_experts > kSparseMoeExpertsMax || top_k == 0) {
    return;
  }
  const DeepSeekPrefillMoeTileScratch scratch =
      deepseek_prefill_moe_tile_scratch(chunk_tokens, top_k, num_experts,
                                        has_shared);
  uint32_t *scratch32 = reinterpret_cast<uint32_t *>(route_scratch);
  const uint32_t *route_ids = scratch32;
  uint32_t *tile_count = scratch32 + scratch.counter_offset;
  uint32_t *pair_list = scratch32 + scratch.pair_list_offset;
  uint32_t *tile_meta = scratch32 + scratch.tile_meta_offset;

  __shared__ uint32_t counts[kSparseMoeExpertsMax];
  __shared__ uint32_t offsets[kSparseMoeExpertsMax + 1];
  __shared__ uint32_t cursors[kSparseMoeExpertsMax];
  const uint32_t tid = threadIdx.x;
  for (uint32_t expert = tid; expert < num_experts; expert += blockDim.x) {
    counts[expert] = 0u;
    cursors[expert] = 0u;
  }
  __syncthreads();
  const uint32_t routed_pairs = scratch.routed_pairs;
  for (uint32_t i = tid; i < routed_pairs; i += blockDim.x) {
    const uint32_t expert = route_ids[i];
    if (expert < num_experts) {
      atomicAdd(&counts[expert], 1u);
    }
  }
  __syncthreads();
  if (tid == 0) {
    uint32_t offset = 0u;
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
      offsets[expert] = offset;
      const uint32_t tiles =
          (counts[expert] + kDeepSeekPrefillMoePairTile - 1u) /
          kDeepSeekPrefillMoePairTile;
      offset += tiles * kDeepSeekPrefillMoePairTile;
    }
    offsets[num_experts] = offset;
  }
  __syncthreads();
  const uint32_t routed_padded = offsets[num_experts];
  const uint32_t shared_padded =
      has_shared != 0
          ? ((scratch.shared_pairs + kDeepSeekPrefillMoePairTile - 1u) /
             kDeepSeekPrefillMoePairTile) *
                kDeepSeekPrefillMoePairTile
          : 0u;
  for (uint32_t i = tid; i < routed_padded + shared_padded; i += blockDim.x) {
    pair_list[i] = kDeepSeekPrefillMoeInvalidPair;
  }
  __syncthreads();
  for (uint32_t i = tid; i < routed_pairs; i += blockDim.x) {
    const uint32_t expert = route_ids[i];
    if (expert < num_experts) {
      const uint32_t position =
          offsets[expert] + atomicAdd(&cursors[expert], 1u);
      pair_list[position] = i;
    }
  }
  for (uint32_t token = tid; token < scratch.shared_pairs;
       token += blockDim.x) {
    pair_list[routed_padded + token] = token;
  }
  __syncthreads();
  if (tid == 0) {
    uint32_t tiles = 0u;
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
      const uint32_t expert_tiles =
          (counts[expert] + kDeepSeekPrefillMoePairTile - 1u) /
          kDeepSeekPrefillMoePairTile;
      for (uint32_t t = 0; t < expert_tiles; ++t) {
        tile_meta[2u * tiles] = expert;
        tile_meta[2u * tiles + 1u] =
            offsets[expert] + t * kDeepSeekPrefillMoePairTile;
        ++tiles;
      }
    }
    const uint32_t shared_tiles =
        shared_padded / kDeepSeekPrefillMoePairTile;
    for (uint32_t t = 0; t < shared_tiles; ++t) {
      tile_meta[2u * tiles] = kDeepSeekPrefillMoeSharedTile;
      tile_meta[2u * tiles + 1u] =
          routed_padded + t * kDeepSeekPrefillMoePairTile;
      ++tiles;
    }
    *tile_count = tiles;
  }
}

// Expert-grouped gate/up projection. Each block computes a 64(pair) x
// 64(row) tile of one expert (or of the shared expert for sentinel tiles)
// with 4x4 register blocking, staging activations and dequantized weights in
// shared memory, then applies swiglu and scatters into the same rank_ff /
// shared_ff layout as the legacy kernel.
__global__ void hf_deepseek_prefill_sparse_moe_gate_up_tiles_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t intermediate, uint32_t chunk_tokens,
    uint32_t has_shared, const uint16_t *norm_in, uint16_t *route_scratch,
    float *gate_up_tmp) {
  constexpr uint32_t TM = kDeepSeekPrefillMoePairTile;
  constexpr uint32_t TN = kDeepSeekPrefillMoeRowTileV2;
  constexpr uint32_t TK = kDeepSeekPrefillMoeKTile;
  const uint32_t num_experts = layout.num_experts;
  const uint32_t top_k = layout.experts_per_token;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  const uint32_t shared_intermediate = layout.shared_expert_intermediate;
  const uint32_t rank_ff_stride =
      top_k * moe_intermediate + shared_intermediate;
  if (arena == nullptr || norm_in == nullptr || route_scratch == nullptr ||
      gate_up_tmp == nullptr || num_experts == 0 || top_k == 0 ||
      moe_intermediate == 0 || moe_intermediate > intermediate ||
      layout.w_expert_gate_up == kMissingOffset) {
    return;
  }
  const DeepSeekPrefillMoeTileScratch scratch =
      deepseek_prefill_moe_tile_scratch(chunk_tokens, top_k, num_experts,
                                        has_shared);
  uint32_t *scratch32 = reinterpret_cast<uint32_t *>(route_scratch);
  const uint32_t tile_id = blockIdx.y;
  if (tile_id >= scratch32[scratch.counter_offset]) {
    return;
  }
  const uint32_t *tile_meta = scratch32 + scratch.tile_meta_offset;
  const uint32_t tile_expert = tile_meta[2u * tile_id];
  const uint32_t pair_base = tile_meta[2u * tile_id + 1u];
  const bool is_shared = tile_expert >= num_experts;
  const uint32_t out_rows = is_shared ? shared_intermediate : moe_intermediate;
  const uint32_t row_block = blockIdx.x * TN;
  if (row_block >= out_rows) {
    return;
  }

  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  uint64_t gate_offset;
  uint64_t up_offset;
  uint64_t gate_scale;
  uint64_t up_scale;
  uint64_t elem_base;
  uint32_t scale_base;
  const uint32_t w_scale_cols = deepseek_device_scale_dim(hidden);
  if (is_shared) {
    gate_offset = layout.w_shared_expert_gate;
    up_offset = layout.w_shared_expert_up;
    gate_scale =
        bf16_storage
            ? kMissingOffset
            : gate_offset +
                  deepseek_device_fp8_slots(shared_intermediate, hidden);
    up_scale =
        bf16_storage
            ? kMissingOffset
            : up_offset +
                  deepseek_device_fp8_slots(shared_intermediate, hidden);
    elem_base = 0ull;
    scale_base = 0u;
  } else {
    const uint64_t expert_gate = layout.w_expert_gate_up;
    const uint64_t expert_gate_data_slots =
        bf16_storage
            ? static_cast<uint64_t>(num_experts) * moe_intermediate * hidden
            : deepseek_device_fp8_slots(
                  static_cast<uint64_t>(num_experts) * moe_intermediate,
                  hidden);
    gate_offset = expert_gate;
    gate_scale =
        bf16_storage ? kMissingOffset : expert_gate + expert_gate_data_slots;
    up_offset =
        bf16_storage
            ? expert_gate + expert_gate_data_slots
            : gate_scale +
                  static_cast<uint64_t>(num_experts) *
                      deepseek_device_scale_f32_slots(moe_intermediate,
                                                      hidden);
    up_scale =
        bf16_storage
            ? kMissingOffset
            : up_offset +
                  deepseek_device_fp8_slots(
                      static_cast<uint64_t>(num_experts) * moe_intermediate,
                      hidden);
    elem_base =
        static_cast<uint64_t>(tile_expert) * moe_intermediate * hidden;
    scale_base = tile_expert * deepseek_device_scale_dim(moe_intermediate) *
                 w_scale_cols;
  }

  __shared__ float x_tile[TK][TM + 1];
  __shared__ float w_tile[TK][TN + 1];
  __shared__ uint32_t pair_vals[TM];
  __shared__ uint32_t pair_tokens[TM];
  const uint32_t tid = threadIdx.x;
  if (tid < TM) {
    const uint32_t value =
        scratch32[scratch.pair_list_offset + pair_base + tid];
    pair_vals[tid] = value;
    pair_tokens[tid] = value == kDeepSeekPrefillMoeInvalidPair
                           ? kDeepSeekPrefillMoeInvalidPair
                           : (is_shared ? value : value / top_k);
  }
  __syncthreads();

  const uint32_t load_m = tid >> 2u;
  const uint32_t load_k = (tid & 3u) * 16u;
  const uint32_t thread_pair = tid >> 4u;
  const uint32_t thread_row = tid & 15u;
  float acc_gate[4][4] = {};
  float acc_up[4][4] = {};
  for (uint32_t k0 = 0; k0 < hidden; k0 += TK) {
    {
      const uint32_t token = pair_tokens[load_m];
      const uint16_t *src =
          token == kDeepSeekPrefillMoeInvalidPair
              ? nullptr
              : norm_in + static_cast<uint64_t>(token) * hidden;
      float staged[16];
      deepseek_prefill_moe_load_encoded_strip(src, dtype, k0 + load_k, hidden,
                                              staged);
#pragma unroll
      for (uint32_t i = 0; i < 16u; ++i) {
        x_tile[load_k + i][load_m] = staged[i];
      }
    }
    {
      float staged[16];
      deepseek_prefill_moe_load_weight_strip(
          arena, gate_offset, gate_scale, bf16_storage, elem_base, scale_base,
          w_scale_cols, out_rows, hidden, row_block + load_m, k0 + load_k,
          staged);
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
        a[i] = x_tile[k][thread_pair * 4u + i];
        b[i] = w_tile[k][thread_row * 4u + i];
      }
#pragma unroll
      for (uint32_t i = 0; i < 4u; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
          acc_gate[i][j] += a[i] * b[j];
        }
      }
    }
    __syncthreads();
    {
      float staged[16];
      deepseek_prefill_moe_load_weight_strip(
          arena, up_offset, up_scale, bf16_storage, elem_base, scale_base,
          w_scale_cols, out_rows, hidden, row_block + load_m, k0 + load_k,
          staged);
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
        a[i] = x_tile[k][thread_pair * 4u + i];
        b[i] = w_tile[k][thread_row * 4u + i];
      }
#pragma unroll
      for (uint32_t i = 0; i < 4u; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
          acc_up[i][j] += a[i] * b[j];
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t i = 0; i < 4u; ++i) {
    const uint32_t pair_slot = thread_pair * 4u + i;
    const uint32_t value = pair_vals[pair_slot];
    if (value == kDeepSeekPrefillMoeInvalidPair) continue;
    const uint32_t token = pair_tokens[pair_slot];
    float *dst;
    if (is_shared) {
      dst = deepseek_prefill_moe_shared_ff(gate_up_tmp, token, rank_ff_stride,
                                           top_k, moe_intermediate);
    } else {
      const uint32_t rank = value - token * top_k;
      dst = deepseek_prefill_moe_rank_ff(gate_up_tmp, token, rank_ff_stride,
                                         moe_intermediate, rank);
    }
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
      const uint32_t row = row_block + thread_row * 4u + j;
      if (row < out_rows) {
        dst[row] = deepseek_swiglu(acc_gate[i][j], acc_up[i][j],
                                   layout.deepseek_swiglu_limit);
      }
    }
  }
}

// Expert-grouped down projection, phase A: for a slab of hidden rows
// [slab_base, slab_base + slab_rows), compute the per-(token, rank) down
// partials (and the shared-expert partial) into the staging area. Route
// weights are applied in the combine phase.
__global__ void hf_deepseek_prefill_sparse_moe_down_tiles_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t hidden,
    uint32_t intermediate, uint32_t chunk_tokens, uint32_t has_shared,
    uint32_t slab_base, uint32_t slab_rows, uint32_t slab_stride,
    uint16_t *route_scratch, const float *gate_up_tmp) {
  constexpr uint32_t TM = kDeepSeekPrefillMoePairTile;
  constexpr uint32_t TN = kDeepSeekPrefillMoeRowTileV2;
  constexpr uint32_t TK = kDeepSeekPrefillMoeKTile;
  const uint32_t num_experts = layout.num_experts;
  const uint32_t top_k = layout.experts_per_token;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  const uint32_t shared_intermediate = layout.shared_expert_intermediate;
  const uint32_t rank_ff_stride =
      top_k * moe_intermediate + shared_intermediate;
  if (arena == nullptr || route_scratch == nullptr || gate_up_tmp == nullptr ||
      num_experts == 0 || top_k == 0 || moe_intermediate == 0 ||
      moe_intermediate > intermediate ||
      layout.w_expert_down == kMissingOffset) {
    return;
  }
  const DeepSeekPrefillMoeTileScratch scratch =
      deepseek_prefill_moe_tile_scratch(chunk_tokens, top_k, num_experts,
                                        has_shared);
  uint32_t *scratch32 = reinterpret_cast<uint32_t *>(route_scratch);
  const uint32_t tile_id = blockIdx.y;
  if (tile_id >= scratch32[scratch.counter_offset]) {
    return;
  }
  const uint32_t *tile_meta = scratch32 + scratch.tile_meta_offset;
  const uint32_t tile_expert = tile_meta[2u * tile_id];
  const uint32_t pair_base = tile_meta[2u * tile_id + 1u];
  const bool is_shared = tile_expert >= num_experts;
  const uint32_t k_len = is_shared ? shared_intermediate : moe_intermediate;
  const uint32_t row_block = blockIdx.x * TN;
  if (row_block >= slab_rows || k_len == 0) {
    return;
  }
  const uint32_t row_limit =
      slab_base + slab_rows < hidden ? slab_base + slab_rows : hidden;
  float *staging =
      reinterpret_cast<float *>(scratch32 + scratch.staging_offset);

  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  uint64_t down_offset;
  uint64_t down_scale;
  uint64_t elem_base;
  uint32_t scale_base;
  uint32_t w_scale_cols;
  if (is_shared) {
    down_offset = layout.w_shared_expert_down;
    down_scale =
        bf16_storage
            ? kMissingOffset
            : down_offset +
                  deepseek_device_fp8_slots(hidden, shared_intermediate);
    elem_base = 0ull;
    scale_base = 0u;
    w_scale_cols = deepseek_device_scale_dim(shared_intermediate);
  } else {
    down_offset = layout.w_expert_down;
    down_scale =
        bf16_storage
            ? kMissingOffset
            : down_offset +
                  deepseek_device_fp8_slots(
                      static_cast<uint64_t>(num_experts) * hidden,
                      moe_intermediate);
    elem_base =
        static_cast<uint64_t>(tile_expert) * hidden * moe_intermediate;
    w_scale_cols = deepseek_device_scale_dim(moe_intermediate);
    scale_base =
        tile_expert * deepseek_device_scale_dim(hidden) * w_scale_cols;
  }

  __shared__ float x_tile[TK][TM + 1];
  __shared__ float w_tile[TK][TN + 1];
  __shared__ uint32_t pair_vals[TM];
  __shared__ uint32_t pair_tokens[TM];
  const uint32_t tid = threadIdx.x;
  if (tid < TM) {
    const uint32_t value =
        scratch32[scratch.pair_list_offset + pair_base + tid];
    pair_vals[tid] = value;
    pair_tokens[tid] = value == kDeepSeekPrefillMoeInvalidPair
                           ? kDeepSeekPrefillMoeInvalidPair
                           : (is_shared ? value : value / top_k);
  }
  __syncthreads();

  const uint32_t load_m = tid >> 2u;
  const uint32_t load_k = (tid & 3u) * 16u;
  const uint32_t thread_pair = tid >> 4u;
  const uint32_t thread_row = tid & 15u;
  float acc[4][4] = {};
  for (uint32_t k0 = 0; k0 < k_len; k0 += TK) {
    {
      const uint32_t token = pair_tokens[load_m];
      const float *src = nullptr;
      if (token != kDeepSeekPrefillMoeInvalidPair) {
        const uint32_t col_base =
            is_shared
                ? top_k * moe_intermediate
                : (pair_vals[load_m] - token * top_k) * moe_intermediate;
        src = gate_up_tmp + static_cast<uint64_t>(token) * rank_ff_stride +
              col_base;
      }
      float staged[16];
      deepseek_prefill_moe_load_f32_strip(src, k0 + load_k, k_len, staged);
#pragma unroll
      for (uint32_t i = 0; i < 16u; ++i) {
        x_tile[load_k + i][load_m] = staged[i];
      }
    }
    {
      float staged[16];
      deepseek_prefill_moe_load_weight_strip(
          arena, down_offset, down_scale, bf16_storage, elem_base, scale_base,
          w_scale_cols, row_limit, k_len, slab_base + row_block + load_m,
          k0 + load_k, staged);
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
        a[i] = x_tile[k][thread_pair * 4u + i];
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
    const uint32_t pair_slot = thread_pair * 4u + i;
    const uint32_t value = pair_vals[pair_slot];
    if (value == kDeepSeekPrefillMoeInvalidPair) continue;
    const uint32_t token = pair_tokens[pair_slot];
    const uint32_t staging_slot =
        is_shared ? scratch.routed_pairs + token : value;
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
      const uint32_t row_local = row_block + thread_row * 4u + j;
      if (row_local < slab_rows && slab_base + row_local < hidden) {
        staging[static_cast<uint64_t>(staging_slot) * slab_stride +
                row_local] = acc[i][j];
      }
    }
  }
}

// Expert-grouped down projection, phase B: deterministically combine the
// staged per-rank partials (route weights applied here, in ascending rank
// order) plus the shared-expert partial into the down output slab.
__global__ void hf_deepseek_prefill_sparse_moe_down_combine_kernel(
    SequenceLayerLayout layout, uint32_t hidden, uint32_t chunk_tokens,
    uint32_t has_shared, uint32_t slab_base, uint32_t slab_rows,
    uint32_t slab_stride, uint16_t *route_scratch, float *down_out) {
  const uint32_t token = blockIdx.y;
  const uint32_t row_local = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t num_experts = layout.num_experts;
  const uint32_t top_k = layout.experts_per_token;
  if (route_scratch == nullptr || down_out == nullptr ||
      token >= chunk_tokens || row_local >= slab_rows ||
      slab_base + row_local >= hidden || top_k == 0) {
    return;
  }
  const DeepSeekPrefillMoeTileScratch scratch =
      deepseek_prefill_moe_tile_scratch(chunk_tokens, top_k, num_experts,
                                        has_shared);
  uint32_t *scratch32 = reinterpret_cast<uint32_t *>(route_scratch);
  const uint32_t *route_ids = scratch32;
  const float *route_weights =
      reinterpret_cast<const float *>(scratch32 + scratch.routed_pairs);
  const float *staging =
      reinterpret_cast<const float *>(scratch32 + scratch.staging_offset);
  float sum = 0.0f;
  for (uint32_t rank = 0; rank < top_k; ++rank) {
    const uint32_t route_offset = token * top_k + rank;
    if (route_ids[route_offset] < num_experts) {
      sum += route_weights[route_offset] *
             staging[static_cast<uint64_t>(route_offset) * slab_stride +
                     row_local];
    }
  }
  if (has_shared != 0) {
    sum += staging[static_cast<uint64_t>(scratch.routed_pairs + token) *
                       slab_stride +
                   row_local];
  }
  down_out[static_cast<uint64_t>(token) * hidden + slab_base + row_local] =
      sum;
}

