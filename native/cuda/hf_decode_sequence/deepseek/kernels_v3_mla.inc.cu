__global__ void hf_deepseek_v3_mla_cache_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t heads, uint32_t *step_cursor,
    uint32_t max_steps, float rope_theta, const float *q,
    const float *kv_a, float *latent_output,
    const uint16_t *kv_latent_norm, uint16_t *kv_keys,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint16_t *projection_input, uint8_t *deepseek_v32_mla_kv,
    uint64_t deepseek_v32_mla_kv_offset_bytes,
    uint32_t deepseek_v32_mla_kv_block_count,
    const uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    const uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint64_t *deepseek_runtime_counters) {
  (void)heads;
  (void)q;
  (void)latent_output;
  (void)projection_input;
  (void)deepseek_indexer_state;
  (void)deepseek_indexer_state_offset_bytes;
  (void)deepseek_indexer_kv;
  (void)deepseek_indexer_kv_offset_bytes;
  (void)deepseek_indexer_kv_block_count;
  (void)deepseek_runtime_counters;
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t kv_cache_width = kv_lora_rank + qk_rope;
  if (kv_lora_rank == 0 || qk_rope == 0 || kv_cache_width == 0) {
    return;
  }

  const uint64_t write_base =
      kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                          position, kv_cache_width, 0);
  for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
       latent += blockDim.x) {
    kv_keys[write_base + latent] = kv_latent_norm[latent];
  }
  for (uint32_t dim = threadIdx.x; dim < qk_rope; dim += blockDim.x) {
    float value = f32_to_model_dtype(kv_a[kv_lora_rank + dim], dtype);
    if ((qk_rope & 1u) == 0u) {
      const uint32_t even = dim & ~1u;
      const uint32_t odd = even + 1u;
      value = deepseek_rope_value_gptj(
          f32_to_model_dtype(kv_a[kv_lora_rank + even], dtype),
          f32_to_model_dtype(kv_a[kv_lora_rank + odd], dtype), dim, qk_rope,
          position, rope_theta, layout);
    }
    kv_keys[write_base + kv_lora_rank + dim] = f32_to_encoded(value, dtype);
  }
  __syncthreads();
  deepseek_session_write_v32_fp8_ds_mla_kv(
      deepseek_v32_mla_kv, deepseek_v32_mla_kv_offset_bytes,
      deepseek_v32_mla_kv_block_count, kv_block_table, kv_block_count,
      layout, position, dtype, kv_latent_norm, kv_a, rope_theta);
}

__global__ void hf_deepseek_rms_norm_encoded_tokens_kernel(
    uint16_t *arena, uint64_t weight_offset, const uint16_t *input,
    uint32_t weight_dtype, uint32_t input_dtype, uint32_t output_dtype,
    uint32_t rows, uint32_t input_stride, uint32_t output_stride,
    uint32_t tokens, float rms_eps, uint16_t *output) {
  const uint32_t token = blockIdx.x;
  if (arena == nullptr || input == nullptr || output == nullptr ||
      weight_offset == kMissingOffset || rows == 0 || input_stride < rows ||
      output_stride < rows || token >= tokens || weight_dtype > kDTypeF32 ||
      input_dtype > kDTypeBF16 || output_dtype > kDTypeBF16) {
    return;
  }
  const uint16_t *token_input =
      input + static_cast<uint64_t>(token) * input_stride;
  uint16_t *token_output =
      output + static_cast<uint64_t>(token) * output_stride;
  const uint16_t *weight = arena + weight_offset;
  float mean_square = 0.0f;
  for (uint32_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const float value = encoded_to_f32(token_input[row], input_dtype);
    mean_square += value * value;
  }
  mean_square = block_sum(mean_square);
  const float scale =
      rsqrtf(mean_square / static_cast<float>(rows) + rms_eps);
  for (uint32_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const float norm_weight =
        weight_dtype == kDTypeF32
            ? f32_weight_to_f32_unaligned(weight, row)
            : encoded_to_f32(weight[row], weight_dtype);
    token_output[row] = f32_to_encoded(
        encoded_to_f32(token_input[row], input_dtype) * scale * norm_weight,
        output_dtype);
  }
}

__global__ void hf_deepseek_rms_norm_f32_tokens_kernel(
    uint16_t *arena, uint64_t weight_offset, const float *input,
    uint32_t weight_dtype, uint32_t output_dtype, uint32_t rows,
    uint32_t input_stride, uint32_t output_stride, uint32_t tokens,
    float rms_eps, uint16_t *output) {
  const uint32_t token = blockIdx.x;
  if (arena == nullptr || input == nullptr || output == nullptr ||
      weight_offset == kMissingOffset || rows == 0 || input_stride < rows ||
      output_stride < rows || token >= tokens || weight_dtype > kDTypeF32 ||
      output_dtype > kDTypeBF16) {
    return;
  }
  const float *token_input =
      input + static_cast<uint64_t>(token) * input_stride;
  uint16_t *token_output =
      output + static_cast<uint64_t>(token) * output_stride;
  const uint16_t *weight = arena + weight_offset;
  float mean_square = 0.0f;
  for (uint32_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const float value = f32_to_model_dtype(token_input[row], output_dtype);
    mean_square += value * value;
  }
  mean_square = block_sum(mean_square);
  const float scale =
      rsqrtf(mean_square / static_cast<float>(rows) + rms_eps);
  for (uint32_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const float norm_weight =
        weight_dtype == kDTypeF32
            ? f32_weight_to_f32_unaligned(weight, row)
            : encoded_to_f32(weight[row], weight_dtype);
    token_output[row] = f32_to_encoded(
        f32_to_model_dtype(token_input[row], output_dtype) * scale * norm_weight,
        output_dtype);
  }
}

__global__ void hf_deepseek_v3_mla_cache_encode_tokens_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t chunk_start, uint32_t chunk_tokens,
    uint32_t max_steps, float rope_theta, const float *kv_a_tokens,
    uint32_t kv_a_stride, const uint16_t *kv_latent_norm_tokens,
    uint32_t kv_latent_norm_stride, uint16_t *kv_keys,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint8_t *deepseek_v32_mla_kv,
    uint64_t deepseek_v32_mla_kv_offset_bytes,
    uint32_t deepseek_v32_mla_kv_block_count) {
  (void)arena;
  const uint32_t local_token = blockIdx.x;
  if (local_token >= chunk_tokens || kv_a_tokens == nullptr ||
      kv_latent_norm_tokens == nullptr || kv_keys == nullptr) {
    return;
  }
  const uint32_t position = chunk_start + local_token;
  if (position >= max_steps) {
    return;
  }
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t kv_cache_width = kv_lora_rank + qk_rope;
  if (kv_lora_rank == 0 || qk_rope == 0 || kv_cache_width == 0 ||
      kv_a_stride < kv_cache_width ||
      kv_latent_norm_stride < kv_lora_rank) {
    return;
  }

  const float *kv_a =
      kv_a_tokens + static_cast<uint64_t>(local_token) * kv_a_stride;
  const uint16_t *kv_latent_norm =
      kv_latent_norm_tokens +
      static_cast<uint64_t>(local_token) * kv_latent_norm_stride;
  const uint64_t write_base =
      kv_cache_token_base(layer_index, kv_block_count, kv_block_table, position,
                          kv_cache_width, 0);
  for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
       latent += blockDim.x) {
    kv_keys[write_base + latent] = kv_latent_norm[latent];
  }
  for (uint32_t dim = threadIdx.x; dim < qk_rope; dim += blockDim.x) {
    float value = f32_to_model_dtype(kv_a[kv_lora_rank + dim], dtype);
    if ((qk_rope & 1u) == 0u) {
      const uint32_t even = dim & ~1u;
      const uint32_t odd = even + 1u;
      value = deepseek_rope_value_gptj(
          f32_to_model_dtype(kv_a[kv_lora_rank + even], dtype),
          f32_to_model_dtype(kv_a[kv_lora_rank + odd], dtype), dim, qk_rope,
          position, rope_theta, layout);
    }
    kv_keys[write_base + kv_lora_rank + dim] = f32_to_encoded(value, dtype);
  }
  __syncthreads();
  deepseek_session_write_v32_fp8_ds_mla_kv(
      deepseek_v32_mla_kv, deepseek_v32_mla_kv_offset_bytes,
      deepseek_v32_mla_kv_block_count, kv_block_table, kv_block_count,
      layout, position, dtype, kv_latent_norm, kv_a, rope_theta);
}

// ===========================================================================
// Unified DeepSeek V3/V3.2 MLA attention family.
//
// One kernel family serves BOTH the single-token decode step and the batched
// prefill path so their numerics are identical by construction:
//
//   1. hf_deepseek_mla_q_latent_tokens_kernel
//        Absorbs the per-head no-position query through w_kc (the first
//        qk_nope rows of layout.w_v) and applies rope to the trailing
//        qk_rope query dims. Emits one encoded (bf16/f16) row of
//        kv_lora_rank + qk_rope values per (token, head). Every output
//        element is a private serial dot product in ascending index order,
//        so batching tokens per block cannot change any element's value.
//
//   2. hf_deepseek_mla_fa_tile_kernel
//        Flash-attention over the shared latent KV cache with bf16
//        mma.sync (m16n8k16, f32 accumulators) for both Q.K^T and P.V.
//        A block owns ONE query position and a fixed tile of
//        kDeepSeekMlaFaHeadTile heads; it walks the position's slot list
//        (or the causal prefix) in fixed ascending tiles of
//        kDeepSeekMlaFaTokenTile tokens, updating a per-row running
//        max/sum once per tile. There is no split-K across blocks and no
//        atomics, so a (position, head) row sees exactly the same
//        instruction sequence whether the launch covers 1 position
//        (decode) or many (prefill).
//
//   3. hf_deepseek_mla_v_proj_tokens_kernel
//        Projects the attended latent through the per-head V rows of
//        layout.w_v. Again one private serial dot per output element.
//
// Padding safety: when heads is not a multiple of the head tile the extra
// rows are staged as zero queries. Each mma.sync output element C[m][n] is
// the dot product of A row m with B column n plus C[m][n]; rows never mix,
// and every softmax reduction here is per row, so padding rows cannot
// perturb real rows. Padding rows are simply not written back.
//
// Dims served by the MMA kernel: kv_lora_rank == kDeepSeekMlaFaLora and
// qk_rope_head_dim == kDeepSeekMlaFaRope with bf16 model dtype (all
// DeepSeek V3/V3.1/V3.2 checkpoints). Anything else takes the single
// generic fallback hf_deepseek_mla_fa_generic_kernel, which is also shared
// verbatim by both paths.
// ===========================================================================

__device__ __forceinline__ uint32_t deepseek_mla_smem_u32addr(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void deepseek_mla_cp_async16(void *dst,
                                                        const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                   deepseek_mla_smem_u32addr(dst)),
               "l"(src));
}

__device__ __forceinline__ void deepseek_mla_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void deepseek_mla_cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n");
}

__device__ __forceinline__ void deepseek_mla_ldmatrix_x4(uint32_t addr,
                                                         uint32_t &r0,
                                                         uint32_t &r1,
                                                         uint32_t &r2,
                                                         uint32_t &r3) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void deepseek_mla_ldmatrix_x2(uint32_t addr,
                                                         uint32_t &r0,
                                                         uint32_t &r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

__device__ __forceinline__ void deepseek_mla_ldmatrix_x2_trans(uint32_t addr,
                                                               uint32_t &r0,
                                                               uint32_t &r1) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(r0), "=r"(r1)
      : "r"(addr));
}

__device__ __forceinline__ void deepseek_mla_mma_bf16(
    float &c0, float &c1, float &c2, float &c3, uint32_t a0, uint32_t a1,
    uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Shared helper: resolve the query position and this position's attention
// slot list. Returns false when the block has nothing to do.
__device__ __forceinline__ bool deepseek_mla_resolve_row(
    uint32_t *step_cursor, uint32_t max_steps, uint32_t chunk_start,
    uint32_t token_count, uint32_t local_token,
    const int32_t *sparse_topk_slots, uint32_t sparse_topk_stride,
    const uint32_t *sparse_topk_count, uint32_t *position_out,
    const int32_t **list_out, uint32_t *list_len_out) {
  if (local_token >= token_count) {
    return false;
  }
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return false;
  }
  const uint32_t position =
      step_cursor != nullptr ? *step_cursor : chunk_start + local_token;
  if (position >= max_steps) {
    return false;
  }
  const int32_t *list = nullptr;
  uint32_t list_len = position + 1u;
  if (sparse_topk_slots != nullptr && sparse_topk_count != nullptr) {
    list_len = sparse_topk_count[local_token];
    if (sparse_topk_stride != 0) {
      list_len = min(list_len, sparse_topk_stride);
      list = sparse_topk_slots +
             static_cast<uint64_t>(local_token) * sparse_topk_stride;
    } else {
      list = sparse_topk_slots;
    }
    list_len = min(list_len, kDeepSeekSessionMaxSparseTopK);
  }
  *position_out = position;
  *list_out = list;
  *list_len_out = list_len;
  return true;
}

__global__ void hf_deepseek_mla_q_latent_tokens_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t heads, uint32_t *step_cursor, uint32_t max_steps,
    uint32_t chunk_start, uint32_t token_count, float rope_theta,
    const float *q_tokens, uint32_t q_stride, uint16_t *q_latent) {
  const uint32_t head = blockIdx.x;
  const uint32_t token_base = blockIdx.y * kDeepSeekMlaQLatentTokensPerBlock;
  if (head >= heads || token_base >= token_count || q_tokens == nullptr ||
      q_latent == nullptr || layout.w_v == kMissingOffset) {
    return;
  }
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t v_head = layout.deepseek_v_head_dim;
  const uint32_t qk_head_dim = qk_nope + qk_rope;
  const uint32_t width = kv_lora_rank + qk_rope;
  if (kv_lora_rank == 0 || qk_nope == 0 || qk_rope == 0 || v_head == 0 ||
      q_stride < heads * qk_head_dim ||
      qk_head_dim > kDeepSeekMlaQLatentMaxHeadDim) {
    return;
  }
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  if (!bf16_storage && layout.deepseek_kv_b_scale == kMissingOffset) {
    return;
  }
  const uint32_t token_group =
      min(kDeepSeekMlaQLatentTokensPerBlock, token_count - token_base);

  // Stage the model-dtype-rounded per-head query for each token in the
  // group. The rounding here matches the f32_to_model_dtype() the previous
  // kernels applied at every use site.
  __shared__ float q_shared[kDeepSeekMlaQLatentTokensPerBlock]
                           [kDeepSeekMlaQLatentMaxHeadDim];
  for (uint32_t idx = threadIdx.x; idx < token_group * qk_head_dim;
       idx += blockDim.x) {
    const uint32_t t = idx / qk_head_dim;
    const uint32_t dim = idx - t * qk_head_dim;
    const float *q_row =
        q_tokens + static_cast<uint64_t>(token_base + t) * q_stride;
    q_shared[t][dim] =
        f32_to_model_dtype(q_row[head * qk_head_dim + dim], dtype);
  }
  __syncthreads();

  const uint32_t kv_b_cols = kv_lora_rank;
  const uint8_t *kv_b_weight =
      reinterpret_cast<const uint8_t *>(arena + layout.w_v);
  const uint16_t *kv_b_scale =
      bf16_storage ? nullptr : arena + layout.deepseek_kv_b_scale;
  const uint32_t kv_b_scale_cols = (kv_b_cols + 127u) / 128u;

  // Each thread owns latent columns; the weight column is loaded once and
  // reused for every token in the group. Per (token, head, latent) the
  // accumulation order over nope is identical for any group size.
  for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
       latent += blockDim.x) {
    float sum[kDeepSeekMlaQLatentTokensPerBlock];
#pragma unroll
    for (uint32_t t = 0; t < kDeepSeekMlaQLatentTokensPerBlock; ++t) {
      sum[t] = 0.0f;
    }
    uint32_t active_scale_row = UINT32_MAX;
    float active_scale = 0.0f;
    for (uint32_t nope = 0; nope < qk_nope; ++nope) {
      const uint32_t row = head * (qk_nope + v_head) + nope;
      const uint32_t scale_row = row / 128u;
      if (!bf16_storage && scale_row != active_scale_row) {
        active_scale_row = scale_row;
        active_scale = f32_from_u16_slots(
            kv_b_scale, scale_row * kv_b_scale_cols + latent / 128u);
      }
      const float weight =
          bf16_storage
              ? deepseek_bf16_weight(arena, layout.w_v,
                                     heads * (qk_nope + v_head), kv_b_cols,
                                     row, latent)
              : nerva::deepseek::f8_e4m3fn_bits_to_f32(
                    kv_b_weight[static_cast<uint64_t>(row) * kv_b_cols +
                                latent]) *
                    active_scale;
#pragma unroll
      for (uint32_t t = 0; t < kDeepSeekMlaQLatentTokensPerBlock; ++t) {
        if (t < token_group) {
          sum[t] += q_shared[t][nope] * weight;
        }
      }
    }
#pragma unroll
    for (uint32_t t = 0; t < kDeepSeekMlaQLatentTokensPerBlock; ++t) {
      if (t < token_group) {
        const uint32_t position = step_cursor != nullptr
                                      ? *step_cursor
                                      : chunk_start + token_base + t;
        if (position >= max_steps) {
          continue;
        }
        uint16_t *out_row =
            q_latent + (static_cast<uint64_t>(token_base + t) * heads + head) *
                           width;
        out_row[latent] = f32_to_encoded(sum[t], dtype);
      }
    }
  }
  for (uint32_t idx = threadIdx.x; idx < token_group * qk_rope;
       idx += blockDim.x) {
    const uint32_t t = idx / qk_rope;
    const uint32_t dim = idx - t * qk_rope;
    const uint32_t position =
        step_cursor != nullptr ? *step_cursor : chunk_start + token_base + t;
    if (position >= max_steps) {
      continue;
    }
    float q_pe = q_shared[t][qk_nope + dim];
    if ((qk_rope & 1u) == 0u) {
      const uint32_t even = dim & ~1u;
      const uint32_t odd = even + 1u;
      q_pe = deepseek_rope_value_gptj(q_shared[t][qk_nope + even],
                                      q_shared[t][qk_nope + odd], dim,
                                      qk_rope, position, rope_theta, layout);
    }
    uint16_t *out_row =
        q_latent +
        (static_cast<uint64_t>(token_base + t) * heads + head) * width;
    out_row[kv_lora_rank + dim] = f32_to_encoded(q_pe, dtype);
  }
}

__global__ void __launch_bounds__(kDecodeThreads, 1)
    hf_deepseek_mla_fa_tile_kernel(
        SequenceLayerLayout layout, uint32_t layer_index, uint32_t heads,
        uint32_t *step_cursor, uint32_t max_steps, uint32_t chunk_start,
        uint32_t token_count, const uint16_t *q_latent,
        const uint16_t *kv_keys, uint32_t kv_block_count,
        const uint32_t *kv_block_table, uint16_t *attn_latent,
        const int32_t *sparse_topk_slots, uint32_t sparse_topk_stride,
        const uint32_t *sparse_topk_count,
        uint64_t *deepseek_runtime_counters) {
  constexpr uint32_t kLora = kDeepSeekMlaFaLora;
  constexpr uint32_t kWidth = kDeepSeekMlaFaWidth;
  constexpr uint32_t kHT = kDeepSeekMlaFaHeadTile;
  constexpr uint32_t kKT = kDeepSeekMlaFaTokenTile;
  constexpr uint32_t kStride = kDeepSeekMlaFaSmemStride;
  constexpr uint32_t kPStride = kDeepSeekMlaFaPStride;
  constexpr uint32_t kWarps = kDeepSeekMlaFaWarps;
  static_assert(kDecodeThreads == kWarps * 32u, "FA kernel warp layout");

  const uint32_t local_token = blockIdx.x;
  const uint32_t head_base = blockIdx.y * kHT;
  if (head_base >= heads || q_latent == nullptr || kv_keys == nullptr ||
      attn_latent == nullptr) {
    return;
  }
  uint32_t position = 0;
  const int32_t *list = nullptr;
  uint32_t list_len = 0;
  if (!deepseek_mla_resolve_row(step_cursor, max_steps, chunk_start,
                                token_count, local_token, sparse_topk_slots,
                                sparse_topk_stride, sparse_topk_count,
                                &position, &list, &list_len)) {
    return;
  }
  if (layout.deepseek_kv_lora_rank != kLora ||
      layout.deepseek_qk_rope_head_dim != kDeepSeekMlaFaRope) {
    return;
  }
  const float softmax_scale = deepseek_mla_attention_scale(
      layout,
      layout.deepseek_qk_nope_head_dim + layout.deepseek_qk_rope_head_dim);

  extern __shared__ uint16_t mla_fa_smem[];
  uint16_t *q_sm = mla_fa_smem;                       // kHT x kStride
  uint16_t *kv_sm = q_sm + kHT * kStride;             // kKT x kStride
  uint16_t *p_sm = kv_sm + kKT * kStride;             // kHT x kPStride
  float *red_sm = reinterpret_cast<float *>(p_sm + kHT * kPStride);
  int32_t *slot_sm = reinterpret_cast<int32_t *>(red_sm + kHT * kWarps);

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t warp = tid >> 5u;

  // Stage the head tile's query rows; rows past `heads` are zero padding.
  for (uint32_t c = tid; c < kHT * (kWidth / 8u); c += blockDim.x) {
    const uint32_t row = c / (kWidth / 8u);
    const uint32_t col8 = c - row * (kWidth / 8u);
    uint16_t *dst = &q_sm[row * kStride + col8 * 8u];
    const uint32_t head = head_base + row;
    if (head < heads) {
      const uint16_t *src =
          q_latent +
          (static_cast<uint64_t>(local_token) * heads + head) * kWidth +
          col8 * 8u;
      deepseek_mla_cp_async16(dst, src);
    } else {
      *reinterpret_cast<uint4 *>(dst) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  deepseek_mla_cp_async_commit();

  // Per-thread running softmax state for the two fragment rows this thread
  // owns (rows lane/4 and lane/4 + 8). Every warp keeps an identical copy,
  // updated redundantly from the shared per-tile row reductions, so no
  // cross-warp state writes are needed.
  const uint32_t row_lo = lane >> 2u;
  float m_state[2] = {-INFINITY, -INFINITY};
  float l_state[2] = {0.0f, 0.0f};
  // Output accumulators: this warp owns latent dims [warp*64, warp*64+64).
  float o_acc[8][4];
#pragma unroll
  for (uint32_t i = 0; i < 8u; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
      o_acc[i][j] = 0.0f;
    }
  }

  const uint32_t tiles = (list_len + kKT - 1u) / kKT;
  deepseek_mla_cp_async_wait_all();
  __syncthreads();

  for (uint32_t tile = 0; tile < tiles; ++tile) {
    const uint32_t base = tile * kKT;
    // Stage this tile's slot ids (-1 marks a masked column). Full-prefix
    // mode uses the identity list over [0, position].
    if (tid < kKT) {
      const uint32_t idx = base + tid;
      int32_t slot = -1;
      if (idx < list_len) {
        if (list != nullptr) {
          const int32_t candidate = list[idx];
          if (candidate >= 0 &&
              static_cast<uint32_t>(candidate) <= position) {
            slot = candidate;
          }
        } else {
          slot = static_cast<int32_t>(idx);
        }
      }
      slot_sm[tid] = slot;
    }
    __syncthreads();
    // Gather the KV tile (kKT rows x 1152B); 4 threads per row. Masked
    // rows load token 0 so the async copy stays valid; their scores are
    // forced to -inf below.
    {
      const uint32_t row = tid >> 2u;
      const uint32_t sub = tid & 3u;
      const int32_t slot = slot_sm[row];
      const uint32_t token = slot < 0 ? 0u : static_cast<uint32_t>(slot);
      const uint64_t token_base = kv_cache_token_base(
          layer_index, kv_block_count, kv_block_table, token, kWidth, 0);
      const uint16_t *src = kv_keys + token_base;
#pragma unroll
      for (uint32_t c = 0; c < kWidth / 8u / 4u; ++c) {
        const uint32_t col8 = sub * (kWidth / 8u / 4u) + c;
        deepseek_mla_cp_async16(&kv_sm[row * kStride + col8 * 8u],
                                &src[col8 * 8u]);
      }
    }
    deepseek_mla_cp_async_commit();
    deepseek_mla_cp_async_wait_all();
    __syncthreads();

    // S = Q.K^T for this warp's 8-token column slice, f32 accumulators.
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    {
      const uint32_t tok8 = warp * 8u;
#pragma unroll 4
      for (uint32_t kk = 0; kk < kWidth / 16u; ++kk) {
        uint32_t a0, a1, a2, a3, b0, b1;
        deepseek_mla_ldmatrix_x4(
            deepseek_mla_smem_u32addr(&q_sm[(lane & 15u) * kStride +
                                            kk * 16u + (lane >> 4u) * 8u]),
            a0, a1, a2, a3);
        deepseek_mla_ldmatrix_x2(
            deepseek_mla_smem_u32addr(
                &kv_sm[(tok8 + (lane & 7u)) * kStride + kk * 16u +
                       ((lane & 15u) >> 3u) * 8u]),
            b0, b1);
        deepseek_mla_mma_bf16(s0, s1, s2, s3, a0, a1, a2, a3, b0, b1);
      }
    }
    // Scale then mask; the two columns this thread holds are col_base and
    // col_base + 1 of the warp's slice.
    const uint32_t col_base = warp * 8u + (lane & 3u) * 2u;
    const bool valid0 = slot_sm[col_base] >= 0;
    const bool valid1 = slot_sm[col_base + 1u] >= 0;
    s0 = valid0 ? s0 * softmax_scale : -INFINITY;
    s1 = valid1 ? s1 * softmax_scale : -INFINITY;
    s2 = valid0 ? s2 * softmax_scale : -INFINITY;
    s3 = valid1 ? s3 * softmax_scale : -INFINITY;
    // Per-row max of this warp's 8 columns (reduction across the quad).
    float mx_lo = fmaxf(s0, s1);
    float mx_hi = fmaxf(s2, s3);
#pragma unroll
    for (uint32_t off = 1; off <= 2; off <<= 1) {
      mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffffu, mx_lo, off));
      mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffffu, mx_hi, off));
    }
    if ((lane & 3u) == 0u) {
      red_sm[row_lo * kWarps + warp] = mx_lo;
      red_sm[(row_lo + 8u) * kWarps + warp] = mx_hi;
    }
    __syncthreads();
    // Fold the per-warp maxima in fixed warp order; all threads compute the
    // same values redundantly.
    float m_new[2];
#pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
      const uint32_t r = row_lo + h * 8u;
      float mx = red_sm[r * kWarps];
#pragma unroll
      for (uint32_t w = 1; w < kWarps; ++w) {
        mx = fmaxf(mx, red_sm[r * kWarps + w]);
      }
      m_new[h] = fmaxf(m_state[h], mx);
    }
    __syncthreads();
    // P = exp(S - m_new) with masked columns forced to zero; publish the
    // per-warp row partial sums and the bf16 P tile for the P.V product.
    const float p0 = valid0 ? expf(s0 - m_new[0]) : 0.0f;
    const float p1 = valid1 ? expf(s1 - m_new[0]) : 0.0f;
    const float p2 = valid0 ? expf(s2 - m_new[1]) : 0.0f;
    const float p3 = valid1 ? expf(s3 - m_new[1]) : 0.0f;
    {
      const uint32_t pk0 =
          (static_cast<uint32_t>(deepseek_session_f32_to_bf16_bits(p1))
           << 16) |
          deepseek_session_f32_to_bf16_bits(p0);
      const uint32_t pk1 =
          (static_cast<uint32_t>(deepseek_session_f32_to_bf16_bits(p3))
           << 16) |
          deepseek_session_f32_to_bf16_bits(p2);
      *reinterpret_cast<uint32_t *>(&p_sm[row_lo * kPStride + col_base]) =
          pk0;
      *reinterpret_cast<uint32_t *>(
          &p_sm[(row_lo + 8u) * kPStride + col_base]) = pk1;
    }
    float sum_lo = p0 + p1;
    float sum_hi = p2 + p3;
#pragma unroll
    for (uint32_t off = 1; off <= 2; off <<= 1) {
      sum_lo += __shfl_xor_sync(0xffffffffu, sum_lo, off);
      sum_hi += __shfl_xor_sync(0xffffffffu, sum_hi, off);
    }
    if ((lane & 3u) == 0u) {
      red_sm[row_lo * kWarps + warp] = sum_lo;
      red_sm[(row_lo + 8u) * kWarps + warp] = sum_hi;
    }
    // Rescale the output accumulators by exp(m_old - m_new) per row while
    // the row sums land in shared memory.
    float alpha[2];
#pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
      alpha[h] =
          (m_state[h] == -INFINITY) ? 0.0f : expf(m_state[h] - m_new[h]);
      m_state[h] = m_new[h];
    }
#pragma unroll
    for (uint32_t i = 0; i < 8u; ++i) {
      o_acc[i][0] *= alpha[0];
      o_acc[i][1] *= alpha[0];
      o_acc[i][2] *= alpha[1];
      o_acc[i][3] *= alpha[1];
    }
    __syncthreads();
#pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
      const uint32_t r = row_lo + h * 8u;
      float sum = 0.0f;
#pragma unroll
      for (uint32_t w = 0; w < kWarps; ++w) {
        sum += red_sm[r * kWarps + w];
      }
      l_state[h] = l_state[h] * alpha[h] + sum;
    }
    // O += P.V over this tile (V is the first kLora columns of the staged
    // KV rows; the tile is read once for both products).
#pragma unroll
    for (uint32_t ks = 0; ks < kKT / 16u; ++ks) {
      uint32_t a0, a1, a2, a3;
      deepseek_mla_ldmatrix_x4(
          deepseek_mla_smem_u32addr(&p_sm[(lane & 15u) * kPStride +
                                          ks * 16u + (lane >> 4u) * 8u]),
          a0, a1, a2, a3);
#pragma unroll
      for (uint32_t nn = 0; nn < 8u; ++nn) {
        uint32_t b0, b1;
        const uint32_t dim = warp * 64u + nn * 8u;
        deepseek_mla_ldmatrix_x2_trans(
            deepseek_mla_smem_u32addr(
                &kv_sm[(ks * 16u + (lane & 15u)) * kStride + dim]),
            b0, b1);
        deepseek_mla_mma_bf16(o_acc[nn][0], o_acc[nn][1], o_acc[nn][2],
                              o_acc[nn][3], a0, a1, a2, a3, b0, b1);
      }
    }
    __syncthreads();
  }

  if (blockIdx.y == 0 && tid == 0 && deepseek_runtime_counters != nullptr) {
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  deepseek_runtime_counters +
                  kDeepSeekRuntimeCounterRawAttentionTokensScanned),
              static_cast<unsigned long long>(list_len));
  }

  // Epilogue: normalize by the row sum (guarded exactly like the previous
  // kernels) and write the attended latent as bf16.
#pragma unroll
  for (uint32_t h = 0; h < 2u; ++h) {
    const uint32_t r = row_lo + h * 8u;
    const uint32_t head = head_base + r;
    if (head >= heads) {
      continue;
    }
    const float l = l_state[h];
    const bool normalize = l > 0.0f && isfinite(l);
    uint16_t *out_row =
        attn_latent +
        (static_cast<uint64_t>(local_token) * heads + head) * kLora;
#pragma unroll
    for (uint32_t nn = 0; nn < 8u; ++nn) {
      const uint32_t dim = warp * 64u + nn * 8u + (lane & 3u) * 2u;
      float v0 = o_acc[nn][h * 2u];
      float v1 = o_acc[nn][h * 2u + 1u];
      if (normalize) {
        v0 /= l;
        v1 /= l;
      }
      const uint32_t packed =
          (static_cast<uint32_t>(deepseek_session_f32_to_bf16_bits(v1))
           << 16) |
          deepseek_session_f32_to_bf16_bits(v0);
      *reinterpret_cast<uint32_t *>(&out_row[dim]) = packed;
    }
  }
}

// Generic fallback for layouts whose dims do not fit the MMA tile design
// (kv_lora_rank != 512 or qk_rope != 64) or non-bf16 model dtypes. Shared
// verbatim by decode and prefill, so it keeps the same consistency-by-
// construction property; it is not expected on any production DeepSeek
// V3/V3.2 checkpoint.
__global__ void hf_deepseek_mla_fa_generic_kernel(
    SequenceLayerLayout layout, uint32_t layer_index, uint32_t dtype,
    uint32_t heads, uint32_t *step_cursor, uint32_t max_steps,
    uint32_t chunk_start, uint32_t token_count, const uint16_t *q_latent,
    const uint16_t *kv_keys, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint16_t *attn_latent,
    const int32_t *sparse_topk_slots, uint32_t sparse_topk_stride,
    const uint32_t *sparse_topk_count,
    uint64_t *deepseek_runtime_counters) {
  const uint32_t local_token = blockIdx.x;
  const uint32_t head = blockIdx.y;
  if (head >= heads || q_latent == nullptr || kv_keys == nullptr ||
      attn_latent == nullptr) {
    return;
  }
  uint32_t position = 0;
  const int32_t *list = nullptr;
  uint32_t list_len = 0;
  if (!deepseek_mla_resolve_row(step_cursor, max_steps, chunk_start,
                                token_count, local_token, sparse_topk_slots,
                                sparse_topk_stride, sparse_topk_count,
                                &position, &list, &list_len)) {
    return;
  }
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t width = kv_lora_rank + qk_rope;
  if (kv_lora_rank == 0 || qk_rope == 0) {
    return;
  }
  const float softmax_scale = deepseek_mla_attention_scale(
      layout,
      layout.deepseek_qk_nope_head_dim + layout.deepseek_qk_rope_head_dim);

  extern __shared__ float generic_shared[];
  float *latent_output = generic_shared;          // kv_lora_rank
  float *q_row = generic_shared + kv_lora_rank;   // width
  const uint16_t *q_src =
      q_latent + (static_cast<uint64_t>(local_token) * heads + head) * width;
  for (uint32_t dim = threadIdx.x; dim < width; dim += blockDim.x) {
    q_row[dim] = encoded_to_f32(q_src[dim], dtype);
    if (dim < kv_lora_rank) {
      latent_output[dim] = 0.0f;
    }
  }
  __syncthreads();

  float local_m = -INFINITY;
  float local_l = 0.0f;
  for (uint32_t index = 0; index < list_len; ++index) {
    uint32_t token = index;
    if (list != nullptr) {
      const int32_t slot = list[index];
      if (slot < 0) {
        continue;
      }
      token = static_cast<uint32_t>(slot);
    }
    if (token > position) {
      continue;
    }
    const uint64_t token_base = kv_cache_token_base(
        layer_index, kv_block_count, kv_block_table, token, width, 0);
    float score_part = 0.0f;
    for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
         latent += blockDim.x) {
      score_part +=
          q_row[latent] * encoded_to_f32(kv_keys[token_base + latent], dtype);
    }
    for (uint32_t dim = threadIdx.x; dim < qk_rope; dim += blockDim.x) {
      score_part +=
          q_row[kv_lora_rank + dim] *
          encoded_to_f32(kv_keys[token_base + kv_lora_rank + dim], dtype);
    }
    const float score = block_sum(score_part) * softmax_scale;
    const float next_m = fmaxf(local_m, score);
    const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
    const float new_scale = expf(score - next_m);
    const float value_scale = f32_to_model_dtype(new_scale, dtype);
    for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
         latent += blockDim.x) {
      latent_output[latent] =
          latent_output[latent] * old_scale +
          encoded_to_f32(kv_keys[token_base + latent], dtype) * value_scale;
    }
    local_l = local_l * old_scale + new_scale;
    local_m = next_m;
  }

  if (blockIdx.y == 0 && threadIdx.x == 0 &&
      deepseek_runtime_counters != nullptr) {
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  deepseek_runtime_counters +
                  kDeepSeekRuntimeCounterRawAttentionTokensScanned),
              static_cast<unsigned long long>(list_len));
  }

  const bool normalize = local_l > 0.0f && isfinite(local_l);
  uint16_t *out_row =
      attn_latent +
      (static_cast<uint64_t>(local_token) * heads + head) * kv_lora_rank;
  for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
       latent += blockDim.x) {
    const float value =
        normalize ? latent_output[latent] / local_l : latent_output[latent];
    out_row[latent] = f32_to_encoded(value, dtype);
  }
}

__global__ void hf_deepseek_mla_v_proj_tokens_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t heads, uint32_t *step_cursor, uint32_t max_steps,
    uint32_t chunk_start, uint32_t token_count, const uint16_t *attn_latent,
    uint16_t *attn_out, uint32_t attn_stride,
    uint64_t *deepseek_runtime_counters, uint32_t record_sparse_attention) {
  const uint32_t head = blockIdx.x;
  const uint32_t token_base = blockIdx.y * kDeepSeekMlaQLatentTokensPerBlock;
  if (head >= heads || token_base >= token_count || attn_latent == nullptr ||
      attn_out == nullptr || layout.w_v == kMissingOffset) {
    return;
  }
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t v_head = layout.deepseek_v_head_dim;
  if (kv_lora_rank == 0 || qk_nope == 0 || v_head == 0 ||
      kv_lora_rank > kDeepSeekMlaVProjMaxLora ||
      attn_stride < heads * v_head) {
    return;
  }
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  if (!bf16_storage && layout.deepseek_kv_b_scale == kMissingOffset) {
    return;
  }
  const uint32_t token_group =
      min(kDeepSeekMlaQLatentTokensPerBlock, token_count - token_base);

  // Stage the attended latents (dequantized once) for the token group.
  __shared__ float latent_shared[kDeepSeekMlaQLatentTokensPerBlock]
                                [kDeepSeekMlaVProjMaxLora];
  for (uint32_t idx = threadIdx.x; idx < token_group * kv_lora_rank;
       idx += blockDim.x) {
    const uint32_t t = idx / kv_lora_rank;
    const uint32_t latent = idx - t * kv_lora_rank;
    const uint16_t *row =
        attn_latent +
        (static_cast<uint64_t>(token_base + t) * heads + head) * kv_lora_rank;
    latent_shared[t][latent] = encoded_to_f32(row[latent], dtype);
  }
  __syncthreads();

  const uint32_t kv_b_cols = kv_lora_rank;
  const uint8_t *kv_b_weight =
      reinterpret_cast<const uint8_t *>(arena + layout.w_v);
  const uint16_t *kv_b_scale =
      bf16_storage ? nullptr : arena + layout.deepseek_kv_b_scale;
  const uint32_t kv_b_scale_cols = (kv_b_cols + 127u) / 128u;
  const bool full_output_hash = heads <= 4;

  for (uint32_t value = threadIdx.x; value < v_head; value += blockDim.x) {
    const uint32_t row = head * (qk_nope + v_head) + qk_nope + value;
    float sum[kDeepSeekMlaQLatentTokensPerBlock];
#pragma unroll
    for (uint32_t t = 0; t < kDeepSeekMlaQLatentTokensPerBlock; ++t) {
      sum[t] = 0.0f;
    }
    const uint32_t row_scale_base = (row / 128u) * kv_b_scale_cols;
    // Vectorized weight streaming (16 fp8 bytes / 8 bf16 halves per load).
    // The dequantized weight values and the ascending-latent accumulation
    // order are identical to the scalar path, so this is a pure memory
    // optimization with bit-identical results.
    const uint8_t *w_row_fp8 =
        bf16_storage ? nullptr
                     : kv_b_weight + static_cast<uint64_t>(row) * kv_b_cols;
    const uint16_t *w_row_bf16 =
        bf16_storage ? arena + layout.w_v +
                           static_cast<uint64_t>(row) * kv_b_cols
                     : nullptr;
    const bool vec16 =
        (kv_b_cols & 15u) == 0u &&
        ((reinterpret_cast<uintptr_t>(bf16_storage
                                          ? static_cast<const void *>(w_row_bf16)
                                          : static_cast<const void *>(w_row_fp8)) &
          15u) == 0u);
    for (uint32_t block_start = 0; block_start < kv_lora_rank;
         block_start += 128u) {
      const uint32_t block_end = min(block_start + 128u, kv_lora_rank);
      const float scale =
          bf16_storage
              ? 1.0f
              : f32_from_u16_slots(kv_b_scale,
                                   row_scale_base + block_start / 128u);
      if (vec16 && !bf16_storage) {
        for (uint32_t latent = block_start; latent < block_end;
             latent += 16u) {
          const uint4 raw =
              *reinterpret_cast<const uint4 *>(w_row_fp8 + latent);
          const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
          for (uint32_t j = 0; j < 16u; ++j) {
            const uint8_t bits = static_cast<uint8_t>(
                (words[j >> 2u] >> ((j & 3u) * 8u)) & 0xffu);
            const float weight =
                nerva::deepseek::f8_e4m3fn_bits_to_f32(bits) * scale;
#pragma unroll
            for (uint32_t t = 0; t < kDeepSeekMlaQLatentTokensPerBlock; ++t) {
              if (t < token_group) {
                sum[t] += latent_shared[t][latent + j] * weight;
              }
            }
          }
        }
      } else if (vec16 && bf16_storage) {
        for (uint32_t latent = block_start; latent < block_end;
             latent += 8u) {
          const uint4 raw =
              *reinterpret_cast<const uint4 *>(w_row_bf16 + latent);
          const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
          for (uint32_t j = 0; j < 8u; ++j) {
            const uint16_t half_bits = static_cast<uint16_t>(
                (words[j >> 1u] >> ((j & 1u) * 16u)) & 0xffffu);
            const float weight = encoded_to_f32(half_bits, kDTypeBF16);
#pragma unroll
            for (uint32_t t = 0; t < kDeepSeekMlaQLatentTokensPerBlock; ++t) {
              if (t < token_group) {
                sum[t] += latent_shared[t][latent + j] * weight;
              }
            }
          }
        }
      } else {
        for (uint32_t latent = block_start; latent < block_end; ++latent) {
          const float weight =
              bf16_storage
                  ? deepseek_bf16_weight(arena, layout.w_v,
                                         heads * (qk_nope + v_head),
                                         kv_b_cols, row, latent)
                  : nerva::deepseek::f8_e4m3fn_bits_to_f32(
                        kv_b_weight[static_cast<uint64_t>(row) * kv_b_cols +
                                    latent]) *
                        scale;
#pragma unroll
          for (uint32_t t = 0; t < kDeepSeekMlaQLatentTokensPerBlock; ++t) {
            if (t < token_group) {
              sum[t] += latent_shared[t][latent] * weight;
            }
          }
        }
      }
    }
#pragma unroll
    for (uint32_t t = 0; t < kDeepSeekMlaQLatentTokensPerBlock; ++t) {
      if (t >= token_group) {
        continue;
      }
      const uint32_t position = step_cursor != nullptr
                                    ? *step_cursor
                                    : chunk_start + token_base + t;
      if (position >= max_steps) {
        continue;
      }
      const uint16_t encoded = f32_to_encoded(sum[t], dtype);
      attn_out[static_cast<uint64_t>(token_base + t) * attn_stride +
               head * v_head + value] = encoded;
      if (record_sparse_attention != 0 && (full_output_hash || head == 0) &&
          deepseek_runtime_counters != nullptr) {
        const unsigned long long term =
            (static_cast<unsigned long long>(position) + 1ull) *
                1315423911ull ^
            (static_cast<unsigned long long>(head) + 1ull) * 2654435761ull ^
            (static_cast<unsigned long long>(value) + 1ull) * 97531ull ^
            static_cast<unsigned long long>(encoded);
        atomicAdd(reinterpret_cast<unsigned long long *>(
                      deepseek_runtime_counters +
                      kDeepSeekRuntimeCounterSparseAttentionOutputHash),
                  term);
      }
    }
  }
}
