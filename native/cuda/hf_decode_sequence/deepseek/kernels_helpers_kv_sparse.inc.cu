__device__ float deepseek_session_read_fp8_ds_mla_compressed_kv(
    const uint8_t *kv_cache, uint64_t kv_offset_bytes,
    uint32_t compressed_block_count, const SequenceLayerLayout &layout,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint32_t compressed_slot, uint32_t dim) {
  const uint32_t nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t head_dim = nope + rope;
  if (kv_cache == nullptr || dim >= head_dim) {
    return 0.0f;
  }
  const uint32_t packed_block_tokens =
      deepseek_v4_packed_kv_block_tokens(layout.deepseek_compress_ratio);
  const uint32_t logical_compressed_block = compressed_slot / packed_block_tokens;
  uint32_t compressed_block = 0;
  if (!deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, compressed_block_count,
          logical_compressed_block,
          packed_block_tokens * layout.deepseek_compress_ratio,
          &compressed_block)) {
    return 0.0f;
  }
  const uint32_t token_stride = nope + rope * 2u;
  const uint32_t scale_dim = nope / 64u + 1u;
  const uint32_t kv_pos = compressed_slot % packed_block_tokens;
  const uint64_t block_stride = deepseek_v4_round_up_u64(
      static_cast<uint64_t>(packed_block_tokens) *
          static_cast<uint64_t>(token_stride + scale_dim),
      kDeepSeekV4PackedKvAlignmentBytes);
  const uint8_t *block_ptr =
      kv_cache + kv_offset_bytes +
      static_cast<uint64_t>(compressed_block) * block_stride;
  const uint8_t *data_ptr =
      block_ptr + static_cast<uint64_t>(kv_pos) * token_stride;
  const uint8_t *scale_ptr =
      block_ptr + static_cast<uint64_t>(packed_block_tokens) * token_stride +
      static_cast<uint64_t>(kv_pos) * scale_dim;
  if (dim < nope) {
    const uint32_t scale_index = dim / 64u;
    const float scale =
        nerva::deepseek::e8m0_exponent_bits_to_f32(scale_ptr[scale_index]);
    return nerva::deepseek::f8_e4m3fn_bits_to_f32(data_ptr[dim]) * scale;
  }
  const uint32_t rope_local = dim - nope;
  const uint32_t offset = nope + rope_local * 2u;
  const uint16_t bits = static_cast<uint16_t>(data_ptr[offset]) |
                        (static_cast<uint16_t>(data_ptr[offset + 1u]) << 8u);
  return deepseek_session_bf16_bits_to_f32(bits);
}

__device__ bool deepseek_session_write_fp8_ds_mla_swa_kv(
    uint8_t *kv_cache, uint64_t kv_offset_bytes, uint32_t block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    const SequenceLayerLayout &layout, uint32_t position, const float *kv) {
  const uint32_t nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t head_dim = nope + rope;
  if (kv_cache == nullptr || kv == nullptr || head_dim == 0) {
    return false;
  }
  const uint32_t logical_block =
      position / kDeepSeekV4PackedKvDefaultBlockTokens;
  uint32_t block = 0;
  if (!deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, block_count, logical_block,
          kDeepSeekV4PackedKvDefaultBlockTokens, &block)) {
    return false;
  }
  const uint32_t token_stride = nope + rope * 2u;
  const uint32_t scale_dim = nope / 64u + 1u;
  const uint32_t kv_pos =
      position % kDeepSeekV4PackedKvDefaultBlockTokens;
  const uint64_t block_stride = deepseek_v4_round_up_u64(
      static_cast<uint64_t>(kDeepSeekV4PackedKvDefaultBlockTokens) *
          static_cast<uint64_t>(token_stride + scale_dim),
      kDeepSeekV4PackedKvAlignmentBytes);
  uint8_t *block_ptr =
      kv_cache + kv_offset_bytes + static_cast<uint64_t>(block) * block_stride;
  uint8_t *data_ptr = block_ptr + static_cast<uint64_t>(kv_pos) * token_stride;
  uint8_t *scale_ptr =
      block_ptr +
      static_cast<uint64_t>(kDeepSeekV4PackedKvDefaultBlockTokens) *
          token_stride +
      static_cast<uint64_t>(kv_pos) * scale_dim;
  const uint32_t quant_block = 64u;
  for (uint32_t scale_index = 0; scale_index < scale_dim; ++scale_index) {
    const uint32_t start = scale_index * quant_block;
    const uint32_t end =
        start + quant_block < nope ? start + quant_block : nope;
    float absmax = 0.0f;
    for (uint32_t dim = start; dim < end; ++dim) {
      const float quant_input = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(kv[dim]));
      absmax = fmaxf(absmax, fabsf(quant_input));
    }
    const float raw = fmaxf(absmax, 1.0e-4f) / 448.0f;
    const float scale = exp2f(ceilf(log2f(raw)));
    scale_ptr[scale_index] =
        start < nope ? deepseek_session_encode_e8m0_scale(scale) : 0u;
    for (uint32_t dim = start; dim < end; ++dim) {
      const float quant_input = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(kv[dim]));
      const float scaled = fminf(fmaxf(quant_input / scale, -448.0f), 448.0f);
      data_ptr[dim] = deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
    }
  }
  for (uint32_t dim = nope; dim < head_dim; ++dim) {
    const uint16_t bits = deepseek_session_f32_to_bf16_bits(kv[dim]);
    const uint32_t rope_local = dim - nope;
    data_ptr[nope + rope_local * 2u] = static_cast<uint8_t>(bits & 0xffu);
    data_ptr[nope + rope_local * 2u + 1u] =
        static_cast<uint8_t>(bits >> 8u);
  }
  return true;
}

__device__ bool deepseek_session_write_v32_fp8_ds_mla_kv(
    uint8_t *kv_cache, uint64_t kv_offset_bytes, uint32_t block_count,
    const uint32_t *kv_block_table, uint32_t kv_block_count,
    const SequenceLayerLayout &layout, uint32_t position, uint32_t dtype,
    const uint16_t *kv_latent_norm, const float *kv_a, float rope_theta) {
  if (kv_cache == nullptr || kv_latent_norm == nullptr || kv_a == nullptr ||
      block_count == 0 ||
      layout.attention_kind != kAttentionKindDeepSeekMla ||
      layout.deepseek_mode != kDeepSeekModeV32MlaIndexer ||
      layout.deepseek_kv_lora_rank != kDeepSeekV32PackedKvNopeBytes ||
      layout.deepseek_qk_rope_head_dim != kDeepSeekV32PackedKvRopeValues) {
    return false;
  }
  const uint32_t logical_block = position / kDeepSeekV32PackedKvBlockTokens;
  uint32_t physical_block = 0;
  if (!deepseek_v32_packed_physical_block(
          kv_block_table, kv_block_count, block_count, logical_block,
          &physical_block)) {
    return false;
  }
  const uint32_t kv_pos = position % kDeepSeekV32PackedKvBlockTokens;
  uint8_t *row =
      kv_cache + kv_offset_bytes +
      (static_cast<uint64_t>(physical_block) * kDeepSeekV32PackedKvBlockTokens +
       kv_pos) *
          kDeepSeekV32PackedKvTokenBytes;
  uint8_t *nope_ptr = row;
  float *scale_ptr = reinterpret_cast<float *>(
      row + kDeepSeekV32PackedKvNopeBytes);
  uint8_t *rope_ptr =
      row + kDeepSeekV32PackedKvNopeBytes + kDeepSeekV32PackedKvScaleBytes;

  constexpr uint32_t kScaleBlockValues = 128;
  constexpr uint32_t kScaleCount =
      kDeepSeekV32PackedKvNopeBytes / kScaleBlockValues;
  for (uint32_t scale_index = 0; scale_index < kScaleCount; ++scale_index) {
    const uint32_t start = scale_index * kScaleBlockValues;
    float absmax = 0.0f;
    for (uint32_t dim = 0; dim < kScaleBlockValues; ++dim) {
      const float value = encoded_to_f32(kv_latent_norm[start + dim], dtype);
      const float quant_input = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(value));
      absmax = fmaxf(absmax, fabsf(quant_input));
    }
    const float raw = fmaxf(absmax, 1.0e-4f) / 448.0f;
    const float scale = exp2f(ceilf(log2f(raw)));
    scale_ptr[scale_index] = scale;
    for (uint32_t dim = 0; dim < kScaleBlockValues; ++dim) {
      const uint32_t source_dim = start + dim;
      const float value = encoded_to_f32(kv_latent_norm[source_dim], dtype);
      const float quant_input = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(value));
      const float scaled = fminf(fmaxf(quant_input / scale, -448.0f), 448.0f);
      nope_ptr[source_dim] =
          deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
    }
  }

  for (uint32_t dim = 0; dim < kDeepSeekV32PackedKvRopeValues; ++dim) {
    float value = kv_a[kDeepSeekV32PackedKvNopeBytes + dim];
    if ((kDeepSeekV32PackedKvRopeValues & 1u) == 0u) {
      const uint32_t even = dim & ~1u;
      const uint32_t odd = even + 1u;
      value = deepseek_rope_value_gptj(
          kv_a[kDeepSeekV32PackedKvNopeBytes + even],
          kv_a[kDeepSeekV32PackedKvNopeBytes + odd], dim,
          kDeepSeekV32PackedKvRopeValues, position, rope_theta, layout);
    }
    const uint16_t bits = deepseek_session_f32_to_bf16_bits(value);
    rope_ptr[dim * 2u] = static_cast<uint8_t>(bits & 0xffu);
    rope_ptr[dim * 2u + 1u] = static_cast<uint8_t>(bits >> 8u);
  }
  return true;
}

__device__ float deepseek_session_read_fp8_ds_mla_swa_kv(
    const uint8_t *kv_cache, uint64_t kv_offset_bytes, uint32_t block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    const SequenceLayerLayout &layout, uint32_t position, uint32_t dim) {
  const uint32_t nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t head_dim = nope + rope;
  if (kv_cache == nullptr || dim >= head_dim) {
    return 0.0f;
  }
  const uint32_t logical_block =
      position / kDeepSeekV4PackedKvDefaultBlockTokens;
  uint32_t block = 0;
  if (!deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, block_count, logical_block,
          kDeepSeekV4PackedKvDefaultBlockTokens, &block)) {
    return 0.0f;
  }
  const uint32_t token_stride = nope + rope * 2u;
  const uint32_t scale_dim = nope / 64u + 1u;
  const uint32_t kv_pos =
      position % kDeepSeekV4PackedKvDefaultBlockTokens;
  const uint64_t block_stride = deepseek_v4_round_up_u64(
      static_cast<uint64_t>(kDeepSeekV4PackedKvDefaultBlockTokens) *
          static_cast<uint64_t>(token_stride + scale_dim),
      kDeepSeekV4PackedKvAlignmentBytes);
  const uint8_t *block_ptr =
      kv_cache + kv_offset_bytes + static_cast<uint64_t>(block) * block_stride;
  const uint8_t *data_ptr =
      block_ptr + static_cast<uint64_t>(kv_pos) * token_stride;
  const uint8_t *scale_ptr =
      block_ptr +
      static_cast<uint64_t>(kDeepSeekV4PackedKvDefaultBlockTokens) *
          token_stride +
      static_cast<uint64_t>(kv_pos) * scale_dim;
  if (dim < nope) {
    const uint32_t scale_index = dim / 64u;
    const float scale =
        nerva::deepseek::e8m0_exponent_bits_to_f32(scale_ptr[scale_index]);
    return nerva::deepseek::f8_e4m3fn_bits_to_f32(data_ptr[dim]) * scale;
  }
  const uint32_t rope_local = dim - nope;
  const uint32_t offset = nope + rope_local * 2u;
  const uint16_t bits = static_cast<uint16_t>(data_ptr[offset]) |
                        (static_cast<uint16_t>(data_ptr[offset + 1u]) << 8u);
  return deepseek_session_bf16_bits_to_f32(bits);
}

__device__ float deepseek_session_read_indexer_fp8_compressed_kv(
    const uint8_t *kv_cache, uint64_t kv_offset_bytes,
    uint32_t compressed_block_count, const SequenceLayerLayout &layout,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint32_t compressed_slot, uint32_t dim) {
  const uint32_t head_dim = layout.deepseek_index_head_dim;
  const uint32_t scale_dim = deepseek_device_scale_dim(head_dim) * sizeof(float);
  if (kv_cache == nullptr || head_dim == 0 || dim >= head_dim ||
      scale_dim < sizeof(float)) {
    return 0.0f;
  }
  const uint32_t packed_block_tokens =
      deepseek_v4_packed_kv_block_tokens(layout.deepseek_compress_ratio);
  const uint32_t logical_compressed_block = compressed_slot / packed_block_tokens;
  uint32_t compressed_block = 0;
  if (!deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, compressed_block_count,
          logical_compressed_block,
          packed_block_tokens * layout.deepseek_compress_ratio,
          &compressed_block)) {
    return 0.0f;
  }
  const uint32_t kv_pos = compressed_slot % packed_block_tokens;
  const uint64_t block_stride = deepseek_v4_round_up_u64(
      static_cast<uint64_t>(packed_block_tokens) *
          static_cast<uint64_t>(head_dim + scale_dim),
      kDeepSeekV4PackedKvAlignmentBytes);
  const uint8_t *block_ptr =
      kv_cache + kv_offset_bytes +
      static_cast<uint64_t>(compressed_block) * block_stride;
  const uint8_t *data_ptr =
      block_ptr + static_cast<uint64_t>(kv_pos) * head_dim;
  const uint8_t *scale_ptr =
      block_ptr + static_cast<uint64_t>(packed_block_tokens) * head_dim +
      static_cast<uint64_t>(kv_pos) * scale_dim;
  const uint32_t scale_index = dim / 128u;
  const float scale =
      reinterpret_cast<const float *>(scale_ptr)[scale_index];
  return nerva::deepseek::f8_e4m3fn_bits_to_f32(data_ptr[dim]) * scale;
}

__device__ float deepseek_session_indexer_query_rope_value(
    const float *query_head, uint32_t head_dim, uint32_t rope_head_dim,
    uint32_t dim, uint32_t position, float rope_theta,
    const SequenceLayerLayout &layout) {
  if (rope_theta <= 0.0f || rope_head_dim < 2 || rope_head_dim > head_dim) {
    return query_head[dim];
  }
  const uint32_t nope_dim = head_dim - rope_head_dim;
  if (dim < nope_dim) {
    return query_head[dim];
  }
  const uint32_t rope_local = dim - nope_dim;
  const uint32_t pair = rope_local / 2u;
  const uint32_t even_dim = nope_dim + pair * 2u;
  const uint32_t odd_dim = even_dim + 1u;
  if (odd_dim >= head_dim) {
    return query_head[dim];
  }
  const float even = query_head[even_dim];
  const float odd = query_head[odd_dim];
  const float angle =
      static_cast<float>(position) *
      deepseek_rope_inv_freq(layout, pair, rope_head_dim, rope_theta);
  float sin_value = 0.0f;
  float cos_value = 0.0f;
  sincosf(angle, &sin_value, &cos_value);
  const float magnitude = deepseek_rope_magnitude(layout);
  return magnitude *
         ((rope_local & 1u) == 0u ? even * cos_value - odd * sin_value
                                  : odd * cos_value + even * sin_value);
}

__device__ bool deepseek_session_sparse_score_is_better(
    float candidate, int32_t slot, float current, int32_t current_slot) {
  if (!isfinite(candidate)) {
    return false;
  }
  if (current_slot < 0) {
    return true;
  }
  return candidate > current ||
         (candidate == current && slot >= 0 && slot < current_slot);
}

__device__ void deepseek_session_sparse_sort_desc(float *scores, int32_t *slots,
                                                  uint32_t sort_size) {
  for (uint32_t width = 2u; width <= sort_size; width <<= 1u) {
    for (uint32_t stride = width >> 1u; stride != 0u; stride >>= 1u) {
      for (uint32_t index = threadIdx.x; index < sort_size;
           index += blockDim.x) {
        const uint32_t other = index ^ stride;
        if (other <= index) {
          continue;
        }
        const bool descending = (index & width) == 0u;
        const float left_score = scores[index];
        const int32_t left_slot = slots[index];
        const float right_score = scores[other];
        const int32_t right_slot = slots[other];
        const bool left_before_right = deepseek_session_sparse_score_is_better(
            left_score, left_slot, right_score, right_slot);
        const bool swap = descending ? !left_before_right : left_before_right;
        if (swap) {
          scores[index] = right_score;
          slots[index] = right_slot;
          scores[other] = left_score;
          slots[other] = left_slot;
        }
      }
      __syncthreads();
    }
  }
}

__device__ uint32_t deepseek_session_select_v4_c4_sparse_slots(
    uint16_t *arena, const SequenceLayerLayout &layout, const float *qr_norm,
    const uint16_t *projection_input, uint32_t dtype, uint32_t hidden,
    uint32_t q_lora_rank, uint32_t position, float rope_theta,
    const uint8_t *indexer_kv, uint64_t indexer_kv_offset_bytes,
    uint32_t indexer_kv_block_count, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint32_t compressed_attention_tokens,
    int32_t *topk_slots, float *topk_scores, float *indexer_query,
    uint32_t *scored_candidates, unsigned long long *selection_hash_out) {
  if (scored_candidates != nullptr) {
    *scored_candidates = 0;
  }
  if (selection_hash_out != nullptr) {
    *selection_hash_out = 0ull;
  }
  if (layout.deepseek_mode != kDeepSeekModeV4CompressedIndexer ||
      (layout.deepseek_flags & kDeepSeekFlagSparseIndexer) == 0 ||
      layout.deepseek_compress_ratio != 4 ||
      layout.deepseek_index_topk == 0 ||
      layout.deepseek_index_n_heads == 0 ||
      layout.deepseek_index_head_dim == 0 ||
      layout.deepseek_index_n_heads > kDeepSeekSessionMaxIndexerHeads ||
      layout.deepseek_index_head_dim > kDeepSeekSessionMaxCompressHeadSize ||
      static_cast<uint64_t>(layout.deepseek_index_n_heads) *
              layout.deepseek_index_head_dim >
          kDeepSeekSessionMaxIndexerQueryValues ||
      compressed_attention_tokens == 0 || indexer_kv == nullptr ||
      indexer_kv_block_count == 0 ||
      layout.deepseek_indexer_q == kMissingOffset ||
      layout.deepseek_indexer_q_scale == kMissingOffset ||
      layout.deepseek_indexer_weights == kMissingOffset) {
    return 0;
  }

  const uint32_t index_heads = layout.deepseek_index_n_heads;
  const uint32_t index_head_dim = layout.deepseek_index_head_dim;
  const uint32_t query_rows = index_heads * index_head_dim;
  const uint32_t rope_head_dim =
      layout.deepseek_qk_rope_head_dim <= index_head_dim
          ? layout.deepseek_qk_rope_head_dim
          : 0u;
  const uint32_t topk_limit =
      min(min(layout.deepseek_index_topk, compressed_attention_tokens),
          kDeepSeekSessionMaxSparseTopK);
  if (topk_limit == 0) {
    return 0;
  }
  if (topk_limit >= compressed_attention_tokens) {
    for (uint32_t slot = 0; slot < compressed_attention_tokens; ++slot) {
      topk_slots[slot] = static_cast<int32_t>(slot);
      topk_scores[slot] = INFINITY;
    }
  } else {
    if (scored_candidates != nullptr) {
      *scored_candidates = compressed_attention_tokens;
    }

    float indexer_weights[kDeepSeekSessionMaxIndexerHeads];
    for (uint32_t head = 0; head < index_heads; ++head) {
      float weight_sum = 0.0f;
      for (uint32_t col = 0; col < hidden; ++col) {
        weight_sum +=
            encoded_to_f32(arena[layout.deepseek_indexer_weights +
                                 static_cast<uint64_t>(head) * hidden + col],
                           kDTypeBF16) *
            encoded_to_f32(projection_input[col], dtype);
      }
      indexer_weights[head] = weight_sum;
    }

    for (uint32_t row = 0; row < query_rows; ++row) {
      float sum = 0.0f;
      for (uint32_t col = 0; col < q_lora_rank; ++col) {
        sum += deepseek_fp8_e8m0_scaled_weight(
                   arena, layout.deepseek_indexer_q,
                   layout.deepseek_indexer_q_scale, query_rows, q_lora_rank,
                   row, col) *
               qr_norm[col];
      }
      indexer_query[row] = sum;
    }

    for (uint32_t head = 0; head < index_heads; ++head) {
      float *query_head = indexer_query + head * index_head_dim;
      float absmax = 0.0f;
      for (uint32_t dim = 0; dim < index_head_dim; ++dim) {
        float value = deepseek_session_indexer_query_rope_value(
            query_head, index_head_dim, rope_head_dim, dim, position,
            rope_theta, layout);
        if (dim >= index_head_dim - rope_head_dim) {
          value = deepseek_session_bf16_bits_to_f32(
              deepseek_session_f32_to_bf16_bits(value));
        }
        query_head[dim] = value;
        absmax = fmaxf(absmax, fabsf(value));
      }
      const float raw = fmaxf(absmax, 1.0e-4f) / 448.0f;
      const float q_scale = exp2f(ceilf(log2f(raw)));
      for (uint32_t dim = 0; dim < index_head_dim; ++dim) {
        const float scaled = fminf(fmaxf(query_head[dim] / q_scale, -448.0f),
                                   448.0f);
        const uint8_t q_bits =
            deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
        query_head[dim] =
            nerva::deepseek::f8_e4m3fn_bits_to_f32(q_bits) * q_scale;
      }
    }

    for (uint32_t rank = 0; rank < topk_limit; ++rank) {
      topk_slots[rank] = -1;
      topk_scores[rank] = -INFINITY;
    }

    const float softmax_scale = rsqrtf(static_cast<float>(index_head_dim));
    const float head_scale = rsqrtf(static_cast<float>(index_heads));
    for (uint32_t slot = 0; slot < compressed_attention_tokens; ++slot) {
      float score = 0.0f;
      for (uint32_t head = 0; head < index_heads; ++head) {
        float dot = 0.0f;
        const float *query_head = indexer_query + head * index_head_dim;
        for (uint32_t dim = 0; dim < index_head_dim; ++dim) {
          dot += query_head[dim] *
                 deepseek_session_read_indexer_fp8_compressed_kv(
                     indexer_kv, indexer_kv_offset_bytes,
                     indexer_kv_block_count, layout, kv_block_count,
                     kv_block_table, slot, dim);
        }
        score += indexer_weights[head] * softmax_scale * head_scale * dot;
      }
      const int32_t slot_i32 = static_cast<int32_t>(slot);
      for (uint32_t rank = 0; rank < topk_limit; ++rank) {
        if (!deepseek_session_sparse_score_is_better(
                score, slot_i32, topk_scores[rank], topk_slots[rank])) {
          continue;
        }
        for (uint32_t shift = topk_limit - 1u; shift > rank; --shift) {
          topk_slots[shift] = topk_slots[shift - 1u];
          topk_scores[shift] = topk_scores[shift - 1u];
        }
        topk_slots[rank] = slot_i32;
        topk_scores[rank] = score;
        break;
      }
    }
  }

  uint32_t selected = 0;
  unsigned long long selection_hash = 0ull;
  for (uint32_t rank = 0; rank < topk_limit; ++rank) {
    if (topk_slots[rank] >= 0) {
      ++selected;
      selection_hash +=
          (static_cast<unsigned long long>(position) + 1ull) *
              1315423911ull ^
          (static_cast<unsigned long long>(rank) + 1ull) * 2654435761ull ^
          (static_cast<unsigned long long>(topk_slots[rank]) + 1ull);
    }
  }
  if (selection_hash_out != nullptr) {
    *selection_hash_out = selection_hash;
  }
  return selected;
}

__device__ uint32_t deepseek_session_compressed_attention_count(
    const uint8_t *kv_cache, const SequenceLayerLayout &layout,
    uint32_t compressed_block_count, uint32_t position) {
  if (kv_cache == nullptr ||
      (layout.deepseek_mode != kDeepSeekModeV4Compressed &&
       layout.deepseek_mode != kDeepSeekModeV4CompressedIndexer) ||
      layout.deepseek_compress_ratio <= 1 || compressed_block_count == 0) {
    return 0;
  }
  uint32_t compressed_tokens =
      (position + 1u) / layout.deepseek_compress_ratio;
  const uint32_t capacity =
      compressed_block_count *
      deepseek_v4_packed_kv_block_tokens(layout.deepseek_compress_ratio);
  if (compressed_tokens > capacity) {
    compressed_tokens = capacity;
  }
  return compressed_tokens;
}
