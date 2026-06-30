__global__ void hf_deepseek_v32_indexer_kv_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t *step_cursor, uint32_t max_steps,
    float rope_theta, const uint16_t *projection_input,
    uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint64_t *deepseek_runtime_counters) {
  if (blockIdx.x != 0 || threadIdx.x != 0 ||
      (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  if (arena == nullptr || projection_input == nullptr ||
      deepseek_indexer_kv == nullptr || deepseek_indexer_kv_block_count == 0 ||
      layout.attention_kind != kAttentionKindDeepSeekMla ||
      layout.deepseek_mode != kDeepSeekModeV32MlaIndexer ||
      (layout.deepseek_flags & kDeepSeekFlagSparseIndexer) == 0 ||
      layout.deepseek_index_head_dim == 0 ||
      layout.deepseek_index_head_dim > kDeepSeekSessionMaxCompressHeadSize ||
      layout.deepseek_indexer_k == kMissingOffset ||
      layout.deepseek_indexer_k_scale == kMissingOffset ||
      layout.deepseek_indexer_k_norm == kMissingOffset ||
      layout.deepseek_indexer_k_norm_bias == kMissingOffset) {
    return;
  }

  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t block = position / kDeepSeekV32IndexerKvBlockTokens;
  if (block >= deepseek_indexer_kv_block_count) {
    return;
  }

  const uint32_t head_dim = layout.deepseek_index_head_dim;
  float values[kDeepSeekSessionMaxCompressHeadSize];
  float mean = 0.0f;
  for (uint32_t row = 0; row < head_dim; ++row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      sum += deepseek_fp8_scaled_weight(
                 arena, layout.deepseek_indexer_k,
                 layout.deepseek_indexer_k_scale, head_dim, hidden, row,
                 col) *
             encoded_to_f32(projection_input[col], dtype);
    }
    values[row] = sum;
    mean += sum;
  }
  mean /= static_cast<float>(head_dim);

  float variance = 0.0f;
  for (uint32_t row = 0; row < head_dim; ++row) {
    const float centered = values[row] - mean;
    variance += centered * centered;
  }
  const float inv_std =
      rsqrtf(variance / static_cast<float>(head_dim) + 1.0e-6f);
  for (uint32_t row = 0; row < head_dim; ++row) {
    const float weight = f32_from_u16_slots(arena + layout.deepseek_indexer_k_norm,
                                            row);
    const float bias = f32_from_u16_slots(
        arena + layout.deepseek_indexer_k_norm_bias, row);
    values[row] = (values[row] - mean) * inv_std * weight + bias;
  }

  const uint32_t rope_dim =
      layout.deepseek_qk_rope_head_dim <= head_dim
          ? layout.deepseek_qk_rope_head_dim
          : 0u;
  const uint32_t rope_half = rope_dim / 2u;
  for (uint32_t offset = 0; offset < rope_half; ++offset) {
    const uint32_t left = offset;
    const uint32_t right = offset + rope_half;
    const float left_value = values[left];
    const float right_value = values[right];
    values[left] = deepseek_rope_value_serial(
        left_value, right_value, offset, rope_dim, position, rope_theta,
        false);
    values[right] = deepseek_rope_value_serial(
        left_value, right_value, offset, rope_dim, position, rope_theta,
        true);
  }

  float absmax = 0.0f;
  for (uint32_t dim = 0; dim < head_dim; ++dim) {
    values[dim] = deepseek_session_bf16_bits_to_f32(
        deepseek_session_f32_to_bf16_bits(values[dim]));
    absmax = fmaxf(absmax, fabsf(values[dim]));
  }
  const float scale = exp2f(ceilf(log2f(fmaxf(absmax, 1.0e-4f) / 448.0f)));

  const uint32_t scale_bytes =
      ((head_dim + 127u) / 128u) * sizeof(float);
  const uint64_t page_bytes =
      static_cast<uint64_t>(kDeepSeekV32IndexerKvBlockTokens) *
      (static_cast<uint64_t>(head_dim) + scale_bytes);
  const uint32_t block_offset =
      position % kDeepSeekV32IndexerKvBlockTokens;
  uint8_t *block_ptr = deepseek_indexer_kv +
                       deepseek_indexer_kv_offset_bytes +
                       static_cast<uint64_t>(block) * page_bytes;
  const uint32_t tile_block_id =
      block_offset / kDeepSeekV32IndexerKvTileTokens;
  const uint32_t tile_block_offset =
      block_offset % kDeepSeekV32IndexerKvTileTokens;
  for (uint32_t dim = 0; dim < head_dim; ++dim) {
    const float scaled = fminf(fmaxf(values[dim] / scale, -448.0f), 448.0f);
    const uint32_t tile_store_offset =
        (dim / kDeepSeekV32IndexerKvTileHeadBytes) *
            kDeepSeekV32IndexerKvTileTokens *
            kDeepSeekV32IndexerKvTileHeadBytes +
        (dim % kDeepSeekV32IndexerKvTileHeadBytes);
    const uint64_t value_offset =
        static_cast<uint64_t>(tile_block_id) *
            kDeepSeekV32IndexerKvTileTokens * head_dim +
        static_cast<uint64_t>(tile_block_offset) *
            kDeepSeekV32IndexerKvTileHeadBytes +
        tile_store_offset;
    block_ptr[value_offset] =
        deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
  }

  uint8_t *scale_ptr =
      block_ptr +
      static_cast<uint64_t>(kDeepSeekV32IndexerKvBlockTokens) * head_dim +
      static_cast<uint64_t>(block_offset) * scale_bytes;
  for (uint32_t scale_index = 0; scale_index < scale_bytes / sizeof(float);
       ++scale_index) {
    reinterpret_cast<float *>(scale_ptr)[scale_index] = scale;
  }
  if (deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterIndexerKvWrites),
        1ull);
  }
}

__device__ __forceinline__ bool deepseek_v32_indexer_query_state_supported(
    const SequenceLayerLayout &layout) {
  return layout.attention_kind == kAttentionKindDeepSeekMla &&
         layout.deepseek_mode == kDeepSeekModeV32MlaIndexer &&
         (layout.deepseek_flags & kDeepSeekFlagSparseIndexer) != 0 &&
         layout.deepseek_q_lora_rank != 0 &&
         layout.deepseek_index_n_heads != 0 &&
         layout.deepseek_index_head_dim != 0 &&
         layout.deepseek_index_n_heads <= kDeepSeekSessionMaxIndexerHeads &&
         layout.deepseek_index_head_dim <= kDeepSeekSessionMaxCompressHeadSize &&
         static_cast<uint64_t>(layout.deepseek_index_n_heads) *
                 layout.deepseek_index_head_dim <=
             kDeepSeekSessionMaxIndexerQueryValues &&
         layout.deepseek_indexer_q != kMissingOffset &&
         layout.deepseek_indexer_q_scale != kMissingOffset &&
         layout.deepseek_indexer_weights != kMissingOffset;
}

__device__ __forceinline__ uint64_t
deepseek_v32_indexer_query_state_q_scale_offset_bytes(
    const SequenceLayerLayout &layout) {
  const uint64_t query_bytes =
      static_cast<uint64_t>(layout.deepseek_index_n_heads) *
      layout.deepseek_index_head_dim;
  return deepseek_v4_round_up_u64(query_bytes, sizeof(float));
}

__device__ __forceinline__ uint64_t
deepseek_v32_indexer_query_state_weights_offset_bytes(
    const SequenceLayerLayout &layout) {
  return deepseek_v32_indexer_query_state_q_scale_offset_bytes(layout) +
         static_cast<uint64_t>(layout.deepseek_index_n_heads) * sizeof(float);
}

__device__ __forceinline__ uint64_t
deepseek_v32_indexer_query_state_token_bytes(
    const SequenceLayerLayout &layout) {
  return deepseek_v32_indexer_query_state_weights_offset_bytes(layout) +
         static_cast<uint64_t>(layout.deepseek_index_n_heads) * sizeof(float);
}

__global__ void hf_deepseek_v32_indexer_weight_state_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t *step_cursor, uint32_t max_steps,
    const uint16_t *projection_input, uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes) {
  if (blockIdx.x != 0 || threadIdx.x != 0 ||
      (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  if (arena == nullptr || projection_input == nullptr ||
      deepseek_indexer_state == nullptr ||
      !deepseek_v32_indexer_query_state_supported(layout)) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint64_t token_bytes =
      deepseek_v32_indexer_query_state_token_bytes(layout);
  uint8_t *token_ptr = deepseek_indexer_state +
                       deepseek_indexer_state_offset_bytes +
                       static_cast<uint64_t>(position) * token_bytes;
  auto *weights = reinterpret_cast<float *>(
      token_ptr +
      deepseek_v32_indexer_query_state_weights_offset_bytes(layout));
  for (uint32_t head = 0; head < layout.deepseek_index_n_heads; ++head) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      sum += encoded_to_f32(arena[layout.deepseek_indexer_weights +
                                  static_cast<uint64_t>(head) * hidden + col],
                            kDTypeBF16) *
             encoded_to_f32(projection_input[col], dtype);
    }
    weights[head] = sum;
  }
}

__global__ void hf_deepseek_v32_indexer_query_state_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t q_lora_rank, uint32_t *step_cursor, uint32_t max_steps,
    float rope_theta, const uint16_t *qr_norm,
    uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    uint64_t *deepseek_runtime_counters) {
  if (blockIdx.x != 0 || threadIdx.x != 0 ||
      (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  if (arena == nullptr || qr_norm == nullptr ||
      deepseek_indexer_state == nullptr ||
      !deepseek_v32_indexer_query_state_supported(layout) ||
      q_lora_rank != layout.deepseek_q_lora_rank) {
    return;
  }

  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t index_heads = layout.deepseek_index_n_heads;
  const uint32_t index_head_dim = layout.deepseek_index_head_dim;
  const uint32_t query_rows = index_heads * index_head_dim;
  const uint64_t token_bytes =
      deepseek_v32_indexer_query_state_token_bytes(layout);
  uint8_t *token_ptr = deepseek_indexer_state +
                       deepseek_indexer_state_offset_bytes +
                       static_cast<uint64_t>(position) * token_bytes;
  uint8_t *q_fp8 = token_ptr;
  auto *q_scales = reinterpret_cast<float *>(
      token_ptr +
      deepseek_v32_indexer_query_state_q_scale_offset_bytes(layout));
  auto *weights = reinterpret_cast<float *>(
      token_ptr +
      deepseek_v32_indexer_query_state_weights_offset_bytes(layout));

  float query[kDeepSeekSessionMaxIndexerQueryValues];
  for (uint32_t row = 0; row < query_rows; ++row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < q_lora_rank; ++col) {
      sum += deepseek_fp8_scaled_weight(
                 arena, layout.deepseek_indexer_q,
                 layout.deepseek_indexer_q_scale, query_rows, q_lora_rank,
                 row, col) *
             encoded_to_f32(qr_norm[col], dtype);
    }
    query[row] = sum;
  }

  const uint32_t rope_dim =
      layout.deepseek_qk_rope_head_dim <= index_head_dim
          ? layout.deepseek_qk_rope_head_dim
          : 0u;
  const uint32_t rope_half = rope_dim / 2u;
  const float softmax_scale = rsqrtf(static_cast<float>(index_head_dim));
  const float head_scale = rsqrtf(static_cast<float>(index_heads));
  for (uint32_t head = 0; head < index_heads; ++head) {
    float *query_head = query + head * index_head_dim;
    for (uint32_t offset = 0; offset < rope_half; ++offset) {
      const uint32_t left = offset;
      const uint32_t right = offset + rope_half;
      const float left_value = query_head[left];
      const float right_value = query_head[right];
      query_head[left] = deepseek_rope_value_serial(
          left_value, right_value, offset, rope_dim, position, rope_theta,
          false);
      query_head[right] = deepseek_rope_value_serial(
          left_value, right_value, offset, rope_dim, position, rope_theta,
          true);
    }

    float absmax = 0.0f;
    for (uint32_t dim = 0; dim < index_head_dim; ++dim) {
      query_head[dim] = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(query_head[dim]));
      absmax = fmaxf(absmax, fabsf(query_head[dim]));
    }
    const float q_scale =
        exp2f(ceilf(log2f(fmaxf(absmax, 1.0e-4f) / 448.0f)));
    q_scales[head] = q_scale;
    weights[head] *= q_scale * softmax_scale * head_scale;

    for (uint32_t dim = 0; dim < index_head_dim; ++dim) {
      const float scaled =
          fminf(fmaxf(query_head[dim] / q_scale, -448.0f), 448.0f);
      q_fp8[static_cast<uint64_t>(head) * index_head_dim + dim] =
          deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
    }
  }

  if (deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterIndexerStateWrites),
        1ull);
  }
}

__device__ float deepseek_session_read_v32_indexer_kv_raw(
    const uint8_t *kv_cache, uint64_t kv_offset_bytes, uint32_t block_count,
    const SequenceLayerLayout &layout, uint32_t position, uint32_t dim,
    float *scale_out) {
  if (scale_out != nullptr) {
    *scale_out = 0.0f;
  }
  const uint32_t head_dim = layout.deepseek_index_head_dim;
  if (kv_cache == nullptr || head_dim == 0 || dim >= head_dim) {
    return 0.0f;
  }
  const uint32_t block = position / kDeepSeekV32IndexerKvBlockTokens;
  if (block >= block_count) {
    return 0.0f;
  }
  const uint32_t scale_bytes =
      ((head_dim + 127u) / 128u) * sizeof(float);
  const uint64_t page_bytes =
      static_cast<uint64_t>(kDeepSeekV32IndexerKvBlockTokens) *
      (static_cast<uint64_t>(head_dim) + scale_bytes);
  const uint32_t block_offset =
      position % kDeepSeekV32IndexerKvBlockTokens;
  const uint32_t tile_block_id =
      block_offset / kDeepSeekV32IndexerKvTileTokens;
  const uint32_t tile_block_offset =
      block_offset % kDeepSeekV32IndexerKvTileTokens;
  const uint32_t tile_store_offset =
      (dim / kDeepSeekV32IndexerKvTileHeadBytes) *
          kDeepSeekV32IndexerKvTileTokens *
          kDeepSeekV32IndexerKvTileHeadBytes +
      (dim % kDeepSeekV32IndexerKvTileHeadBytes);
  const uint64_t value_offset =
      static_cast<uint64_t>(tile_block_id) *
          kDeepSeekV32IndexerKvTileTokens * head_dim +
      static_cast<uint64_t>(tile_block_offset) *
          kDeepSeekV32IndexerKvTileHeadBytes +
      tile_store_offset;
  const uint8_t *block_ptr = kv_cache + kv_offset_bytes +
                             static_cast<uint64_t>(block) * page_bytes;
  const uint8_t *scale_ptr =
      block_ptr +
      static_cast<uint64_t>(kDeepSeekV32IndexerKvBlockTokens) * head_dim +
      static_cast<uint64_t>(block_offset) * scale_bytes;
  const float scale =
      reinterpret_cast<const float *>(scale_ptr)[dim / 128u];
  if (scale_out != nullptr) {
    *scale_out = scale;
  }
  return nerva::deepseek::f8_e4m3fn_bits_to_f32(block_ptr[value_offset]);
}

__device__ uint32_t deepseek_session_select_v32_sparse_slots(
    SequenceLayerLayout layout, uint32_t *step_cursor, uint32_t max_steps,
    const uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    const uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    int32_t *topk_slots, float *topk_scores,
    uint32_t *candidates_scored,
    unsigned long long *selection_hash_out) {
  if (candidates_scored != nullptr) {
    *candidates_scored = 0;
  }
  if (selection_hash_out != nullptr) {
    *selection_hash_out = 0ull;
  }
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return 0;
  }
  if (topk_slots == nullptr || topk_scores == nullptr ||
      deepseek_indexer_state == nullptr || deepseek_indexer_kv == nullptr ||
      !deepseek_v32_indexer_query_state_supported(layout) ||
      layout.deepseek_index_topk == 0 ||
      deepseek_indexer_kv_block_count == 0) {
    return 0;
  }

  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t index_heads = layout.deepseek_index_n_heads;
  const uint32_t index_head_dim = layout.deepseek_index_head_dim;
  const uint32_t capacity =
      deepseek_indexer_kv_block_count * kDeepSeekV32IndexerKvBlockTokens;
  const uint32_t candidate_tokens =
      min(position + 1u, capacity);
  const uint32_t topk_limit =
      min(min(layout.deepseek_index_topk, candidate_tokens),
          kDeepSeekSessionMaxSparseTopK);
  if (candidate_tokens == 0 || topk_limit == 0) {
    return 0;
  }

  const uint64_t token_bytes =
      deepseek_v32_indexer_query_state_token_bytes(layout);
  const uint8_t *token_ptr = deepseek_indexer_state +
                             deepseek_indexer_state_offset_bytes +
                             static_cast<uint64_t>(position) * token_bytes;
  const uint8_t *q_fp8 = token_ptr;
  const auto *weights = reinterpret_cast<const float *>(
      token_ptr +
      deepseek_v32_indexer_query_state_weights_offset_bytes(layout));

  for (uint32_t rank = 0; rank < topk_limit; ++rank) {
    topk_slots[rank] = -1;
    topk_scores[rank] = -INFINITY;
  }

  if (topk_limit >= candidate_tokens) {
    for (uint32_t slot = 0; slot < candidate_tokens; ++slot) {
      topk_slots[slot] = static_cast<int32_t>(slot);
      topk_scores[slot] = INFINITY;
    }
  } else {
    if (candidates_scored != nullptr) {
      *candidates_scored = candidate_tokens;
    }
    for (uint32_t slot = 0; slot < candidate_tokens; ++slot) {
      float slot_scale = 0.0f;
      float score = 0.0f;
      for (uint32_t head = 0; head < index_heads; ++head) {
        float dot = 0.0f;
        for (uint32_t dim = 0; dim < index_head_dim; ++dim) {
          float k_scale = 0.0f;
          const float k_value = deepseek_session_read_v32_indexer_kv_raw(
              deepseek_indexer_kv, deepseek_indexer_kv_offset_bytes,
              deepseek_indexer_kv_block_count, layout, slot, dim, &k_scale);
          if (head == 0 && dim == 0) {
            slot_scale = k_scale;
          }
          const uint8_t q_bits =
              q_fp8[static_cast<uint64_t>(head) * index_head_dim + dim];
          dot += nerva::deepseek::f8_e4m3fn_bits_to_f32(q_bits) * k_value;
        }
        score += fmaxf(dot, 0.0f) * weights[head];
      }
      score *= slot_scale;
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
    if (topk_slots[rank] < 0) {
      continue;
    }
    ++selected;
    const unsigned long long term =
        (static_cast<unsigned long long>(position) + 1ull) * 1315423911ull ^
        (static_cast<unsigned long long>(rank) + 1ull) * 2654435761ull ^
        (static_cast<unsigned long long>(topk_slots[rank]) + 1ull);
    selection_hash += term;
  }
  if (selection_hash_out != nullptr) {
    *selection_hash_out = selection_hash;
  }
  return selected;
}

__global__ void hf_deepseek_v32_sparse_topk_select_kernel(
    SequenceLayerLayout layout, uint32_t *step_cursor, uint32_t max_steps,
    const uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    const uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint64_t *deepseek_runtime_counters) {
  if (blockIdx.x != 0 || threadIdx.x != 0 ||
      (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  if (deepseek_runtime_counters == nullptr) {
    return;
  }
  int32_t topk_slots[kDeepSeekSessionMaxSparseTopK];
  float topk_scores[kDeepSeekSessionMaxSparseTopK];
  uint32_t candidates_scored = 0;
  unsigned long long selection_hash = 0ull;
  const uint32_t selected = deepseek_session_select_v32_sparse_slots(
      layout, step_cursor, max_steps, deepseek_indexer_state,
      deepseek_indexer_state_offset_bytes, deepseek_indexer_kv,
      deepseek_indexer_kv_offset_bytes, deepseek_indexer_kv_block_count,
      topk_slots, topk_scores, &candidates_scored, &selection_hash);
  if (selected == 0) {
    return;
  }
  atomicAdd(
      reinterpret_cast<unsigned long long *>(
          deepseek_runtime_counters +
          kDeepSeekRuntimeCounterSparseTopkSelections),
      1ull);
  atomicAdd(
      reinterpret_cast<unsigned long long *>(
          deepseek_runtime_counters +
          kDeepSeekRuntimeCounterSparseTopkSlotsSelected),
      static_cast<unsigned long long>(selected));
  atomicAdd(
      reinterpret_cast<unsigned long long *>(
          deepseek_runtime_counters +
          kDeepSeekRuntimeCounterSparseTopkCandidatesScored),
      static_cast<unsigned long long>(candidates_scored));
  atomicAdd(
      reinterpret_cast<unsigned long long *>(
          deepseek_runtime_counters +
          kDeepSeekRuntimeCounterSparseTopkSelectionHash),
      selection_hash);
}
