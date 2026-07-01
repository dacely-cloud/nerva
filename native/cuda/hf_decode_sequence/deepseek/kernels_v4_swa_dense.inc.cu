__global__ void hf_deepseek_v4_q_a_norm_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t hidden,
    uint32_t heads, uint32_t head_dim, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps,
    float *scratch) {
  if ((step_cursor != nullptr && *step_cursor >= max_steps) ||
      arena == nullptr || scratch == nullptr || hidden == 0 ||
      heads == 0 || head_dim == 0 || layout.q_norm == kMissingOffset) {
    return;
  }
  const uint32_t q_lora_rank = layout.deepseek_q_lora_rank;
  const uint32_t attention_hidden = heads * head_dim;
  if (q_lora_rank == 0 || q_lora_rank > attention_hidden) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, head_dim,
                         intermediate);
  float norm_sum = 0.0f;
  for (uint32_t index = threadIdx.x; index < q_lora_rank;
       index += blockDim.x) {
    const float value = s.q[index];
    norm_sum += value * value;
  }
  norm_sum = block_sum(norm_sum);
  const float norm_scale =
      rsqrtf(norm_sum / static_cast<float>(q_lora_rank) + rms_eps);
  for (uint32_t index = threadIdx.x; index < q_lora_rank;
       index += blockDim.x) {
    s.q_gate[index] =
        s.q[index] * norm_scale *
        encoded_to_f32(arena[layout.q_norm + index], kDTypeBF16);
  }
}

__global__ void hf_deepseek_v4_finalize_preprojected_qk_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rms_eps, float rope_theta, float *scratch) {
  if ((step_cursor != nullptr && *step_cursor >= max_steps) ||
      arena == nullptr || scratch == nullptr || hidden == 0 ||
      heads == 0 || head_dim == 0 || layout.k_norm == kMissingOffset) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  if (qk_nope + qk_rope != head_dim) {
    return;
  }

  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, heads * head_dim, head_dim,
                         intermediate);
  const uint32_t rope_half = qk_rope / 2u;
  if (blockIdx.x == 0) {
    float norm_sum = 0.0f;
    for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      const float value = s.k[dim];
      norm_sum += value * value;
    }
    norm_sum = block_sum(norm_sum);
    const float norm_scale =
        rsqrtf(norm_sum / static_cast<float>(head_dim) + rms_eps);
    for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      s.k[dim] *= norm_scale *
                  encoded_to_f32(arena[layout.k_norm + dim], kDTypeBF16);
    }
    __syncthreads();
    if (rope_half != 0) {
      for (uint32_t offset = threadIdx.x; offset < rope_half;
           offset += blockDim.x) {
        const uint32_t left = qk_nope + offset;
        const uint32_t right = left + rope_half;
        const float left_value = s.k[left];
        const float right_value = s.k[right];
        s.k[left] = deepseek_rope_value_serial(
            left_value, right_value, offset, qk_rope, position, rope_theta,
            false, layout);
        s.k[right] = deepseek_rope_value_serial(
            left_value, right_value, offset, qk_rope, position, rope_theta,
            true, layout);
      }
    }
    return;
  }

  const uint32_t head = blockIdx.x - 1u;
  if (head >= heads) {
    return;
  }
  const uint32_t head_start = head * head_dim;
  float norm_sum = 0.0f;
  for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const float value = s.q[head_start + dim];
    norm_sum += value * value;
  }
  norm_sum = block_sum(norm_sum);
  const float norm_scale =
      rsqrtf(norm_sum / static_cast<float>(head_dim) + rms_eps);
  for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    s.q[head_start + dim] *= norm_scale;
  }
  __syncthreads();
  if (rope_half != 0) {
    for (uint32_t offset = threadIdx.x; offset < rope_half;
         offset += blockDim.x) {
      const uint32_t left = head_start + qk_nope + offset;
      const uint32_t right = left + rope_half;
      const float left_value = s.q[left];
      const float right_value = s.q[right];
      s.q[left] = deepseek_rope_value_serial(
          left_value, right_value, offset, qk_rope, position, rope_theta,
          false, layout);
      s.q[right] = deepseek_rope_value_serial(
          left_value, right_value, offset, qk_rope, position, rope_theta,
          true, layout);
    }
  }
}

__global__ void hf_deepseek_v4_compressor_state_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t head_dim, uint32_t *step_cursor,
    uint32_t max_steps, const uint16_t *projection_input,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    float *deepseek_compressor_state,
    uint64_t deepseek_compressor_state_offset_bytes,
    uint64_t *deepseek_runtime_counters) {
  if ((step_cursor != nullptr && *step_cursor >= max_steps) ||
      arena == nullptr || projection_input == nullptr ||
      deepseek_compressor_state == nullptr || hidden == 0 ||
      head_dim == 0 || dtype > kDTypeBF16 ||
      layout.deepseek_compress_ratio <= 1 ||
      layout.deepseek_compressor_wkv == kMissingOffset ||
      layout.deepseek_compressor_wgate == kMissingOffset ||
      layout.deepseek_compressor_ape == kMissingOffset ||
      (layout.deepseek_mode != kDeepSeekModeV4Compressed &&
       layout.deepseek_mode != kDeepSeekModeV4CompressedIndexer)) {
    return;
  }
  const uint32_t coff = layout.deepseek_compress_ratio == 4 ? 2u : 1u;
  const uint32_t state_width = coff * head_dim;
  const uint32_t row = blockIdx.x;
  if (row >= state_width || state_width == 0) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t logical_block = position / kKvCacheBlockTokens;
  if (logical_block >= kv_block_count) {
    return;
  }
  const uint32_t physical_block =
      kv_block_table == nullptr ? logical_block : kv_block_table[logical_block];
  const uint32_t pos_in_block = position % kKvCacheBlockTokens;
  const uint64_t token_index =
      static_cast<uint64_t>(physical_block) * kKvCacheBlockTokens +
      pos_in_block;
  const uint64_t state_base =
      deepseek_compressor_state_offset_bytes / sizeof(float) +
      token_index * static_cast<uint64_t>(state_width) * 2u;
  float kv_sum = 0.0f;
  float score_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float input_value = encoded_to_f32(projection_input[col], dtype);
    kv_sum += encoded_to_f32(
                  arena[layout.deepseek_compressor_wkv +
                        static_cast<uint64_t>(row) * hidden + col],
                  kDTypeBF16) *
              input_value;
    score_sum += encoded_to_f32(
                     arena[layout.deepseek_compressor_wgate +
                           static_cast<uint64_t>(row) * hidden + col],
                     kDTypeBF16) *
                 input_value;
  }
  kv_sum = block_sum(kv_sum);
  score_sum = block_sum(score_sum);
  if (threadIdx.x == 0) {
    const uint32_t ape_row = position % layout.deepseek_compress_ratio;
    const float ape =
        f32_from_u16_slots(arena + layout.deepseek_compressor_ape,
                           ape_row * state_width + row);
    deepseek_compressor_state[state_base + row] = kv_sum;
    deepseek_compressor_state[state_base + state_width + row] =
        score_sum + ape;
    if (row == 0 && deepseek_runtime_counters != nullptr) {
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterCompressorStateWrites),
          1ull);
    }
  }
}

__global__ void hf_deepseek_v4_indexer_state_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t *step_cursor, uint32_t max_steps,
    const uint16_t *projection_input, uint32_t kv_block_count,
    const uint32_t *kv_block_table, float *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    uint64_t *deepseek_runtime_counters) {
  if ((step_cursor != nullptr && *step_cursor >= max_steps) ||
      arena == nullptr || projection_input == nullptr ||
      deepseek_indexer_state == nullptr || hidden == 0 ||
      dtype > kDTypeBF16 ||
      layout.deepseek_mode != kDeepSeekModeV4CompressedIndexer ||
      layout.deepseek_compress_ratio <= 1 ||
      layout.deepseek_index_head_dim == 0 ||
      layout.deepseek_indexer_compressor_wkv == kMissingOffset ||
      layout.deepseek_indexer_compressor_wgate == kMissingOffset ||
      layout.deepseek_indexer_compressor_ape == kMissingOffset) {
    return;
  }
  const uint32_t coff = layout.deepseek_compress_ratio == 4 ? 2u : 1u;
  const uint32_t state_width = coff * layout.deepseek_index_head_dim;
  const uint32_t row = blockIdx.x;
  if (row >= state_width || state_width == 0) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t logical_block = position / kKvCacheBlockTokens;
  if (logical_block >= kv_block_count) {
    return;
  }
  const uint32_t physical_block =
      kv_block_table == nullptr ? logical_block : kv_block_table[logical_block];
  const uint32_t pos_in_block = position % kKvCacheBlockTokens;
  const uint64_t token_index =
      static_cast<uint64_t>(physical_block) * kKvCacheBlockTokens +
      pos_in_block;
  const uint64_t state_base =
      deepseek_indexer_state_offset_bytes / sizeof(float) +
      token_index * static_cast<uint64_t>(state_width) * 2u;
  float kv_sum = 0.0f;
  float score_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float input_value = encoded_to_f32(projection_input[col], dtype);
    kv_sum += encoded_to_f32(
                  arena[layout.deepseek_indexer_compressor_wkv +
                        static_cast<uint64_t>(row) * hidden + col],
                  kDTypeBF16) *
              input_value;
    score_sum += encoded_to_f32(
                     arena[layout.deepseek_indexer_compressor_wgate +
                           static_cast<uint64_t>(row) * hidden + col],
                     kDTypeBF16) *
                 input_value;
  }
  kv_sum = block_sum(kv_sum);
  score_sum = block_sum(score_sum);
  if (threadIdx.x == 0) {
    const uint32_t ape_row = position % layout.deepseek_compress_ratio;
    const float ape =
        f32_from_u16_slots(arena + layout.deepseek_indexer_compressor_ape,
                           ape_row * state_width + row);
    deepseek_indexer_state[state_base + row] = kv_sum;
    deepseek_indexer_state[state_base + state_width + row] =
        score_sum + ape;
    if (row == 0 && deepseek_runtime_counters != nullptr) {
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterIndexerStateWrites),
          1ull);
    }
  }
}

__global__ void hf_deepseek_v4_swa_dense_layer_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rms_eps, float rope_theta, float *scratch, uint16_t *kv_keys,
    uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint32_t vocab_size,
    const uint32_t *prompt_tokens, uint32_t prompt_token_count,
    const NervaCudaSyntheticTokenSlot *slots, uint16_t *projection_input,
    uint8_t *deepseek_swa_kv, uint64_t deepseek_swa_kv_offset_bytes,
    uint32_t deepseek_swa_kv_block_count,
    float *deepseek_compressor_state,
    uint64_t deepseek_compressor_state_offset_bytes,
    uint8_t *deepseek_compressed_kv,
    uint64_t deepseek_compressed_kv_offset_bytes,
    uint32_t deepseek_compressed_kv_block_count,
    float *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    float *deepseek_mhc_residual, float *deepseek_mhc_post_mix,
    float *deepseek_mhc_comb_mix,
    uint64_t *deepseek_runtime_counters, uint32_t local_window_tokens,
    uint32_t preprojected_qk, uint32_t precomputed_compressor_state,
    uint32_t precomputed_indexer_state) {
  if (threadIdx.x != 0 ||
      (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t current_token =
      position < prompt_token_count ? prompt_tokens[position]
                                    : slots[position - 1u].token;
  const uint32_t q_lora_rank = layout.deepseek_q_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t o_lora_rank = layout.deepseek_o_lora_rank;
  const uint32_t o_groups = layout.deepseek_o_groups;
  const uint32_t attention_hidden = heads * head_dim;
  if (q_lora_rank == 0 || q_lora_rank > attention_hidden ||
      qk_rope == 0 || head_dim == 0 || qk_nope + qk_rope != head_dim ||
      o_lora_rank == 0 || o_groups == 0 || heads % o_groups != 0 ||
      layout.w_q == kMissingOffset || layout.deepseek_q_a_scale == kMissingOffset ||
      layout.q_norm == kMissingOffset || layout.deepseek_q_b == kMissingOffset ||
      layout.deepseek_q_b_scale == kMissingOffset ||
      layout.w_k == kMissingOffset || layout.deepseek_kv_a_scale == kMissingOffset ||
      layout.k_norm == kMissingOffset || layout.w_o == kMissingOffset ||
      layout.deepseek_o_a_scale == kMissingOffset ||
      layout.deepseek_o_b == kMissingOffset ||
      layout.deepseek_o_b_scale == kMissingOffset) {
    return;
  }

  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, heads * head_dim, head_dim, intermediate);
  if (preprojected_qk == 0) {
    deepseek_session_apply_v4_mhc_pre_state(
        arena, layout, dtype, hidden, position, rms_eps, s.input,
        layer_index == 0 ? 1u : 0u, layout.deepseek_hc_attn_base,
        layout.deepseek_hc_attn_fn, layout.deepseek_hc_attn_scale,
        layout.rms_attn, deepseek_mhc_residual, deepseek_mhc_post_mix,
        deepseek_mhc_comb_mix, s.mlp_norm, projection_input);
    for (uint32_t row = 0; row < q_lora_rank; ++row) {
      float sum = 0.0f;
      for (uint32_t col = 0; col < hidden; ++col) {
        sum += deepseek_fp8_e8m0_scaled_weight(
                   arena, layout.w_q, layout.deepseek_q_a_scale, q_lora_rank,
                   hidden, row, col) *
               encoded_to_f32(projection_input[col], dtype);
      }
      s.q[row] = sum;
    }
    for (uint32_t row = 0; row < head_dim; ++row) {
      float sum = 0.0f;
      for (uint32_t col = 0; col < hidden; ++col) {
        sum += deepseek_fp8_e8m0_scaled_weight(
                   arena, layout.w_k, layout.deepseek_kv_a_scale, head_dim,
                   hidden, row, col) *
               encoded_to_f32(projection_input[col], dtype);
      }
      s.k[row] = sum;
    }
  }

  if (precomputed_compressor_state == 0 &&
      (layout.deepseek_mode == kDeepSeekModeV4Compressed ||
       layout.deepseek_mode == kDeepSeekModeV4CompressedIndexer) &&
      layout.deepseek_compress_ratio > 1 &&
      deepseek_compressor_state != nullptr &&
      layout.deepseek_compressor_wkv != kMissingOffset &&
      layout.deepseek_compressor_wgate != kMissingOffset &&
      layout.deepseek_compressor_ape != kMissingOffset) {
    const uint32_t coff =
        layout.deepseek_compress_ratio == 4 ? 2u : 1u;
    const uint32_t state_width = coff * head_dim;
    const uint32_t logical_block = position / kKvCacheBlockTokens;
    if (state_width != 0 && logical_block < kv_block_count) {
      const uint32_t physical_block =
          kv_block_table == nullptr ? logical_block : kv_block_table[logical_block];
      const uint32_t pos_in_block = position % kKvCacheBlockTokens;
      const uint64_t token_index =
          static_cast<uint64_t>(physical_block) * kKvCacheBlockTokens +
          pos_in_block;
      const uint64_t state_base =
          deepseek_compressor_state_offset_bytes / sizeof(float) +
          token_index * static_cast<uint64_t>(state_width) * 2u;
      const uint32_t ape_row = position % layout.deepseek_compress_ratio;
      for (uint32_t row = 0; row < state_width; ++row) {
        float kv_sum = 0.0f;
        float score_sum = 0.0f;
        for (uint32_t col = 0; col < hidden; ++col) {
          const float input_value = encoded_to_f32(projection_input[col], dtype);
          kv_sum += encoded_to_f32(
                        arena[layout.deepseek_compressor_wkv +
                              static_cast<uint64_t>(row) * hidden + col],
                        kDTypeBF16) *
                    input_value;
          score_sum += encoded_to_f32(
                           arena[layout.deepseek_compressor_wgate +
                                 static_cast<uint64_t>(row) * hidden + col],
                           kDTypeBF16) *
                       input_value;
        }
        const float ape =
            f32_from_u16_slots(arena + layout.deepseek_compressor_ape,
                               ape_row * state_width + row);
        deepseek_compressor_state[state_base + row] = kv_sum;
        deepseek_compressor_state[state_base + state_width + row] =
            score_sum + ape;
      }
      if (deepseek_runtime_counters != nullptr) {
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                deepseek_runtime_counters +
                kDeepSeekRuntimeCounterCompressorStateWrites),
            1ull);
      }
    }
  }
  if (precomputed_indexer_state == 0 &&
      layout.deepseek_mode == kDeepSeekModeV4CompressedIndexer &&
      layout.deepseek_compress_ratio > 1 &&
      layout.deepseek_index_head_dim > 0 &&
      deepseek_indexer_state != nullptr &&
      layout.deepseek_indexer_compressor_wkv != kMissingOffset &&
      layout.deepseek_indexer_compressor_wgate != kMissingOffset &&
      layout.deepseek_indexer_compressor_ape != kMissingOffset) {
    const uint32_t coff =
        layout.deepseek_compress_ratio == 4 ? 2u : 1u;
    const uint32_t state_width = coff * layout.deepseek_index_head_dim;
    const uint32_t logical_block = position / kKvCacheBlockTokens;
    if (state_width != 0 && logical_block < kv_block_count) {
      const uint32_t physical_block =
          kv_block_table == nullptr ? logical_block : kv_block_table[logical_block];
      const uint32_t pos_in_block = position % kKvCacheBlockTokens;
      const uint64_t token_index =
          static_cast<uint64_t>(physical_block) * kKvCacheBlockTokens +
          pos_in_block;
      const uint64_t state_base =
          deepseek_indexer_state_offset_bytes / sizeof(float) +
          token_index * static_cast<uint64_t>(state_width) * 2u;
      const uint32_t ape_row = position % layout.deepseek_compress_ratio;
      for (uint32_t row = 0; row < state_width; ++row) {
        float kv_sum = 0.0f;
        float score_sum = 0.0f;
        for (uint32_t col = 0; col < hidden; ++col) {
          const float input_value = encoded_to_f32(projection_input[col], dtype);
          kv_sum += encoded_to_f32(
                        arena[layout.deepseek_indexer_compressor_wkv +
                              static_cast<uint64_t>(row) * hidden + col],
                        kDTypeBF16) *
                    input_value;
          score_sum += encoded_to_f32(
                           arena[layout.deepseek_indexer_compressor_wgate +
                                 static_cast<uint64_t>(row) * hidden + col],
                           kDTypeBF16) *
                       input_value;
        }
        const float ape =
            f32_from_u16_slots(arena + layout.deepseek_indexer_compressor_ape,
                               ape_row * state_width + row);
        deepseek_indexer_state[state_base + row] = kv_sum;
        deepseek_indexer_state[state_base + state_width + row] =
            score_sum + ape;
      }
      if (deepseek_runtime_counters != nullptr) {
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                deepseek_runtime_counters +
                kDeepSeekRuntimeCounterIndexerStateWrites),
            1ull);
      }
    }
  }

  float deepseek_compressed_scratch[kDeepSeekSessionMaxCompressHeadSize];
  if (deepseek_session_write_fp8_ds_mla_compressed_kv(
          arena, deepseek_compressor_state,
          deepseek_compressor_state_offset_bytes, deepseek_compressed_kv,
          deepseek_compressed_kv_offset_bytes,
          deepseek_compressed_kv_block_count, layout, kv_block_count,
          kv_block_table, position, rms_eps, rope_theta,
          deepseek_compressed_scratch)) {
    if (deepseek_runtime_counters != nullptr) {
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterCompressedKvWrites),
          1ull);
    }
  }
  if (deepseek_session_write_indexer_fp8_compressed_kv(
          arena, deepseek_indexer_state, deepseek_indexer_state_offset_bytes,
          deepseek_indexer_kv, deepseek_indexer_kv_offset_bytes,
          deepseek_indexer_kv_block_count, layout, kv_block_count,
          kv_block_table, position, rms_eps, rope_theta,
          deepseek_compressed_scratch)) {
    if (deepseek_runtime_counters != nullptr) {
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterIndexerKvWrites),
          1ull);
    }
  }

  if (preprojected_qk < 2u) {
    float q_norm_sum = 0.0f;
    for (uint32_t index = 0; index < q_lora_rank; ++index) {
      q_norm_sum += s.q[index] * s.q[index];
    }
    const float q_norm_scale =
        rsqrtf(q_norm_sum / static_cast<float>(q_lora_rank) + rms_eps);
    for (uint32_t index = 0; index < q_lora_rank; ++index) {
      s.q_gate[index] =
          s.q[index] * q_norm_scale *
          encoded_to_f32(arena[layout.q_norm + index], kDTypeBF16);
    }
  }

  const uint32_t compressed_attention_tokens =
      deepseek_session_compressed_attention_count(
          deepseek_compressed_kv, layout, deepseek_compressed_kv_block_count,
          position);
  int32_t sparse_compressed_slots[kDeepSeekSessionMaxSparseTopK];
  float sparse_compressed_scores[kDeepSeekSessionMaxSparseTopK];
  float sparse_indexer_query[kDeepSeekSessionMaxIndexerQueryValues];
  uint32_t sparse_compressed_candidates_scored = 0;
  unsigned long long sparse_compressed_selection_hash = 0ull;
  const uint32_t sparse_compressed_attention_tokens =
      deepseek_session_select_v4_c4_sparse_slots(
          arena, layout, s.q_gate, projection_input, dtype, hidden, q_lora_rank,
          position, rope_theta, deepseek_indexer_kv,
          deepseek_indexer_kv_offset_bytes, deepseek_indexer_kv_block_count,
          compressed_attention_tokens, sparse_compressed_slots,
          sparse_compressed_scores, sparse_indexer_query,
          &sparse_compressed_candidates_scored,
          &sparse_compressed_selection_hash);
  if (sparse_compressed_attention_tokens != 0 &&
      deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterSparseTopkSelections),
        1ull);
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterSparseTopkSlotsSelected),
        static_cast<unsigned long long>(sparse_compressed_attention_tokens));
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterSparseTopkCandidatesScored),
        static_cast<unsigned long long>(
            sparse_compressed_candidates_scored));
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterSparseTopkSelectionHash),
        sparse_compressed_selection_hash);
  }

  const uint32_t rope_half = qk_rope / 2u;
  if (preprojected_qk < 2u) {
    for (uint32_t row = 0; row < attention_hidden; ++row) {
      float sum = 0.0f;
      for (uint32_t col = 0; col < q_lora_rank; ++col) {
        sum += deepseek_fp8_e8m0_scaled_weight(
                   arena, layout.deepseek_q_b, layout.deepseek_q_b_scale,
                   attention_hidden, q_lora_rank, row, col) *
               s.q_gate[col];
      }
      s.q[row] = sum;
    }

    float kv_norm_sum = 0.0f;
    for (uint32_t index = 0; index < head_dim; ++index) {
      kv_norm_sum += s.k[index] * s.k[index];
    }
    const float kv_norm_scale =
        rsqrtf(kv_norm_sum / static_cast<float>(head_dim) + rms_eps);
    for (uint32_t index = 0; index < head_dim; ++index) {
      s.k[index] *= kv_norm_scale *
                    encoded_to_f32(arena[layout.k_norm + index], kDTypeBF16);
    }

    for (uint32_t head = 0; head < heads; ++head) {
      const uint32_t head_start = head * head_dim;
      float q_head_norm = 0.0f;
      for (uint32_t dim = 0; dim < head_dim; ++dim) {
        const float value = s.q[head_start + dim];
        q_head_norm += value * value;
      }
      const float q_head_scale =
          rsqrtf(q_head_norm / static_cast<float>(head_dim) + rms_eps);
      for (uint32_t dim = 0; dim < head_dim; ++dim) {
        s.q[head_start + dim] *= q_head_scale;
      }
      if (rope_half != 0) {
        for (uint32_t offset = 0; offset < rope_half; ++offset) {
          const uint32_t left = head_start + qk_nope + offset;
          const uint32_t right = left + rope_half;
          const float left_value = s.q[left];
          const float right_value = s.q[right];
          s.q[left] = deepseek_rope_value_serial(
              left_value, right_value, offset, qk_rope, position, rope_theta,
              false, layout);
          s.q[right] = deepseek_rope_value_serial(
              left_value, right_value, offset, qk_rope, position, rope_theta,
              true, layout);
        }
      }
    }
    if (rope_half != 0) {
      for (uint32_t offset = 0; offset < rope_half; ++offset) {
        const uint32_t left = qk_nope + offset;
        const uint32_t right = left + rope_half;
        const float left_value = s.k[left];
        const float right_value = s.k[right];
        s.k[left] = deepseek_rope_value_serial(
            left_value, right_value, offset, qk_rope, position, rope_theta,
            false, layout);
        s.k[right] = deepseek_rope_value_serial(
            left_value, right_value, offset, qk_rope, position, rope_theta,
            true, layout);
      }
    }
  }

  const bool use_packed_swa_kv =
      deepseek_session_write_fp8_ds_mla_swa_kv(
          deepseek_swa_kv, deepseek_swa_kv_offset_bytes,
          deepseek_swa_kv_block_count, layout, position, s.k);
  if (!use_packed_swa_kv) {
    const uint64_t write_base =
        kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                            position, head_dim, 0);
    for (uint32_t dim = 0; dim < head_dim; ++dim) {
      const uint16_t encoded = f32_to_encoded(s.k[dim], dtype);
      kv_keys[write_base + dim] = encoded;
      kv_values[write_base + dim] = encoded;
    }
  }

  const float attn_scale = rsqrtf(static_cast<float>(head_dim));
  const uint32_t window_raw_start =
      local_window_tokens == 0 || position + 1u <= local_window_tokens
          ? 0u
          : position + 1u - local_window_tokens;
  const uint32_t raw_attention_start = window_raw_start;
  const uint32_t raw_attention_tokens =
      position + 1u > raw_attention_start ? position + 1u - raw_attention_start
                                          : 0u;
  const uint32_t compressed_attention_loop_tokens =
      sparse_compressed_attention_tokens == 0
          ? compressed_attention_tokens
          : sparse_compressed_attention_tokens;
  if (raw_attention_tokens != 0 && deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterRawAttentionTokensScanned),
        static_cast<unsigned long long>(raw_attention_tokens));
  }
  if (compressed_attention_tokens != 0 &&
      deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterCompressedKvAttentionReads),
        1ull);
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterCompressedKvAttentionSlotsScanned),
        static_cast<unsigned long long>(compressed_attention_loop_tokens));
  }
  for (uint32_t head = 0; head < heads; ++head) {
    const uint32_t head_start = head * head_dim;
    float local_m = layout.deepseek_attention_sink == kMissingOffset
                        ? -INFINITY
                        : f32_from_u16_slots(arena + layout.deepseek_attention_sink,
                                             head);
    float local_l = isfinite(local_m) ? 1.0f : 0.0f;
    for (uint32_t dim = 0; dim < head_dim; ++dim) {
      s.attn[head_start + dim] = 0.0f;
    }
    for (uint32_t compressed_index = 0;
         compressed_index < compressed_attention_loop_tokens;
         ++compressed_index) {
      const uint32_t compressed_slot =
          sparse_compressed_attention_tokens == 0
              ? compressed_index
              : static_cast<uint32_t>(
                    sparse_compressed_slots[compressed_index]);
      float score = 0.0f;
      for (uint32_t dim = 0; dim < head_dim; ++dim) {
        score += s.q[head_start + dim] *
                 deepseek_session_read_fp8_ds_mla_compressed_kv(
                     deepseek_compressed_kv,
                     deepseek_compressed_kv_offset_bytes,
                     deepseek_compressed_kv_block_count, layout,
                     compressed_slot, dim);
      }
      score *= attn_scale;
      const float next_m = fmaxf(local_m, score);
      const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
      const float new_scale = expf(score - next_m);
      for (uint32_t dim = 0; dim < head_dim; ++dim) {
        const uint32_t out = head_start + dim;
        s.attn[out] =
            s.attn[out] * old_scale +
            deepseek_session_read_fp8_ds_mla_compressed_kv(
                deepseek_compressed_kv, deepseek_compressed_kv_offset_bytes,
                deepseek_compressed_kv_block_count, layout, compressed_slot,
                dim) *
                new_scale;
      }
      local_l = local_l * old_scale + new_scale;
      local_m = next_m;
    }
    for (uint32_t token = raw_attention_start; token <= position; ++token) {
      const uint64_t token_base =
          kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                              token, head_dim, 0);
      float score = 0.0f;
      for (uint32_t dim = 0; dim < head_dim; ++dim) {
        const float key_value =
            use_packed_swa_kv
                ? deepseek_session_read_fp8_ds_mla_swa_kv(
                      deepseek_swa_kv, deepseek_swa_kv_offset_bytes,
                      deepseek_swa_kv_block_count, layout, token, dim)
                : encoded_to_f32(kv_keys[token_base + dim], dtype);
        score += s.q[head_start + dim] * key_value;
      }
      score *= attn_scale;
      const float next_m = fmaxf(local_m, score);
      const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
      const float new_scale = expf(score - next_m);
      for (uint32_t dim = 0; dim < head_dim; ++dim) {
        const uint32_t out = head_start + dim;
        const float value =
            use_packed_swa_kv
                ? deepseek_session_read_fp8_ds_mla_swa_kv(
                      deepseek_swa_kv, deepseek_swa_kv_offset_bytes,
                      deepseek_swa_kv_block_count, layout, token, dim)
                : encoded_to_f32(kv_values[token_base + dim], dtype);
        s.attn[out] =
            s.attn[out] * old_scale + value * new_scale;
      }
      local_l = local_l * old_scale + new_scale;
      local_m = next_m;
    }
    if (local_l > 0.0f && isfinite(local_l)) {
      for (uint32_t dim = 0; dim < head_dim; ++dim) {
        s.attn[head_start + dim] /= local_l;
      }
    }
    if (rope_half != 0) {
      for (uint32_t offset = 0; offset < rope_half; ++offset) {
        const uint32_t left = head_start + qk_nope + offset;
        const uint32_t right = left + rope_half;
        const float angle =
            static_cast<float>(position) *
            deepseek_rope_inv_freq(layout, offset, qk_rope, rope_theta);
        float sin_value = 0.0f;
        float cos_value = 0.0f;
        sincosf(angle, &sin_value, &cos_value);
        const float magnitude = deepseek_rope_magnitude(layout);
        const float left_value = s.attn[left];
        const float right_value = s.attn[right];
        s.attn[left] =
            magnitude * (left_value * cos_value + right_value * sin_value);
        s.attn[right] =
            magnitude * (right_value * cos_value - left_value * sin_value);
      }
    }
  }

  const uint32_t heads_per_group = heads / o_groups;
  const uint32_t wo_a_cols = heads_per_group * head_dim;
  const uint32_t wo_a_rows = o_groups * o_lora_rank;
  for (uint32_t group = 0; group < o_groups; ++group) {
    for (uint32_t row = 0; row < o_lora_rank; ++row) {
      float sum = 0.0f;
      const uint32_t global_row = group * o_lora_rank + row;
      for (uint32_t col = 0; col < wo_a_cols; ++col) {
        sum += deepseek_fp8_e8m0_scaled_weight(
                   arena, layout.w_o, layout.deepseek_o_a_scale, wo_a_rows,
                   wo_a_cols, global_row, col) *
               s.attn[group * wo_a_cols + col];
      }
      s.q_gate[global_row] = sum;
    }
  }
  for (uint32_t row = 0; row < hidden; ++row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < wo_a_rows; ++col) {
      sum += deepseek_fp8_e8m0_scaled_weight(
                 arena, layout.deepseek_o_b, layout.deepseek_o_b_scale, hidden,
                 wo_a_rows, row, col) *
             s.q_gate[col];
    }
    s.residual[row] = sum;
  }

  deepseek_session_apply_v4_mhc_pre_state(
      arena, layout, dtype, hidden, position, rms_eps, s.residual, 0u,
      layout.deepseek_hc_ffn_base, layout.deepseek_hc_ffn_fn,
      layout.deepseek_hc_ffn_scale, layout.rms_mlp, deepseek_mhc_residual,
      deepseek_mhc_post_mix, deepseek_mhc_comb_mix, s.mlp_norm,
      projection_input);
  if (layout.mlp_kind == kMlpKindSparseMoe) {
    float router_logits[kSparseMoeExpertsMax];
    float correction_bias[kSparseMoeExpertsMax];
    uint32_t selected_experts[kSparseMoeTopKMax];
    float selected_weights[kSparseMoeTopKMax];
    const uint32_t num_experts = layout.num_experts;
    const uint32_t top_k = layout.experts_per_token;
    const uint32_t moe_intermediate = layout.moe_intermediate;
    if (layout.w_router == kMissingOffset ||
        layout.w_expert_gate_up == kMissingOffset ||
        layout.w_expert_down == kMissingOffset ||
        num_experts == 0 || num_experts > kSparseMoeExpertsMax ||
        top_k == 0 || top_k > kSparseMoeTopKMax ||
        top_k > num_experts || moe_intermediate == 0 ||
        moe_intermediate > intermediate || (hidden & 1u) != 0 ||
        (moe_intermediate & 1u) != 0) {
      return;
    }
    const uint64_t router_metadata_offset =
        layout.w_router + static_cast<uint64_t>(num_experts) * hidden;
    const bool hash_router =
        (layout.deepseek_flags & kDeepSeekFlagHashRouter) != 0;
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
      float sum = 0.0f;
      const uint64_t row = layout.w_router +
                           static_cast<uint64_t>(expert) * hidden;
      for (uint32_t col = 0; col < hidden; ++col) {
        sum += encoded_to_f32(arena[row + col], kDTypeBF16) * s.mlp_norm[col];
      }
      router_logits[expert] = sum;
      correction_bias[expert] =
          hash_router ? 0.0f
                      : f32_from_u16_slots(arena + router_metadata_offset,
                                           expert);
    }
    const float routed_scale =
        isfinite(layout.deepseek_routed_scaling_factor) &&
                layout.deepseek_routed_scaling_factor != 0.0f
            ? layout.deepseek_routed_scaling_factor
            : 1.0f;
    if (hash_router) {
      if (current_token >= vocab_size) {
        return;
      }
      float weight_sum = 0.0f;
      for (uint32_t rank = 0; rank < top_k; ++rank) {
        const uint64_t table_index =
            static_cast<uint64_t>(current_token) * top_k + rank;
        const uint64_t expert64 =
            deepseek_u64_from_u16_slots(arena + router_metadata_offset,
                                        table_index);
        if (expert64 >= num_experts) {
          return;
        }
        const uint32_t expert = static_cast<uint32_t>(expert64);
        selected_experts[rank] = expert;
        selected_weights[rank] =
            nerva::deepseek::router::sqrtsoftplus_score(router_logits[expert]);
        weight_sum += selected_weights[rank];
      }
      const float scale = nerva::deepseek::router::route_scale(
          weight_sum, layout.norm_topk_prob, routed_scale);
      for (uint32_t rank = 0; rank < top_k; ++rank) {
        selected_weights[rank] *= scale;
      }
      if (deepseek_runtime_counters != nullptr) {
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                deepseek_runtime_counters +
                kDeepSeekRuntimeCounterV4HashRouterSelections),
            1ull);
      }
    } else {
      if (nerva::deepseek::router::route_v4_sqrtsoftplus(
              router_logits, correction_bias, num_experts, top_k,
              layout.norm_topk_prob, routed_scale, selected_experts,
              selected_weights) != 0) {
        return;
      }
      if (deepseek_runtime_counters != nullptr) {
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                deepseek_runtime_counters +
                kDeepSeekRuntimeCounterV4BiasRouterSelections),
            1ull);
      }
    }
    for (uint32_t row = 0; row < hidden; ++row) {
      s.down[row] = 0.0f;
    }

    const uint32_t half_hidden = hidden >> 1u;
    const uint32_t half_intermediate = moe_intermediate >> 1u;
    const uint64_t expert_gate = layout.w_expert_gate_up;
    const uint64_t expert_gate_scale =
        expert_gate +
        deepseek_device_rank3_slots(num_experts, moe_intermediate,
                                    half_hidden);
    const uint32_t gate_scale_cols = (half_hidden + 15u) / 16u;
    const uint64_t expert_up =
        expert_gate_scale +
        deepseek_device_rank3_slots(num_experts, moe_intermediate,
                                    gate_scale_cols);
    const uint64_t expert_up_scale =
        expert_up +
        deepseek_device_rank3_slots(num_experts, moe_intermediate,
                                    half_hidden);
    const uint64_t expert_down = layout.w_expert_down;
    const uint64_t expert_down_scale =
        expert_down +
        deepseek_device_rank3_slots(num_experts, hidden, half_intermediate);
    for (uint32_t rank = 0; rank < top_k; ++rank) {
      const uint32_t expert = selected_experts[rank];
      const float expert_weight = selected_weights[rank];
      for (uint32_t row = 0; row < moe_intermediate; ++row) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (uint32_t col = 0; col < hidden; ++col) {
          gate_sum += deepseek_mxfp4_rank3_scaled_weight(
                          arena, expert_gate, expert_gate_scale,
                          moe_intermediate, half_hidden, expert, row, col) *
                      s.mlp_norm[col];
          up_sum += deepseek_mxfp4_rank3_scaled_weight(
                        arena, expert_up, expert_up_scale, moe_intermediate,
                        half_hidden, expert, row, col) *
                    s.mlp_norm[col];
        }
        s.ff[row] =
            deepseek_swiglu(gate_sum, up_sum, layout.deepseek_swiglu_limit);
      }
      for (uint32_t row = 0; row < hidden; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < moe_intermediate; ++col) {
          sum += deepseek_mxfp4_rank3_scaled_weight(
                     arena, expert_down, expert_down_scale, hidden,
                     half_intermediate, expert, row, col) *
                 s.ff[col];
        }
        s.down[row] += expert_weight * sum;
      }
    }

    const uint32_t shared_intermediate = layout.shared_expert_intermediate;
    if (shared_intermediate != 0) {
      if (layout.w_shared_expert_gate == kMissingOffset ||
          layout.w_shared_expert_up == kMissingOffset ||
          layout.w_shared_expert_down == kMissingOffset ||
          shared_intermediate > intermediate) {
        return;
      }
      const uint64_t shared_gate_scale =
          layout.w_shared_expert_gate +
          deepseek_device_fp8_slots(shared_intermediate, hidden);
      const uint64_t shared_up_scale =
          layout.w_shared_expert_up +
          deepseek_device_fp8_slots(shared_intermediate, hidden);
      const uint64_t shared_down_scale =
          layout.w_shared_expert_down +
          deepseek_device_fp8_slots(hidden, shared_intermediate);
      for (uint32_t row = 0; row < shared_intermediate; ++row) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (uint32_t col = 0; col < hidden; ++col) {
          gate_sum += deepseek_fp8_e8m0_scaled_weight(
                          arena, layout.w_shared_expert_gate,
                          shared_gate_scale, shared_intermediate, hidden, row,
                          col) *
                      s.mlp_norm[col];
          up_sum += deepseek_fp8_e8m0_scaled_weight(
                        arena, layout.w_shared_expert_up, shared_up_scale,
                        shared_intermediate, hidden, row, col) *
                    s.mlp_norm[col];
        }
        s.ff[row] =
            deepseek_swiglu(gate_sum, up_sum, layout.deepseek_swiglu_limit);
      }
      for (uint32_t row = 0; row < hidden; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < shared_intermediate; ++col) {
          sum += deepseek_fp8_e8m0_scaled_weight(
                     arena, layout.w_shared_expert_down, shared_down_scale,
                     hidden, shared_intermediate, row, col) *
                 s.ff[col];
        }
        s.down[row] += sum;
      }
    }
  } else {
    const uint64_t gate_scale =
        layout.w_gate + deepseek_device_fp8_slots(intermediate, hidden);
    const uint64_t up_scale =
        layout.w_up + deepseek_device_fp8_slots(intermediate, hidden);
    const uint64_t down_scale =
        layout.w_down + deepseek_device_fp8_slots(hidden, intermediate);
    for (uint32_t row = 0; row < intermediate; ++row) {
      float gate_sum = 0.0f;
      float up_sum = 0.0f;
      for (uint32_t col = 0; col < hidden; ++col) {
        gate_sum += deepseek_fp8_scaled_weight(
                        arena, layout.w_gate, gate_scale, intermediate, hidden,
                        row, col) *
                    s.mlp_norm[col];
        up_sum += deepseek_fp8_scaled_weight(
                      arena, layout.w_up, up_scale, intermediate, hidden, row,
                      col) *
                  s.mlp_norm[col];
      }
      s.ff[row] =
          deepseek_swiglu(gate_sum, up_sum, layout.deepseek_swiglu_limit);
    }
    for (uint32_t row = 0; row < hidden; ++row) {
      float sum = 0.0f;
      for (uint32_t col = 0; col < intermediate; ++col) {
        sum += deepseek_fp8_scaled_weight(
                   arena, layout.w_down, down_scale, hidden, intermediate, row,
                   col) *
               s.ff[col];
      }
      s.down[row] = sum;
    }
  }
  for (uint32_t row = 0; row < hidden; ++row) {
    s.residual[row] = 0.0f;
  }
}
