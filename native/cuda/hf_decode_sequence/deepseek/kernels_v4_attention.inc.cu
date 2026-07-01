__global__ void hf_deepseek_v4_swa_attention_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rope_theta, float *scratch, uint16_t *kv_keys,
    uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint8_t *deepseek_swa_kv,
    uint64_t deepseek_swa_kv_offset_bytes,
    uint32_t deepseek_swa_kv_block_count,
    uint64_t *deepseek_runtime_counters, uint32_t local_window_tokens) {
  if (blockIdx.x >= heads ||
      (step_cursor != nullptr && *step_cursor >= max_steps) ||
      arena == nullptr || scratch == nullptr || head_dim == 0 ||
      layout.deepseek_mode != kDeepSeekModeV4Swa ||
      layout.deepseek_qk_nope_head_dim + layout.deepseek_qk_rope_head_dim !=
          head_dim) {
    return;
  }
  const uint32_t head = blockIdx.x;
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t rope_half = qk_rope / 2u;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, heads * head_dim, head_dim,
                         intermediate);

  const uint32_t window_raw_start =
      local_window_tokens == 0 || position + 1u <= local_window_tokens
          ? 0u
          : position + 1u - local_window_tokens;
  const uint32_t raw_attention_start = window_raw_start;
  const uint32_t raw_attention_tokens =
      position + 1u > raw_attention_start ? position + 1u - raw_attention_start
                                          : 0u;
  if (threadIdx.x == 0 && head == 0 && raw_attention_tokens != 0 &&
      deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterRawAttentionTokensScanned),
        static_cast<unsigned long long>(raw_attention_tokens));
  }

  uint32_t packed_block = 0;
  const uint32_t logical_block =
      position / kDeepSeekV4PackedKvDefaultBlockTokens;
  const bool use_packed_swa_kv =
      deepseek_swa_kv != nullptr &&
      deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, deepseek_swa_kv_block_count,
          logical_block, kDeepSeekV4PackedKvDefaultBlockTokens,
          &packed_block);
  (void)packed_block;

  const uint32_t head_start = head * head_dim;
  for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    s.attn[head_start + dim] = 0.0f;
  }
  __syncthreads();

  const float attn_scale = rsqrtf(static_cast<float>(head_dim));
  float local_m = layout.deepseek_attention_sink == kMissingOffset
                      ? -INFINITY
                      : f32_from_u16_slots(arena + layout.deepseek_attention_sink,
                                           head);
  float local_l = isfinite(local_m) ? 1.0f : 0.0f;
  for (uint32_t token = raw_attention_start; token <= position; ++token) {
    const uint64_t token_base =
        kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                            token, head_dim, 0);
    float score_part = 0.0f;
    for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      const float key_value =
          use_packed_swa_kv
              ? deepseek_session_read_fp8_ds_mla_swa_kv(
                    deepseek_swa_kv, deepseek_swa_kv_offset_bytes,
                    deepseek_swa_kv_block_count, kv_block_count,
                    kv_block_table, layout, token, dim)
              : encoded_to_f32(kv_keys[token_base + dim], dtype);
      score_part += s.q[head_start + dim] * key_value;
    }
    const float score = block_sum(score_part) * attn_scale;
    const float next_m = fmaxf(local_m, score);
    const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
    const float new_scale = expf(score - next_m);
    for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      const float value =
          use_packed_swa_kv
              ? deepseek_session_read_fp8_ds_mla_swa_kv(
                    deepseek_swa_kv, deepseek_swa_kv_offset_bytes,
                    deepseek_swa_kv_block_count, kv_block_count,
                    kv_block_table, layout, token, dim)
              : encoded_to_f32(kv_values[token_base + dim], dtype);
      const uint32_t out = head_start + dim;
      s.attn[out] = s.attn[out] * old_scale + value * new_scale;
    }
    local_l = local_l * old_scale + new_scale;
    local_m = next_m;
  }

  if (local_l > 0.0f && isfinite(local_l)) {
    for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      s.attn[head_start + dim] /= local_l;
    }
  }
  __syncthreads();

  if (rope_half != 0) {
    for (uint32_t offset = threadIdx.x; offset < rope_half;
         offset += blockDim.x) {
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

__global__ void hf_deepseek_v4_compressed_indexer_sparse_topk_select_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rope_theta, float *scratch, const uint16_t *projection_input,
    uint8_t *deepseek_compressed_kv,
    uint64_t deepseek_compressed_kv_offset_bytes,
    uint32_t deepseek_compressed_kv_block_count,
    uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    int32_t *sparse_topk_slots, uint32_t *sparse_topk_count,
    uint64_t *deepseek_runtime_counters) {
  if (blockIdx.x != 0 || threadIdx.x != 0 ||
      (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  if (sparse_topk_count != nullptr) {
    *sparse_topk_count = 0;
  }
  if (arena == nullptr || scratch == nullptr || projection_input == nullptr ||
      sparse_topk_slots == nullptr || sparse_topk_count == nullptr ||
      hidden == 0 || heads == 0 || head_dim == 0 ||
      layout.deepseek_mode != kDeepSeekModeV4CompressedIndexer ||
      layout.deepseek_q_lora_rank == 0 ||
      layout.deepseek_q_lora_rank > heads * head_dim) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t compressed_attention_tokens =
      deepseek_session_compressed_attention_count(
          deepseek_compressed_kv, layout, deepseek_compressed_kv_block_count,
          position);
  int32_t topk_slots[kDeepSeekSessionMaxSparseTopK];
  float topk_scores[kDeepSeekSessionMaxSparseTopK];
  float indexer_query[kDeepSeekSessionMaxIndexerQueryValues];
  uint32_t candidates_scored = 0;
  unsigned long long selection_hash = 0ull;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, heads * head_dim, head_dim,
                         intermediate);
  const uint32_t selected = deepseek_session_select_v4_c4_sparse_slots(
      arena, layout, s.q_gate, projection_input, dtype, hidden,
      layout.deepseek_q_lora_rank, position, rope_theta, deepseek_indexer_kv,
      deepseek_indexer_kv_offset_bytes, deepseek_indexer_kv_block_count,
      kv_block_count, kv_block_table, compressed_attention_tokens, topk_slots,
      topk_scores, indexer_query, &candidates_scored, &selection_hash);
  *sparse_topk_count = selected;
  for (uint32_t rank = 0; rank < selected; ++rank) {
    sparse_topk_slots[rank] = topk_slots[rank];
  }
  if (selected != 0 && deepseek_runtime_counters != nullptr) {
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
}

__global__ void hf_deepseek_v4_compressed_attention_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rope_theta, float *scratch, uint16_t *kv_keys,
    uint16_t *kv_values, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint8_t *deepseek_swa_kv,
    uint64_t deepseek_swa_kv_offset_bytes,
    uint32_t deepseek_swa_kv_block_count,
    uint8_t *deepseek_compressed_kv,
    uint64_t deepseek_compressed_kv_offset_bytes,
    uint32_t deepseek_compressed_kv_block_count,
    const int32_t *sparse_topk_slots, const uint32_t *sparse_topk_count,
    uint64_t *deepseek_runtime_counters, uint32_t local_window_tokens) {
  if (blockIdx.x >= heads ||
      (step_cursor != nullptr && *step_cursor >= max_steps) ||
      arena == nullptr || scratch == nullptr || head_dim == 0 ||
      (layout.deepseek_mode != kDeepSeekModeV4Compressed &&
       layout.deepseek_mode != kDeepSeekModeV4CompressedIndexer) ||
      layout.deepseek_qk_nope_head_dim + layout.deepseek_qk_rope_head_dim !=
          head_dim) {
    return;
  }
  const uint32_t head = blockIdx.x;
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t rope_half = qk_rope / 2u;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, heads * head_dim, head_dim,
                         intermediate);

  const uint32_t compressed_attention_tokens =
      deepseek_session_compressed_attention_count(
          deepseek_compressed_kv, layout, deepseek_compressed_kv_block_count,
          position);
  uint32_t sparse_attention_tokens = 0;
  if (layout.deepseek_mode == kDeepSeekModeV4CompressedIndexer &&
      sparse_topk_slots != nullptr && sparse_topk_count != nullptr) {
    sparse_attention_tokens = min(*sparse_topk_count,
                                  compressed_attention_tokens);
  }
  const uint32_t compressed_attention_loop_tokens =
      sparse_attention_tokens == 0 ? compressed_attention_tokens
                                   : sparse_attention_tokens;
  const uint32_t window_raw_start =
      local_window_tokens == 0 || position + 1u <= local_window_tokens
          ? 0u
          : position + 1u - local_window_tokens;
  const uint32_t raw_attention_start = window_raw_start;
  const uint32_t raw_attention_tokens =
      position + 1u > raw_attention_start ? position + 1u - raw_attention_start
                                          : 0u;
  if (threadIdx.x == 0 && head == 0 && deepseek_runtime_counters != nullptr) {
    if (raw_attention_tokens != 0) {
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterRawAttentionTokensScanned),
          static_cast<unsigned long long>(raw_attention_tokens));
    }
    if (compressed_attention_tokens != 0) {
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
  }

  uint32_t packed_block = 0;
  const uint32_t logical_block =
      position / kDeepSeekV4PackedKvDefaultBlockTokens;
  const bool use_packed_swa_kv =
      deepseek_swa_kv != nullptr &&
      deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, deepseek_swa_kv_block_count,
          logical_block, kDeepSeekV4PackedKvDefaultBlockTokens,
          &packed_block);
  (void)packed_block;

  const uint32_t head_start = head * head_dim;
  for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    s.attn[head_start + dim] = 0.0f;
  }
  __syncthreads();

  const float attn_scale = rsqrtf(static_cast<float>(head_dim));
  float local_m = layout.deepseek_attention_sink == kMissingOffset
                      ? -INFINITY
                      : f32_from_u16_slots(arena + layout.deepseek_attention_sink,
                                           head);
  float local_l = isfinite(local_m) ? 1.0f : 0.0f;
  for (uint32_t compressed_index = 0;
       compressed_index < compressed_attention_loop_tokens; ++compressed_index) {
    const int32_t selected_slot =
        sparse_attention_tokens == 0
            ? static_cast<int32_t>(compressed_index)
            : sparse_topk_slots[compressed_index];
    if (selected_slot < 0) {
      continue;
    }
    const uint32_t compressed_slot = static_cast<uint32_t>(selected_slot);
    float score_part = 0.0f;
    for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      score_part +=
          s.q[head_start + dim] *
          deepseek_session_read_fp8_ds_mla_compressed_kv(
              deepseek_compressed_kv, deepseek_compressed_kv_offset_bytes,
              deepseek_compressed_kv_block_count, layout, kv_block_count,
              kv_block_table, compressed_slot, dim);
    }
    const float score = block_sum(score_part) * attn_scale;
    const float next_m = fmaxf(local_m, score);
    const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
    const float new_scale = expf(score - next_m);
    for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      const uint32_t out = head_start + dim;
      s.attn[out] =
          s.attn[out] * old_scale +
          deepseek_session_read_fp8_ds_mla_compressed_kv(
              deepseek_compressed_kv, deepseek_compressed_kv_offset_bytes,
              deepseek_compressed_kv_block_count, layout, kv_block_count,
              kv_block_table, compressed_slot, dim) *
              new_scale;
    }
    local_l = local_l * old_scale + new_scale;
    local_m = next_m;
  }

  for (uint32_t token = raw_attention_start; token <= position; ++token) {
    const uint64_t token_base =
        kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                            token, head_dim, 0);
    float score_part = 0.0f;
    for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      const float key_value =
          use_packed_swa_kv
              ? deepseek_session_read_fp8_ds_mla_swa_kv(
                    deepseek_swa_kv, deepseek_swa_kv_offset_bytes,
                    deepseek_swa_kv_block_count, kv_block_count,
                    kv_block_table, layout, token, dim)
              : encoded_to_f32(kv_keys[token_base + dim], dtype);
      score_part += s.q[head_start + dim] * key_value;
    }
    const float score = block_sum(score_part) * attn_scale;
    const float next_m = fmaxf(local_m, score);
    const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
    const float new_scale = expf(score - next_m);
    for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      const float value =
          use_packed_swa_kv
              ? deepseek_session_read_fp8_ds_mla_swa_kv(
                    deepseek_swa_kv, deepseek_swa_kv_offset_bytes,
                    deepseek_swa_kv_block_count, kv_block_count,
                    kv_block_table, layout, token, dim)
              : encoded_to_f32(kv_values[token_base + dim], dtype);
      const uint32_t out = head_start + dim;
      s.attn[out] = s.attn[out] * old_scale + value * new_scale;
    }
    local_l = local_l * old_scale + new_scale;
    local_m = next_m;
  }

  if (local_l > 0.0f && isfinite(local_l)) {
    for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      s.attn[head_start + dim] /= local_l;
    }
  }
  __syncthreads();

  if (rope_half != 0) {
    for (uint32_t offset = threadIdx.x; offset < rope_half;
         offset += blockDim.x) {
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
