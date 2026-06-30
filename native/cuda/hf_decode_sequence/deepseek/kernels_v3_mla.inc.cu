__global__ void hf_deepseek_v3_mla_attention_encode_kernel(
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
  if (blockIdx.x != 0 || threadIdx.x != 0 ||
      (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t q_lora_rank = layout.deepseek_q_lora_rank;
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t v_head = layout.deepseek_v_head_dim;
  const uint32_t qk_head_dim = qk_nope + qk_rope;
  const uint32_t kv_cache_width = kv_lora_rank + qk_rope;
  (void)q_lora_rank;
  if (heads == 0 || kv_lora_rank == 0 || qk_nope == 0 || qk_rope == 0 ||
      v_head == 0 || qk_head_dim == 0 ||
      layout.w_v == kMissingOffset ||
      layout.deepseek_kv_b_scale == kMissingOffset) {
    return;
  }

  const uint64_t write_base =
      kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                          position, kv_cache_width, 0);
  for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
    kv_keys[write_base + latent] = kv_latent_norm[latent];
  }
  const uint32_t rope_half = qk_rope / 2u;
  for (uint32_t dim = 0; dim < qk_rope; ++dim) {
    float value = kv_a[kv_lora_rank + dim];
    if (rope_half != 0) {
      const uint32_t offset = dim % rope_half;
      const uint32_t pair = dim < rope_half ? dim + rope_half : dim - rope_half;
      value = deepseek_rope_value_serial(
          kv_a[kv_lora_rank + offset], kv_a[kv_lora_rank + offset + rope_half],
          offset, qk_rope, position, rope_theta, dim >= rope_half);
      (void)pair;
    }
    kv_keys[write_base + kv_lora_rank + dim] = f32_to_encoded(value, dtype);
  }
  deepseek_session_write_v32_fp8_ds_mla_kv(
      deepseek_v32_mla_kv, deepseek_v32_mla_kv_offset_bytes,
      deepseek_v32_mla_kv_block_count, layout, position, dtype,
      kv_latent_norm, kv_a, rope_theta);

  int32_t sparse_slots[kDeepSeekSessionMaxSparseTopK];
  float sparse_scores[kDeepSeekSessionMaxSparseTopK];
  uint32_t sparse_candidates_scored = 0;
  unsigned long long sparse_selection_hash = 0ull;
  const uint32_t sparse_attention_tokens =
      deepseek_session_select_v32_sparse_slots(
          layout, step_cursor, max_steps, deepseek_indexer_state,
          deepseek_indexer_state_offset_bytes, deepseek_indexer_kv,
          deepseek_indexer_kv_offset_bytes, deepseek_indexer_kv_block_count,
          sparse_slots, sparse_scores, &sparse_candidates_scored,
          &sparse_selection_hash);
  const bool use_sparse_attention = sparse_attention_tokens != 0;
  const uint32_t attention_tokens =
      use_sparse_attention ? sparse_attention_tokens : position + 1u;
  if (deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterRawAttentionTokensScanned),
        static_cast<unsigned long long>(attention_tokens));
    if (use_sparse_attention) {
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterSparseTopkSelections),
          1ull);
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterSparseTopkSlotsSelected),
          static_cast<unsigned long long>(sparse_attention_tokens));
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterSparseTopkCandidatesScored),
          static_cast<unsigned long long>(sparse_candidates_scored));
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterSparseTopkSelectionHash),
          sparse_selection_hash);
    }
  }

  const float softmax_scale = rsqrtf(static_cast<float>(qk_head_dim));
  const uint32_t kv_b_cols = kv_lora_rank;
  const uint32_t kv_b_rows = heads * (qk_nope + v_head);
  for (uint32_t head = 0; head < heads; ++head) {
    for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
      latent_output[latent] = 0.0f;
    }

    float local_m = -INFINITY;
    float local_l = 0.0f;
    for (uint32_t attention_index = 0; attention_index < attention_tokens;
         ++attention_index) {
      const uint32_t token =
          use_sparse_attention
              ? static_cast<uint32_t>(sparse_slots[attention_index])
              : attention_index;
      if (token > position) {
        continue;
      }
      const uint64_t token_base =
          kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                              token, kv_cache_width, 0);
      float score = 0.0f;
      for (uint32_t nope = 0; nope < qk_nope; ++nope) {
        const uint32_t row = head * (qk_nope + v_head) + nope;
        const float q_value = q[head * qk_head_dim + nope];
        for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
          score += q_value *
                   deepseek_fp8_scaled_weight(arena, layout.w_v,
                                              layout.deepseek_kv_b_scale,
                                              kv_b_rows, kv_b_cols, row,
                                              latent) *
                   encoded_to_f32(kv_keys[token_base + latent], dtype);
        }
      }
      const uint32_t q_pe_base = head * qk_head_dim + qk_nope;
      for (uint32_t dim = 0; dim < qk_rope; ++dim) {
        float q_pe = q[q_pe_base + dim];
        if (rope_half != 0) {
          const uint32_t offset = dim % rope_half;
          q_pe = deepseek_rope_value_serial(
              q[q_pe_base + offset], q[q_pe_base + offset + rope_half],
              offset, qk_rope, position, rope_theta, dim >= rope_half);
        }
        score += q_pe *
                 encoded_to_f32(kv_keys[token_base + kv_lora_rank + dim],
                                dtype);
      }
      score *= softmax_scale;
      const float next_m = fmaxf(local_m, score);
      const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
      const float new_scale = expf(score - next_m);
      for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
        latent_output[latent] =
            latent_output[latent] * old_scale +
            encoded_to_f32(kv_keys[token_base + latent], dtype) * new_scale;
      }
      local_l = local_l * old_scale + new_scale;
      local_m = next_m;
    }

    if (local_l > 0.0f && isfinite(local_l)) {
      for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
        latent_output[latent] /= local_l;
      }
    }
    for (uint32_t value = 0; value < v_head; ++value) {
      const uint32_t row = head * (qk_nope + v_head) + qk_nope + value;
      float sum = 0.0f;
      for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
        sum += latent_output[latent] *
               deepseek_fp8_scaled_weight(arena, layout.w_v,
                                          layout.deepseek_kv_b_scale,
                                          kv_b_rows, kv_b_cols, row, latent);
      }
      const uint16_t encoded = f32_to_encoded(sum, dtype);
      projection_input[head * v_head + value] = encoded;
      if (use_sparse_attention && deepseek_runtime_counters != nullptr) {
        const unsigned long long term =
            (static_cast<unsigned long long>(position) + 1ull) *
                1315423911ull ^
            (static_cast<unsigned long long>(head) + 1ull) * 2654435761ull ^
            (static_cast<unsigned long long>(value) + 1ull) * 97531ull ^
            static_cast<unsigned long long>(encoded);
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                deepseek_runtime_counters +
                kDeepSeekRuntimeCounterSparseAttentionOutputHash),
            term);
      }
    }
  }
}
