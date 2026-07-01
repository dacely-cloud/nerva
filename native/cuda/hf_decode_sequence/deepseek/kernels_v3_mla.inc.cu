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
  const uint32_t rope_half = qk_rope / 2u;
  for (uint32_t dim = threadIdx.x; dim < qk_rope; dim += blockDim.x) {
    float value = kv_a[kv_lora_rank + dim];
    if (rope_half != 0) {
      const uint32_t offset = dim % rope_half;
      value = deepseek_rope_value_serial(
          kv_a[kv_lora_rank + offset], kv_a[kv_lora_rank + offset + rope_half],
          offset, qk_rope, position, rope_theta, dim >= rope_half, layout);
    }
    kv_keys[write_base + kv_lora_rank + dim] = f32_to_encoded(value, dtype);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    deepseek_session_write_v32_fp8_ds_mla_kv(
        deepseek_v32_mla_kv, deepseek_v32_mla_kv_offset_bytes,
        deepseek_v32_mla_kv_block_count, layout, position, dtype,
        kv_latent_norm, kv_a, rope_theta);
  }
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
    const float value = token_input[row];
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
    token_output[row] =
        f32_to_encoded(token_input[row] * scale * norm_weight, output_dtype);
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
  const uint32_t rope_half = qk_rope / 2u;
  for (uint32_t dim = threadIdx.x; dim < qk_rope; dim += blockDim.x) {
    float value = kv_a[kv_lora_rank + dim];
    if (rope_half != 0) {
      const uint32_t offset = dim % rope_half;
      value = deepseek_rope_value_serial(
          kv_a[kv_lora_rank + offset], kv_a[kv_lora_rank + offset + rope_half],
          offset, qk_rope, position, rope_theta, dim >= rope_half, layout);
    }
    kv_keys[write_base + kv_lora_rank + dim] = f32_to_encoded(value, dtype);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    deepseek_session_write_v32_fp8_ds_mla_kv(
        deepseek_v32_mla_kv, deepseek_v32_mla_kv_offset_bytes,
        deepseek_v32_mla_kv_block_count, layout, position, dtype,
        kv_latent_norm, kv_a, rope_theta);
  }
}

__global__ void hf_deepseek_v3_mla_attention_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t heads, uint32_t *step_cursor,
    uint32_t max_steps, float rope_theta, const float *q,
    uint16_t *kv_keys, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint16_t *projection_input,
    const uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    const uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint64_t *deepseek_runtime_counters) {
  if (blockIdx.x >= heads ||
      (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  const uint32_t head = blockIdx.x;
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t v_head = layout.deepseek_v_head_dim;
  const uint32_t qk_head_dim = qk_nope + qk_rope;
  const uint32_t kv_cache_width = kv_lora_rank + qk_rope;
  if (heads == 0 || kv_lora_rank == 0 || qk_nope == 0 || qk_rope == 0 ||
      v_head == 0 || qk_head_dim == 0 ||
      layout.w_v == kMissingOffset ||
      layout.deepseek_kv_b_scale == kMissingOffset) {
    return;
  }

  extern __shared__ float shared[];
  float *latent_output = shared;

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
  if (threadIdx.x == 0 && head == 0 && deepseek_runtime_counters != nullptr) {
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

  for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
       latent += blockDim.x) {
    latent_output[latent] = 0.0f;
  }
  __syncthreads();

  const uint32_t rope_half = qk_rope / 2u;
  const float softmax_scale = rsqrtf(static_cast<float>(qk_head_dim));
  const uint32_t kv_b_cols = kv_lora_rank;
  const uint32_t kv_b_rows = heads * (qk_nope + v_head);
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
    float score_part = 0.0f;
    const uint32_t nope_terms = qk_nope * kv_lora_rank;
    for (uint32_t term = threadIdx.x; term < nope_terms;
         term += blockDim.x) {
      const uint32_t nope = term / kv_lora_rank;
      const uint32_t latent = term - nope * kv_lora_rank;
      const uint32_t row = head * (qk_nope + v_head) + nope;
      const float q_value = q[head * qk_head_dim + nope];
      score_part +=
          q_value *
          deepseek_fp8_scaled_weight(arena, layout.w_v,
                                     layout.deepseek_kv_b_scale, kv_b_rows,
                                     kv_b_cols, row, latent) *
          encoded_to_f32(kv_keys[token_base + latent], dtype);
    }
    for (uint32_t dim = threadIdx.x; dim < qk_rope; dim += blockDim.x) {
      float q_pe = q[head * qk_head_dim + qk_nope + dim];
      if (rope_half != 0) {
        const uint32_t offset = dim % rope_half;
        q_pe = deepseek_rope_value_serial(
            q[head * qk_head_dim + qk_nope + offset],
            q[head * qk_head_dim + qk_nope + offset + rope_half],
            offset, qk_rope, position, rope_theta, dim >= rope_half, layout);
      }
      score_part += q_pe * encoded_to_f32(
                               kv_keys[token_base + kv_lora_rank + dim],
                               dtype);
    }
    float score = block_sum(score_part) * softmax_scale;
    const float next_m = fmaxf(local_m, score);
    const float old_scale = local_l == 0.0f ? 0.0f : expf(local_m - next_m);
    const float new_scale = expf(score - next_m);
    for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
         latent += blockDim.x) {
      latent_output[latent] =
          latent_output[latent] * old_scale +
          encoded_to_f32(kv_keys[token_base + latent], dtype) * new_scale;
    }
    local_l = local_l * old_scale + new_scale;
    local_m = next_m;
    __syncthreads();
  }

  if (local_l > 0.0f && isfinite(local_l)) {
    for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
         latent += blockDim.x) {
      latent_output[latent] /= local_l;
    }
  }
  __syncthreads();

  for (uint32_t value = threadIdx.x; value < v_head; value += blockDim.x) {
    const uint32_t row = head * (qk_nope + v_head) + qk_nope + value;
    float sum = 0.0f;
    for (uint32_t latent = 0; latent < kv_lora_rank; ++latent) {
      sum += latent_output[latent] *
             deepseek_fp8_scaled_weight(arena, layout.w_v,
                                        layout.deepseek_kv_b_scale, kv_b_rows,
                                        kv_b_cols, row, latent);
    }
    const uint16_t encoded = f32_to_encoded(sum, dtype);
    projection_input[head * v_head + value] = encoded;
    if (use_sparse_attention && deepseek_runtime_counters != nullptr) {
      const unsigned long long term =
          (static_cast<unsigned long long>(position) + 1ull) * 1315423911ull ^
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
