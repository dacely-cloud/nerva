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
    const int32_t *sparse_topk_slots,
    const uint32_t *sparse_topk_count,
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
      layout.w_v == kMissingOffset) {
    return;
  }
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  if (!bf16_storage && layout.deepseek_kv_b_scale == kMissingOffset) {
    return;
  }

  extern __shared__ float shared[];
  float *latent_output = shared;
  float *q_nope_latent = shared + kv_lora_rank;
  float *q_rope = q_nope_latent + kv_lora_rank;
  int32_t *local_sparse_slots = reinterpret_cast<int32_t *>(q_rope + qk_rope);
  float *sparse_scores =
      reinterpret_cast<float *>(local_sparse_slots + kDeepSeekSessionMaxSparseTopK);
  __shared__ uint32_t sparse_attention_tokens_shared;
  __shared__ uint32_t sparse_candidates_scored_shared;
  __shared__ unsigned long long sparse_selection_hash_shared;
  __shared__ uint32_t use_sparse_attention_shared;

  uint32_t sparse_candidates_scored = 0;
  unsigned long long sparse_selection_hash = 0ull;
  const bool sparse_indexer_available =
      deepseek_indexer_state != nullptr && deepseek_indexer_kv != nullptr &&
      deepseek_v32_indexer_query_state_supported(layout) &&
      layout.deepseek_index_topk != 0 &&
      deepseek_indexer_kv_block_count != 0;
  const uint32_t sparse_capacity =
      sparse_indexer_available
          ? deepseek_indexer_kv_block_count * kDeepSeekV32IndexerKvBlockTokens
          : 0u;
  const uint32_t sparse_candidate_tokens =
      sparse_indexer_available ? min(position + 1u, sparse_capacity) : 0u;
  const uint32_t sparse_topk_limit =
      min(min(layout.deepseek_index_topk, sparse_candidate_tokens),
          kDeepSeekSessionMaxSparseTopK);
  const bool sparse_full_prefix =
      sparse_candidate_tokens != 0 &&
      sparse_topk_limit >= sparse_candidate_tokens;
  const bool has_precomputed_sparse =
      sparse_topk_slots != nullptr && sparse_topk_count != nullptr;
  uint32_t sparse_attention_tokens = 0;
  bool use_sparse_attention = false;
  if (threadIdx.x == 0) {
    sparse_attention_tokens_shared = 0u;
    sparse_candidates_scored_shared = 0u;
    sparse_selection_hash_shared = 0ull;
    use_sparse_attention_shared = 0u;
  }
  __syncthreads();
  if (has_precomputed_sparse) {
    if (threadIdx.x == 0) {
      sparse_attention_tokens = min(*sparse_topk_count,
                                    kDeepSeekSessionMaxSparseTopK);
      sparse_attention_tokens_shared = sparse_attention_tokens;
      use_sparse_attention_shared = sparse_attention_tokens != 0 ? 1u : 0u;
    }
  } else if (sparse_full_prefix) {
    sparse_attention_tokens = sparse_candidate_tokens;
    if (threadIdx.x == 0) {
      for (uint32_t rank = 0; rank < sparse_attention_tokens; ++rank) {
        sparse_selection_hash +=
            (static_cast<unsigned long long>(position) + 1ull) *
                1315423911ull ^
            (static_cast<unsigned long long>(rank) + 1ull) * 2654435761ull ^
            (static_cast<unsigned long long>(rank) + 1ull);
      }
      sparse_attention_tokens_shared = sparse_attention_tokens;
      sparse_selection_hash_shared = sparse_selection_hash;
    }
  } else {
    if (threadIdx.x == 0) {
      sparse_attention_tokens =
          deepseek_session_select_v32_sparse_slots(
            layout, step_cursor, max_steps, deepseek_indexer_state,
            deepseek_indexer_state_offset_bytes, deepseek_indexer_kv,
            deepseek_indexer_kv_offset_bytes, deepseek_indexer_kv_block_count,
            kv_block_count, kv_block_table,
            local_sparse_slots, sparse_scores, &sparse_candidates_scored,
            &sparse_selection_hash);
      sparse_attention_tokens_shared = sparse_attention_tokens;
      sparse_candidates_scored_shared = sparse_candidates_scored;
      sparse_selection_hash_shared = sparse_selection_hash;
      use_sparse_attention_shared = sparse_attention_tokens != 0 ? 1u : 0u;
    }
  }
  __syncthreads();
  sparse_attention_tokens = sparse_attention_tokens_shared;
  sparse_candidates_scored = sparse_candidates_scored_shared;
  sparse_selection_hash = sparse_selection_hash_shared;
  use_sparse_attention = use_sparse_attention_shared != 0u;
  const bool record_sparse_attention = sparse_full_prefix || use_sparse_attention;
  const bool record_sparse_selection_metrics =
      record_sparse_attention && !has_precomputed_sparse;
  const uint32_t attention_tokens =
      record_sparse_attention ? sparse_attention_tokens : position + 1u;
  if (threadIdx.x == 0 && head == 0 && deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterRawAttentionTokensScanned),
        static_cast<unsigned long long>(attention_tokens));
    if (record_sparse_selection_metrics) {
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

  const uint32_t kv_b_cols = kv_lora_rank;
  const uint8_t *kv_b_weight =
      reinterpret_cast<const uint8_t *>(arena + layout.w_v);
  const uint16_t *kv_b_scale =
      bf16_storage ? nullptr : arena + layout.deepseek_kv_b_scale;
  const uint32_t kv_b_scale_cols = (kv_b_cols + 127u) / 128u;
  for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
       latent += blockDim.x) {
    latent_output[latent] = 0.0f;
    float sum = 0.0f;
    uint32_t active_scale_row = UINT32_MAX;
    float active_scale = 0.0f;
    for (uint32_t nope = 0; nope < qk_nope; ++nope) {
      const uint32_t row = head * (qk_nope + v_head) + nope;
      const uint32_t scale_row = row / 128u;
      if (!bf16_storage && scale_row != active_scale_row) {
        active_scale_row = scale_row;
        active_scale =
            f32_from_u16_slots(kv_b_scale,
                               scale_row * kv_b_scale_cols + latent / 128u);
      }
      const float weight =
          bf16_storage
              ? deepseek_bf16_weight(arena, layout.w_v,
                                     heads * (qk_nope + v_head),
                                     kv_b_cols, row, latent)
              : nerva::deepseek::f8_e4m3fn_bits_to_f32(
                    kv_b_weight[static_cast<uint64_t>(row) * kv_b_cols +
                                latent]) *
                    active_scale;
      sum += f32_to_model_dtype(q[head * qk_head_dim + nope], dtype) * weight;
    }
    q_nope_latent[latent] = f32_to_model_dtype(sum, dtype);
  }
  for (uint32_t dim = threadIdx.x; dim < qk_rope; dim += blockDim.x) {
    float q_pe =
        f32_to_model_dtype(q[head * qk_head_dim + qk_nope + dim], dtype);
    if ((qk_rope & 1u) == 0u) {
      const uint32_t even = dim & ~1u;
      const uint32_t odd = even + 1u;
      q_pe = deepseek_rope_value_gptj(
          f32_to_model_dtype(q[head * qk_head_dim + qk_nope + even], dtype),
          f32_to_model_dtype(q[head * qk_head_dim + qk_nope + odd], dtype),
          dim, qk_rope, position, rope_theta, layout);
    }
    q_rope[dim] = f32_to_model_dtype(q_pe, dtype);
  }
  __syncthreads();

  const float softmax_scale = deepseek_mla_attention_scale(layout, qk_head_dim);
  float local_m = -INFINITY;
  float local_l = 0.0f;
  for (uint32_t attention_index = 0; attention_index < attention_tokens;
       ++attention_index) {
    const uint32_t token =
        use_sparse_attention
            ? static_cast<uint32_t>(
                  has_precomputed_sparse
                      ? sparse_topk_slots[attention_index]
                      : local_sparse_slots[attention_index])
            : attention_index;
    if (token > position) {
      continue;
    }
    const uint64_t token_base =
        kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                            token, kv_cache_width, 0);
    float score_part = 0.0f;
    for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
         latent += blockDim.x) {
      score_part += q_nope_latent[latent] *
                    encoded_to_f32(kv_keys[token_base + latent], dtype);
    }
    for (uint32_t dim = threadIdx.x; dim < qk_rope; dim += blockDim.x) {
      score_part += q_rope[dim] *
                    encoded_to_f32(kv_keys[token_base + kv_lora_rank + dim],
                                   dtype);
    }
    float score = block_sum(score_part) * softmax_scale;
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

  if (local_l > 0.0f && isfinite(local_l)) {
    for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
         latent += blockDim.x) {
      latent_output[latent] = f32_to_model_dtype(latent_output[latent] / local_l,
                                                 dtype);
    }
  }
  __syncthreads();

  for (uint32_t value = threadIdx.x; value < v_head; value += blockDim.x) {
    const uint32_t row = head * (qk_nope + v_head) + qk_nope + value;
    float sum = 0.0f;
    const uint32_t row_scale_base = (row / 128u) * kv_b_scale_cols;
    for (uint32_t block_start = 0; block_start < kv_lora_rank;
         block_start += 128u) {
      const uint32_t block_end =
          min(block_start + 128u, kv_lora_rank);
      const float scale =
          bf16_storage
              ? 1.0f
              : f32_from_u16_slots(kv_b_scale,
                                   row_scale_base + block_start / 128u);
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
        sum += latent_output[latent] * weight;
      }
    }
    const uint16_t encoded = f32_to_encoded(sum, dtype);
    projection_input[head * v_head + value] = encoded;
    const bool full_output_hash = heads <= 4;
    if ((full_output_hash || head == 0) && record_sparse_attention &&
        deepseek_runtime_counters != nullptr) {
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

__global__ void hf_deepseek_v3_mla_query_latent_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype, uint32_t heads,
    const float *q, float *q_nope_latent) {
  const uint32_t head = blockIdx.x;
  if (head >= heads || q == nullptr || q_nope_latent == nullptr ||
      layout.w_v == kMissingOffset) {
    return;
  }
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t v_head = layout.deepseek_v_head_dim;
  const uint32_t qk_head_dim = qk_nope + qk_rope;
  if (kv_lora_rank == 0 || qk_nope == 0 || qk_head_dim == 0 ||
      v_head == 0) {
    return;
  }
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  if (!bf16_storage && layout.deepseek_kv_b_scale == kMissingOffset) {
    return;
  }
  const uint32_t kv_b_cols = kv_lora_rank;
  const uint8_t *kv_b_weight =
      reinterpret_cast<const uint8_t *>(arena + layout.w_v);
  const uint16_t *kv_b_scale =
      bf16_storage ? nullptr : arena + layout.deepseek_kv_b_scale;
  const uint32_t kv_b_scale_cols = (kv_b_cols + 127u) / 128u;
  float *head_latent =
      q_nope_latent + static_cast<uint64_t>(head) * kv_lora_rank;
  for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
       latent += blockDim.x) {
    float sum = 0.0f;
    uint32_t active_scale_row = UINT32_MAX;
    float active_scale = 0.0f;
    for (uint32_t nope = 0; nope < qk_nope; ++nope) {
      const uint32_t row = head * (qk_nope + v_head) + nope;
      const uint32_t scale_row = row / 128u;
      if (!bf16_storage && scale_row != active_scale_row) {
        active_scale_row = scale_row;
        active_scale =
            f32_from_u16_slots(kv_b_scale,
                               scale_row * kv_b_scale_cols + latent / 128u);
      }
      const float weight =
          bf16_storage
              ? deepseek_bf16_weight(arena, layout.w_v,
                                     heads * (qk_nope + v_head),
                                     kv_b_cols, row, latent)
              : nerva::deepseek::f8_e4m3fn_bits_to_f32(
                    kv_b_weight[static_cast<uint64_t>(row) * kv_b_cols +
                                latent]) *
                    active_scale;
      sum += f32_to_model_dtype(q[head * qk_head_dim + nope], dtype) * weight;
    }
    head_latent[latent] = f32_to_model_dtype(sum, dtype);
  }
}

__global__ void hf_deepseek_v3_mla_attention_chunk_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t heads, uint32_t *step_cursor,
    uint32_t max_steps, float rope_theta, const float *q,
    const float *q_nope_latent, uint16_t *kv_keys, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint32_t attention_chunks,
    float *partial_latent, float *partial_m, float *partial_l,
    const int32_t *sparse_topk_slots,
    const uint32_t *sparse_topk_count,
    uint64_t *deepseek_runtime_counters) {
  __shared__ uint32_t position_shared;
  __shared__ uint32_t attention_tokens_shared;
  if (threadIdx.x == 0) {
    position_shared = step_cursor == nullptr ? 0 : *step_cursor;
    const bool has_precomputed_sparse =
        sparse_topk_slots != nullptr && sparse_topk_count != nullptr;
    attention_tokens_shared =
        has_precomputed_sparse
            ? min(*sparse_topk_count, kDeepSeekSessionMaxSparseTopK)
            : position_shared + 1u;
  }
  __syncthreads();

  const uint32_t head = blockIdx.x;
  const uint32_t selected_slot = blockIdx.y;
  const uint32_t position = position_shared;
  if (head >= heads || selected_slot >= attention_chunks ||
      attention_chunks == 0 || position >= max_steps ||
      partial_latent == nullptr || partial_m == nullptr ||
      partial_l == nullptr || q == nullptr || q_nope_latent == nullptr ||
      kv_keys == nullptr) {
    return;
  }

  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t v_head = layout.deepseek_v_head_dim;
  const uint32_t qk_head_dim = qk_nope + qk_rope;
  const uint32_t kv_cache_width = kv_lora_rank + qk_rope;
  if (heads == 0 || kv_lora_rank == 0 || qk_nope == 0 || qk_rope == 0 ||
      v_head == 0 || qk_head_dim == 0 || layout.w_v == kMissingOffset) {
    return;
  }
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  if (!bf16_storage && layout.deepseek_kv_b_scale == kMissingOffset) {
    return;
  }

  const uint64_t partial_slot =
      static_cast<uint64_t>(head) * attention_chunks + selected_slot;
  const uint32_t attention_tokens = attention_tokens_shared;
  const uint32_t chunk_start =
      selected_slot * kDeepSeekMlaDecodeAttentionChunkTokens;
  if (chunk_start >= attention_tokens) {
    if (threadIdx.x == 0) {
      partial_m[partial_slot] = -INFINITY;
      partial_l[partial_slot] = 0.0f;
    }
    return;
  }
  const uint32_t chunk_end =
      min(chunk_start + kDeepSeekMlaDecodeAttentionChunkTokens,
          attention_tokens);

  extern __shared__ float shared[];
  float *latent_output = shared;
  float *q_rope = shared + kv_lora_rank;
  const float *head_q_nope_latent =
      q_nope_latent + static_cast<uint64_t>(head) * kv_lora_rank;
  for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
       latent += blockDim.x) {
    latent_output[latent] = 0.0f;
  }
  for (uint32_t dim = threadIdx.x; dim < qk_rope; dim += blockDim.x) {
    float q_pe =
        f32_to_model_dtype(q[head * qk_head_dim + qk_nope + dim], dtype);
    if ((qk_rope & 1u) == 0u) {
      const uint32_t even = dim & ~1u;
      const uint32_t odd = even + 1u;
      q_pe = deepseek_rope_value_gptj(
          f32_to_model_dtype(q[head * qk_head_dim + qk_nope + even], dtype),
          f32_to_model_dtype(q[head * qk_head_dim + qk_nope + odd], dtype),
          dim, qk_rope, position, rope_theta, layout);
    }
    q_rope[dim] = f32_to_model_dtype(q_pe, dtype);
  }
  __syncthreads();

  const bool use_sparse_attention =
      sparse_topk_slots != nullptr && sparse_topk_count != nullptr;
  const float softmax_scale = deepseek_mla_attention_scale(layout, qk_head_dim);
  float local_m = -INFINITY;
  float local_l = 0.0f;
  for (uint32_t attention_index = chunk_start; attention_index < chunk_end;
       ++attention_index) {
    uint32_t token = attention_index;
    if (use_sparse_attention) {
      const int32_t sparse_slot = sparse_topk_slots[attention_index];
      if (sparse_slot < 0) {
        continue;
      }
      token = static_cast<uint32_t>(sparse_slot);
    }
    if (token > position) {
      continue;
    }
    const uint64_t token_base =
        kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                            token, kv_cache_width, 0);
    float score_part = 0.0f;
    for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
         latent += blockDim.x) {
      score_part += head_q_nope_latent[latent] *
                    encoded_to_f32(kv_keys[token_base + latent], dtype);
    }
    for (uint32_t dim = threadIdx.x; dim < qk_rope; dim += blockDim.x) {
      score_part += q_rope[dim] *
                    encoded_to_f32(kv_keys[token_base + kv_lora_rank + dim],
                                   dtype);
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

  if (threadIdx.x == 0) {
    partial_m[partial_slot] = local_m;
    partial_l[partial_slot] = local_l;
    if (head == 0 && deepseek_runtime_counters != nullptr) {
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterRawAttentionTokensScanned),
          static_cast<unsigned long long>(chunk_end - chunk_start));
    }
  }
  for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
       latent += blockDim.x) {
    partial_latent[partial_slot * kv_lora_rank + latent] =
        latent_output[latent];
  }
}

__global__ void hf_deepseek_v3_mla_attention_reduce_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t heads, uint32_t *step_cursor, uint32_t max_steps,
    uint32_t attention_chunks, const float *partial_latent,
    const float *partial_m, const float *partial_l,
    uint16_t *projection_input, uint64_t *deepseek_runtime_counters,
    uint32_t record_sparse_attention) {
  __shared__ uint32_t position_shared;
  if (threadIdx.x == 0) {
    position_shared = step_cursor == nullptr ? 0 : *step_cursor;
  }
  __syncthreads();

  const uint32_t head = blockIdx.x;
  const uint32_t position = position_shared;
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t v_head = layout.deepseek_v_head_dim;
  if (head >= heads || position >= max_steps || attention_chunks == 0 ||
      kv_lora_rank == 0 || qk_nope == 0 || v_head == 0 ||
      partial_latent == nullptr || partial_m == nullptr ||
      partial_l == nullptr || projection_input == nullptr ||
      layout.w_v == kMissingOffset) {
    return;
  }
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  if (!bf16_storage && layout.deepseek_kv_b_scale == kMissingOffset) {
    return;
  }

  extern __shared__ float shared[];
  float *chunk_weights = shared;
  float *latent_output = shared + attention_chunks;
  if (threadIdx.x == 0) {
    float global_m = -INFINITY;
    float global_l = 0.0f;
    for (uint32_t chunk = 0; chunk < attention_chunks; ++chunk) {
      const uint64_t slot =
          static_cast<uint64_t>(head) * attention_chunks + chunk;
      const float chunk_l = partial_l[slot];
      if (chunk_l <= 0.0f || !isfinite(chunk_l)) {
        continue;
      }
      const float chunk_m = partial_m[slot];
      const float next_m = fmaxf(global_m, chunk_m);
      const float old_scale =
          global_l == 0.0f ? 0.0f : expf(global_m - next_m);
      const float new_scale = expf(chunk_m - next_m);
      global_l = global_l * old_scale + chunk_l * new_scale;
      global_m = next_m;
    }
    for (uint32_t chunk = 0; chunk < attention_chunks; ++chunk) {
      const uint64_t slot =
          static_cast<uint64_t>(head) * attention_chunks + chunk;
      const float chunk_l = partial_l[slot];
      if (global_l > 0.0f && isfinite(global_l) && chunk_l > 0.0f &&
          isfinite(chunk_l)) {
        chunk_weights[chunk] = expf(partial_m[slot] - global_m) / global_l;
      } else {
        chunk_weights[chunk] = 0.0f;
      }
    }
  }
  __syncthreads();

  for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
       latent += blockDim.x) {
    float value = 0.0f;
    for (uint32_t chunk = 0; chunk < attention_chunks; ++chunk) {
      const float weight = chunk_weights[chunk];
      if (weight != 0.0f) {
        const uint64_t slot =
            static_cast<uint64_t>(head) * attention_chunks + chunk;
        value += partial_latent[slot * kv_lora_rank + latent] * weight;
      }
    }
    latent_output[latent] = f32_to_model_dtype(value, dtype);
  }
  __syncthreads();

  const uint32_t kv_b_cols = kv_lora_rank;
  const uint8_t *kv_b_weight =
      reinterpret_cast<const uint8_t *>(arena + layout.w_v);
  const uint16_t *kv_b_scale =
      bf16_storage ? nullptr : arena + layout.deepseek_kv_b_scale;
  const uint32_t kv_b_scale_cols = (kv_b_cols + 127u) / 128u;
  for (uint32_t value = threadIdx.x; value < v_head; value += blockDim.x) {
    const uint32_t row = head * (qk_nope + v_head) + qk_nope + value;
    float sum = 0.0f;
    const uint32_t row_scale_base = (row / 128u) * kv_b_scale_cols;
    for (uint32_t block_start = 0; block_start < kv_lora_rank;
         block_start += 128u) {
      const uint32_t block_end = min(block_start + 128u, kv_lora_rank);
      const float scale =
          bf16_storage
              ? 1.0f
              : f32_from_u16_slots(kv_b_scale,
                                   row_scale_base + block_start / 128u);
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
        sum += latent_output[latent] * weight;
      }
    }
    const uint16_t encoded = f32_to_encoded(sum, dtype);
    projection_input[head * v_head + value] = encoded;
    const bool full_output_hash = heads <= 4;
    if (record_sparse_attention != 0 &&
        (full_output_hash || head == 0) &&
        deepseek_runtime_counters != nullptr) {
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

__global__ void hf_deepseek_v3_mla_attention_tokens_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t heads, uint32_t max_steps, float rope_theta,
    uint32_t chunk_start, uint32_t chunk_tokens, const float *q_tokens,
    uint32_t q_stride, uint16_t *kv_keys, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint16_t *attn_out,
    uint32_t attn_stride, const int32_t *sparse_topk_slots,
    uint32_t sparse_topk_stride, const uint32_t *sparse_topk_count,
    uint64_t *deepseek_runtime_counters) {
  const uint32_t local_token = blockIdx.x;
  const uint32_t head = blockIdx.y;
  if (local_token >= chunk_tokens || head >= heads || q_tokens == nullptr ||
      kv_keys == nullptr || attn_out == nullptr) {
    return;
  }
  const uint32_t position = chunk_start + local_token;
  if (position >= max_steps) {
    return;
  }
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t v_head = layout.deepseek_v_head_dim;
  const uint32_t qk_head_dim = qk_nope + qk_rope;
  const uint32_t kv_cache_width = kv_lora_rank + qk_rope;
  if (heads == 0 || kv_lora_rank == 0 || qk_nope == 0 || qk_rope == 0 ||
      v_head == 0 || qk_head_dim == 0 ||
      q_stride < heads * qk_head_dim || attn_stride < heads * v_head ||
      layout.w_v == kMissingOffset) {
    return;
  }
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  if (!bf16_storage && layout.deepseek_kv_b_scale == kMissingOffset) {
    return;
  }

  extern __shared__ float shared[];
  float *latent_output = shared;
  float *q_nope_latent = shared + kv_lora_rank;
  float *q_rope = q_nope_latent + kv_lora_rank;

  const float *q =
      q_tokens + static_cast<uint64_t>(local_token) * q_stride;
  uint16_t *out =
      attn_out + static_cast<uint64_t>(local_token) * attn_stride;
  const uint32_t kv_b_cols = kv_lora_rank;
  const uint8_t *kv_b_weight =
      reinterpret_cast<const uint8_t *>(arena + layout.w_v);
  const uint16_t *kv_b_scale =
      bf16_storage ? nullptr : arena + layout.deepseek_kv_b_scale;
  const uint32_t kv_b_scale_cols = (kv_b_cols + 127u) / 128u;

  for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
       latent += blockDim.x) {
    latent_output[latent] = 0.0f;
    float sum = 0.0f;
    uint32_t active_scale_row = UINT32_MAX;
    float active_scale = 0.0f;
    for (uint32_t nope = 0; nope < qk_nope; ++nope) {
      const uint32_t row = head * (qk_nope + v_head) + nope;
      const uint32_t scale_row = row / 128u;
      if (!bf16_storage && scale_row != active_scale_row) {
        active_scale_row = scale_row;
        active_scale =
            f32_from_u16_slots(kv_b_scale,
                               scale_row * kv_b_scale_cols + latent / 128u);
      }
      const float weight =
          bf16_storage
              ? deepseek_bf16_weight(arena, layout.w_v,
                                     heads * (qk_nope + v_head),
                                     kv_b_cols, row, latent)
              : nerva::deepseek::f8_e4m3fn_bits_to_f32(
                    kv_b_weight[static_cast<uint64_t>(row) * kv_b_cols +
                                latent]) *
                    active_scale;
      sum += f32_to_model_dtype(q[head * qk_head_dim + nope], dtype) * weight;
    }
    q_nope_latent[latent] = f32_to_model_dtype(sum, dtype);
  }
  for (uint32_t dim = threadIdx.x; dim < qk_rope; dim += blockDim.x) {
    float q_pe =
        f32_to_model_dtype(q[head * qk_head_dim + qk_nope + dim], dtype);
    if ((qk_rope & 1u) == 0u) {
      const uint32_t even = dim & ~1u;
      const uint32_t odd = even + 1u;
      q_pe = deepseek_rope_value_gptj(
          f32_to_model_dtype(q[head * qk_head_dim + qk_nope + even], dtype),
          f32_to_model_dtype(q[head * qk_head_dim + qk_nope + odd], dtype),
          dim, qk_rope, position, rope_theta, layout);
    }
    q_rope[dim] = f32_to_model_dtype(q_pe, dtype);
  }
  __syncthreads();

  const float softmax_scale = deepseek_mla_attention_scale(layout, qk_head_dim);
  float local_m = -INFINITY;
  float local_l = 0.0f;
  const bool use_sparse_attention =
      sparse_topk_slots != nullptr && sparse_topk_count != nullptr &&
      sparse_topk_stride != 0 && sparse_topk_count[local_token] != 0;
  const uint32_t attention_tokens =
      use_sparse_attention
          ? min(sparse_topk_count[local_token], sparse_topk_stride)
          : position + 1u;
  const int32_t *token_sparse_slots =
      use_sparse_attention
          ? sparse_topk_slots +
                static_cast<uint64_t>(local_token) * sparse_topk_stride
          : nullptr;
  for (uint32_t attention_index = 0; attention_index < attention_tokens;
       ++attention_index) {
    const uint32_t token =
        use_sparse_attention
            ? static_cast<uint32_t>(token_sparse_slots[attention_index])
            : attention_index;
    if (token > position) {
      continue;
    }
    const uint64_t token_base =
        kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                            token, kv_cache_width, 0);
    float score_part = 0.0f;
    for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
         latent += blockDim.x) {
      score_part += q_nope_latent[latent] *
                    encoded_to_f32(kv_keys[token_base + latent], dtype);
    }
    for (uint32_t dim = threadIdx.x; dim < qk_rope; dim += blockDim.x) {
      score_part += q_rope[dim] *
                    encoded_to_f32(kv_keys[token_base + kv_lora_rank + dim],
                                   dtype);
    }
    float score = block_sum(score_part) * softmax_scale;
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

  if (local_l > 0.0f && isfinite(local_l)) {
    for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
         latent += blockDim.x) {
      latent_output[latent] = f32_to_model_dtype(latent_output[latent] / local_l,
                                                 dtype);
    }
  }
  __syncthreads();

  for (uint32_t value = threadIdx.x; value < v_head; value += blockDim.x) {
    const uint32_t row = head * (qk_nope + v_head) + qk_nope + value;
    float sum = 0.0f;
    const uint32_t row_scale_base = (row / 128u) * kv_b_scale_cols;
    for (uint32_t block_start = 0; block_start < kv_lora_rank;
         block_start += 128u) {
      const uint32_t block_end = min(block_start + 128u, kv_lora_rank);
      const float scale =
          bf16_storage
              ? 1.0f
              : f32_from_u16_slots(kv_b_scale,
                                   row_scale_base + block_start / 128u);
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
        sum += weight * latent_output[latent];
      }
    }
    out[head * v_head + value] = f32_to_encoded(sum, dtype);
  }
  if (threadIdx.x == 0 && head == 0 && deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterRawAttentionTokensScanned),
        static_cast<unsigned long long>(attention_tokens));
  }
}

// Grouped variant of hf_deepseek_v3_mla_attention_tokens_kernel: one block
// processes kDeepSeekMlaPrefillHeadGroup heads of the same token, so the
// latent KV history is fetched from global memory once per head group
// instead of once per head. All per-head arithmetic (score partition per
// thread, block reduction tree, running max/scale softmax accumulation and
// the final V projection) is bit-identical to the ungrouped kernel; heads
// within the group are simply interleaved per attention token while each
// head still observes the exact same sequence of operations.
// Requires kv_lora_rank <= kDeepSeekMlaPrefillMaxLatentSlots * blockDim.x
// and qk_rope <= blockDim.x (the launcher falls back to the ungrouped
// kernel otherwise).
__global__ void __launch_bounds__(kDecodeThreads, 4)
    hf_deepseek_v3_mla_attention_tokens_grouped_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t layer_index,
    uint32_t dtype, uint32_t heads, uint32_t max_steps, float rope_theta,
    uint32_t chunk_start, uint32_t chunk_tokens, const float *q_tokens,
    uint32_t q_stride, uint16_t *kv_keys, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint16_t *attn_out,
    uint32_t attn_stride, const int32_t *sparse_topk_slots,
    uint32_t sparse_topk_stride, const uint32_t *sparse_topk_count,
    uint64_t *deepseek_runtime_counters) {
  const uint32_t local_token = blockIdx.x;
  const uint32_t group_start = blockIdx.y * kDeepSeekMlaPrefillHeadGroup;
  if (local_token >= chunk_tokens || group_start >= heads ||
      q_tokens == nullptr || kv_keys == nullptr || attn_out == nullptr) {
    return;
  }
  const uint32_t position = chunk_start + local_token;
  if (position >= max_steps) {
    return;
  }
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t v_head = layout.deepseek_v_head_dim;
  const uint32_t qk_head_dim = qk_nope + qk_rope;
  const uint32_t kv_cache_width = kv_lora_rank + qk_rope;
  if (heads == 0 || kv_lora_rank == 0 || qk_nope == 0 || qk_rope == 0 ||
      v_head == 0 || qk_head_dim == 0 ||
      q_stride < heads * qk_head_dim || attn_stride < heads * v_head ||
      layout.w_v == kMissingOffset ||
      kv_lora_rank > kDeepSeekMlaPrefillMaxLatentSlots * blockDim.x ||
      qk_rope > blockDim.x ||
      blockDim.x > kDeepSeekMlaPrefillMaxWarps * 32u) {
    return;
  }
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  if (!bf16_storage && layout.deepseek_kv_b_scale == kMissingOffset) {
    return;
  }

  const uint32_t group_size =
      min(kDeepSeekMlaPrefillHeadGroup, heads - group_start);

  // Per-head latent accumulators. Every latent slot is owned by the same
  // thread that owns it in the ungrouped kernel
  // (latent % blockDim.x == threadIdx.x), so the updates stay race free.
  extern __shared__ float shared[];
  float *latent_output = shared;
  // Reduction scratch and per-token rescale factors, double buffered so
  // that consecutive iterations of the attention loop only need the two
  // barriers of the reduction itself (the previous iteration's readers are
  // always separated from the next writer by at least two barriers).
  __shared__ float group_warp_sums[2][kDeepSeekMlaPrefillTokensPerIter *
                                      kDeepSeekMlaPrefillHeadGroup *
                                      kDeepSeekMlaPrefillMaxWarps];
  __shared__ float group_old_scale[2][kDeepSeekMlaPrefillTokensPerIter]
                                  [kDeepSeekMlaPrefillHeadGroup];
  __shared__ float group_value_scale[2][kDeepSeekMlaPrefillTokensPerIter]
                                    [kDeepSeekMlaPrefillHeadGroup];
  // Running softmax state per head. The state update is computed once (by
  // one lane of warp 0) with arithmetic identical to the per-thread
  // computation of the ungrouped kernel and the resulting rescale factors
  // are broadcast through shared memory.
  __shared__ float group_m[kDeepSeekMlaPrefillHeadGroup];
  __shared__ float group_l[kDeepSeekMlaPrefillHeadGroup];

  const float *q =
      q_tokens + static_cast<uint64_t>(local_token) * q_stride;
  uint16_t *out =
      attn_out + static_cast<uint64_t>(local_token) * attn_stride;
  const uint32_t kv_b_cols = kv_lora_rank;
  const uint8_t *kv_b_weight =
      reinterpret_cast<const uint8_t *>(arena + layout.w_v);
  const uint16_t *kv_b_scale =
      bf16_storage ? nullptr : arena + layout.deepseek_kv_b_scale;
  const uint32_t kv_b_scale_cols = (kv_b_cols + 127u) / 128u;

  // Thread-private query values for the strided score partition: thread t
  // only touches latent indices t, t + blockDim.x, ... and rope dim t, the
  // exact slots it owns in the ungrouped kernel, so they live in registers.
  float q_nope_latent[kDeepSeekMlaPrefillHeadGroup]
                     [kDeepSeekMlaPrefillMaxLatentSlots];
  float q_rope[kDeepSeekMlaPrefillHeadGroup];

#pragma unroll
  for (uint32_t rank = 0; rank < kDeepSeekMlaPrefillHeadGroup; ++rank) {
    if (rank >= group_size) {
      break;
    }
    const uint32_t head = group_start + rank;
    float *head_latent_output = latent_output + rank * kv_lora_rank;
#pragma unroll
    for (uint32_t slot = 0; slot < kDeepSeekMlaPrefillMaxLatentSlots;
         ++slot) {
      const uint32_t latent = threadIdx.x + slot * blockDim.x;
      if (latent >= kv_lora_rank) {
        break;
      }
      head_latent_output[latent] = 0.0f;
      float sum = 0.0f;
      uint32_t active_scale_row = UINT32_MAX;
      float active_scale = 0.0f;
      for (uint32_t nope = 0; nope < qk_nope; ++nope) {
        const uint32_t row = head * (qk_nope + v_head) + nope;
        const uint32_t scale_row = row / 128u;
        if (!bf16_storage && scale_row != active_scale_row) {
          active_scale_row = scale_row;
          active_scale =
              f32_from_u16_slots(kv_b_scale,
                                 scale_row * kv_b_scale_cols + latent / 128u);
        }
        const float weight =
            bf16_storage
                ? deepseek_bf16_weight(arena, layout.w_v,
                                       heads * (qk_nope + v_head),
                                       kv_b_cols, row, latent)
                : nerva::deepseek::f8_e4m3fn_bits_to_f32(
                      kv_b_weight[static_cast<uint64_t>(row) * kv_b_cols +
                                  latent]) *
                      active_scale;
        sum +=
            f32_to_model_dtype(q[head * qk_head_dim + nope], dtype) * weight;
      }
      q_nope_latent[rank][slot] = f32_to_model_dtype(sum, dtype);
    }
    q_rope[rank] = 0.0f;
    if (threadIdx.x < qk_rope) {
      const uint32_t dim = threadIdx.x;
      float q_pe =
          f32_to_model_dtype(q[head * qk_head_dim + qk_nope + dim], dtype);
      if ((qk_rope & 1u) == 0u) {
        const uint32_t even = dim & ~1u;
        const uint32_t odd = even + 1u;
        q_pe = deepseek_rope_value_gptj(
            f32_to_model_dtype(q[head * qk_head_dim + qk_nope + even], dtype),
            f32_to_model_dtype(q[head * qk_head_dim + qk_nope + odd], dtype),
            dim, qk_rope, position, rope_theta, layout);
      }
      q_rope[rank] = f32_to_model_dtype(q_pe, dtype);
    }
  }

  const float softmax_scale = deepseek_mla_attention_scale(layout, qk_head_dim);
  if (threadIdx.x < kDeepSeekMlaPrefillHeadGroup) {
    group_m[threadIdx.x] = -INFINITY;
    group_l[threadIdx.x] = 0.0f;
  }
  __syncthreads();
  const bool use_sparse_attention =
      sparse_topk_slots != nullptr && sparse_topk_count != nullptr &&
      sparse_topk_stride != 0 && sparse_topk_count[local_token] != 0;
  const uint32_t attention_tokens =
      use_sparse_attention
          ? min(sparse_topk_count[local_token], sparse_topk_stride)
          : position + 1u;
  const int32_t *token_sparse_slots =
      use_sparse_attention
          ? sparse_topk_slots +
                static_cast<uint64_t>(local_token) * sparse_topk_stride
          : nullptr;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5u;
  const uint32_t warp_count = (blockDim.x + 31u) >> 5u;
  uint32_t parity = 0;
  for (uint32_t attention_index = 0; attention_index < attention_tokens;
       attention_index += kDeepSeekMlaPrefillTokensPerIter) {
    // Gather up to kDeepSeekMlaPrefillTokensPerIter attention tokens for a
    // single reduction round. Skipped entries (out of range or beyond the
    // causal position, both uniform across the block) contribute nothing:
    // their softmax-state updates and accumulator updates are suppressed,
    // preserving the exact per-token sequence of the ungrouped kernel.
    bool pair_valid[kDeepSeekMlaPrefillTokensPerIter];
    float kv_latent[kDeepSeekMlaPrefillTokensPerIter]
                   [kDeepSeekMlaPrefillMaxLatentSlots];
    float *warp_sums = group_warp_sums[parity];
#pragma unroll
    for (uint32_t sub = 0; sub < kDeepSeekMlaPrefillTokensPerIter; ++sub) {
      pair_valid[sub] = false;
      const uint32_t index = attention_index + sub;
      if (index >= attention_tokens) {
        continue;
      }
      const uint32_t token =
          use_sparse_attention
              ? static_cast<uint32_t>(token_sparse_slots[index])
              : index;
      if (token > position) {
        continue;
      }
      pair_valid[sub] = true;
      const uint64_t token_base =
          kv_cache_token_base(layer_index, kv_block_count, kv_block_table,
                              token, kv_cache_width, 0);
      // Fetch this thread's KV slice once per token and reuse it for every
      // head in the group; encoded_to_f32 of identical bits yields
      // identical values. The rope slot is only used by the score below,
      // so it does not outlive this iteration.
#pragma unroll
      for (uint32_t slot = 0; slot < kDeepSeekMlaPrefillMaxLatentSlots;
           ++slot) {
        const uint32_t latent = threadIdx.x + slot * blockDim.x;
        if (latent >= kv_lora_rank) {
          break;
        }
        kv_latent[sub][slot] =
            encoded_to_f32(kv_keys[token_base + latent], dtype);
      }
      float kv_rope = 0.0f;
      if (threadIdx.x < qk_rope) {
        kv_rope = encoded_to_f32(
            kv_keys[token_base + kv_lora_rank + threadIdx.x], dtype);
      }
      // Per-thread score partition, identical per head and token to the
      // ungrouped kernel (latent slots in ascending order, then the rope
      // slot), followed by the first stage of a block reduction whose tree
      // is bit-identical to block_sum() per (token, head) value.
#pragma unroll
      for (uint32_t rank = 0; rank < kDeepSeekMlaPrefillHeadGroup; ++rank) {
        if (rank >= group_size) {
          break;
        }
        float score_part = 0.0f;
#pragma unroll
        for (uint32_t slot = 0; slot < kDeepSeekMlaPrefillMaxLatentSlots;
             ++slot) {
          const uint32_t latent = threadIdx.x + slot * blockDim.x;
          if (latent >= kv_lora_rank) {
            break;
          }
          score_part += q_nope_latent[rank][slot] * kv_latent[sub][slot];
        }
        if (threadIdx.x < qk_rope) {
          score_part += q_rope[rank] * kv_rope;
        }
        for (uint32_t offset = 16; offset > 0; offset >>= 1) {
          score_part += __shfl_down_sync(0xffffffffu, score_part, offset);
        }
        if (lane == 0) {
          warp_sums[(sub * kDeepSeekMlaPrefillHeadGroup + rank) *
                        kDeepSeekMlaPrefillMaxWarps +
                    warp] = score_part;
        }
      }
    }
    __syncthreads();
    // Second reduction stage, heads distributed across warps (the tree
    // below is the one block_sum() runs in warp 0, and it is warp
    // agnostic). The warp's lane 0 then advances the head's running
    // max/scale softmax state in token order with the reduced total still
    // in a register; the math matches the ungrouped kernel term for term.
    for (uint32_t rank = warp; rank < group_size; rank += warp_count) {
      float prev_m = group_m[rank];
      float prev_l = group_l[rank];
#pragma unroll
      for (uint32_t sub = 0; sub < kDeepSeekMlaPrefillTokensPerIter; ++sub) {
        if (!pair_valid[sub]) {
          continue;
        }
        float total =
            lane < warp_count
                ? warp_sums[(sub * kDeepSeekMlaPrefillHeadGroup + rank) *
                                kDeepSeekMlaPrefillMaxWarps +
                            lane]
                : 0.0f;
        for (uint32_t offset = 16; offset > 0; offset >>= 1) {
          total += __shfl_down_sync(0xffffffffu, total, offset);
        }
        if (lane == 0) {
          const float score = total * softmax_scale;
          const float next_m = fmaxf(prev_m, score);
          // next_m == max(prev_m, score), so one of the two exponent
          // arguments below is exactly zero; CUDA guarantees
          // expf(+/-0) == 1.0f, so the == 0.0f fast paths are bit-exact
          // (non-zero, inf and NaN arguments still call expf).
          const float old_arg = prev_m - next_m;
          const float old_scale =
              prev_l == 0.0f ? 0.0f
                             : (old_arg == 0.0f ? 1.0f : expf(old_arg));
          const float new_arg = score - next_m;
          const float new_scale = new_arg == 0.0f ? 1.0f : expf(new_arg);
          group_old_scale[parity][sub][rank] = old_scale;
          group_value_scale[parity][sub][rank] =
              f32_to_model_dtype(new_scale, dtype);
          prev_l = prev_l * old_scale + new_scale;
          prev_m = next_m;
        }
      }
      if (lane == 0) {
        group_m[rank] = prev_m;
        group_l[rank] = prev_l;
      }
    }
    __syncthreads();
    // Apply the broadcast rescale factors to the per-head accumulators in
    // token order. Keeping the accumulator in a register between the two
    // updates is exact: the intermediate value is a plain f32 either way.
#pragma unroll
    for (uint32_t rank = 0; rank < kDeepSeekMlaPrefillHeadGroup; ++rank) {
      if (rank >= group_size) {
        break;
      }
      float *head_latent_output = latent_output + rank * kv_lora_rank;
#pragma unroll
      for (uint32_t slot = 0; slot < kDeepSeekMlaPrefillMaxLatentSlots;
           ++slot) {
        const uint32_t latent = threadIdx.x + slot * blockDim.x;
        if (latent >= kv_lora_rank) {
          break;
        }
        float accumulator = head_latent_output[latent];
#pragma unroll
        for (uint32_t sub = 0; sub < kDeepSeekMlaPrefillTokensPerIter;
             ++sub) {
          if (!pair_valid[sub]) {
            continue;
          }
          accumulator =
              accumulator * group_old_scale[parity][sub][rank] +
              kv_latent[sub][slot] * group_value_scale[parity][sub][rank];
        }
        head_latent_output[latent] = accumulator;
      }
    }
    parity ^= 1u;
  }

#pragma unroll
  for (uint32_t rank = 0; rank < kDeepSeekMlaPrefillHeadGroup; ++rank) {
    if (rank >= group_size) {
      break;
    }
    const float head_l = group_l[rank];
    if (head_l > 0.0f && isfinite(head_l)) {
      float *head_latent_output = latent_output + rank * kv_lora_rank;
      for (uint32_t latent = threadIdx.x; latent < kv_lora_rank;
           latent += blockDim.x) {
        head_latent_output[latent] =
            f32_to_model_dtype(head_latent_output[latent] / head_l, dtype);
      }
    }
  }
  __syncthreads();

  for (uint32_t rank = 0; rank < kDeepSeekMlaPrefillHeadGroup; ++rank) {
    if (rank >= group_size) {
      break;
    }
    const uint32_t head = group_start + rank;
    const float *head_latent_output = latent_output + rank * kv_lora_rank;
    for (uint32_t value = threadIdx.x; value < v_head; value += blockDim.x) {
      const uint32_t row = head * (qk_nope + v_head) + qk_nope + value;
      float sum = 0.0f;
      const uint32_t row_scale_base = (row / 128u) * kv_b_scale_cols;
      for (uint32_t block_start = 0; block_start < kv_lora_rank;
           block_start += 128u) {
        const uint32_t block_end = min(block_start + 128u, kv_lora_rank);
        const float scale =
            bf16_storage
                ? 1.0f
                : f32_from_u16_slots(kv_b_scale,
                                     row_scale_base + block_start / 128u);
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
          sum += weight * head_latent_output[latent];
        }
      }
      out[head * v_head + value] = f32_to_encoded(sum, dtype);
    }
  }
  if (threadIdx.x == 0 && group_start == 0 &&
      deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterRawAttentionTokensScanned),
        static_cast<unsigned long long>(attention_tokens));
  }
}
