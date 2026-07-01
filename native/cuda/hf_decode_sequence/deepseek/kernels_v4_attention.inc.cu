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
      const uint32_t left = head_start + qk_nope + offset * 2u;
      const uint32_t right = left + 1u;
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
    float *sparse_topk_score_workspace, uint32_t sparse_topk_score_capacity,
    uint64_t *deepseek_runtime_counters) {
  __shared__ float reduce_scores[kDecodeThreads];
  __shared__ int32_t reduce_slots[kDecodeThreads];
  __shared__ float indexer_weights[kDeepSeekSessionMaxIndexerHeads];
  __shared__ float sort_scores[4096];
  __shared__ int32_t sort_slots[4096];
  __shared__ uint32_t sort_selected;
  __shared__ unsigned long long sort_hash;

  if (blockIdx.x != 0 ||
      (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  if (threadIdx.x == 0 && sparse_topk_count != nullptr) {
    *sparse_topk_count = 0;
  }
  __syncthreads();
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
  const bool can_parallel_select =
      (layout.deepseek_flags & kDeepSeekFlagSparseIndexer) != 0 &&
      layout.deepseek_compress_ratio == 4 && layout.deepseek_index_topk != 0 &&
      index_heads != 0 && index_head_dim != 0 &&
      index_heads <= kDeepSeekSessionMaxIndexerHeads &&
      index_head_dim <= kDeepSeekSessionMaxCompressHeadSize &&
      query_rows != 0 && query_rows <= heads * head_dim &&
      query_rows <= kDeepSeekSessionMaxIndexerQueryValues &&
      compressed_attention_tokens != 0 && topk_limit != 0 &&
      deepseek_indexer_kv != nullptr && deepseek_indexer_kv_block_count != 0 &&
      layout.deepseek_indexer_q != kMissingOffset &&
      layout.deepseek_indexer_q_scale != kMissingOffset &&
      layout.deepseek_indexer_weights != kMissingOffset &&
      sparse_topk_score_workspace != nullptr &&
      sparse_topk_score_capacity >= compressed_attention_tokens;

  if (can_parallel_select && topk_limit >= compressed_attention_tokens) {
    for (uint32_t slot = threadIdx.x; slot < compressed_attention_tokens;
         slot += blockDim.x) {
      sparse_topk_slots[slot] = static_cast<int32_t>(slot);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned long long selection_hash = 0ull;
      for (uint32_t rank = 0; rank < topk_limit; ++rank) {
        selection_hash +=
            (static_cast<unsigned long long>(position) + 1ull) *
                1315423911ull ^
            (static_cast<unsigned long long>(rank) + 1ull) * 2654435761ull ^
            (static_cast<unsigned long long>(rank) + 1ull);
      }
      *sparse_topk_count = topk_limit;
      if (deepseek_runtime_counters != nullptr) {
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                deepseek_runtime_counters +
                kDeepSeekRuntimeCounterSparseTopkSelections),
            1ull);
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                deepseek_runtime_counters +
                kDeepSeekRuntimeCounterSparseTopkSlotsSelected),
            static_cast<unsigned long long>(topk_limit));
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                deepseek_runtime_counters +
                kDeepSeekRuntimeCounterSparseTopkSelectionHash),
            selection_hash);
      }
    }
    return;
  }

  if (can_parallel_select) {
    LayerScratch s =
        layer_scratch_ptrs(scratch, hidden, heads * head_dim, head_dim,
                           intermediate);
    float *indexer_query = s.attn;
    if (threadIdx.x == 0) {
      for (uint32_t head = 0; head < index_heads; ++head) {
        float weight_sum = 0.0f;
        for (uint32_t col = 0; col < hidden; ++col) {
          weight_sum +=
              encoded_to_f32(
                  arena[layout.deepseek_indexer_weights +
                        static_cast<uint64_t>(head) * hidden + col],
                  kDTypeBF16) *
              encoded_to_f32(projection_input[col], dtype);
        }
        indexer_weights[head] = weight_sum;
      }

      for (uint32_t row = 0; row < query_rows; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < layout.deepseek_q_lora_rank; ++col) {
          sum += deepseek_fp8_e8m0_scaled_weight(
                     arena, layout.deepseek_indexer_q,
                     layout.deepseek_indexer_q_scale, query_rows,
                     layout.deepseek_q_lora_rank, row, col) *
                 s.q_gate[col];
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
          const float scaled =
              fminf(fmaxf(query_head[dim] / q_scale, -448.0f), 448.0f);
          const uint8_t q_bits =
              deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
          query_head[dim] =
              nerva::deepseek::f8_e4m3fn_bits_to_f32(q_bits) * q_scale;
        }
      }
    }
    __syncthreads();

    const float softmax_scale = rsqrtf(static_cast<float>(index_head_dim));
    const float head_scale = rsqrtf(static_cast<float>(index_heads));
    for (uint32_t slot = threadIdx.x; slot < compressed_attention_tokens;
         slot += blockDim.x) {
      float score = 0.0f;
      for (uint32_t head = 0; head < index_heads; ++head) {
        float dot = 0.0f;
        const float *query_head = indexer_query + head * index_head_dim;
        for (uint32_t dim = 0; dim < index_head_dim; ++dim) {
          dot += query_head[dim] *
                 deepseek_session_read_indexer_fp8_compressed_kv(
                     deepseek_indexer_kv, deepseek_indexer_kv_offset_bytes,
                     deepseek_indexer_kv_block_count, layout, kv_block_count,
                     kv_block_table, slot, dim);
        }
        score += indexer_weights[head] * softmax_scale * head_scale * dot;
      }
      sparse_topk_score_workspace[slot] = score;
    }
    __syncthreads();

    if (topk_limit <= kDeepSeekSessionMaxSparseTopK &&
        compressed_attention_tokens <= 4096u) {
      const uint32_t sort_size =
          compressed_attention_tokens <= 1024u ? 1024u : 4096u;
      for (uint32_t slot = threadIdx.x; slot < sort_size; slot += blockDim.x) {
        if (slot < compressed_attention_tokens) {
          sort_scores[slot] = sparse_topk_score_workspace[slot];
          sort_slots[slot] = static_cast<int32_t>(slot);
        } else {
          sort_scores[slot] = -INFINITY;
          sort_slots[slot] = -1;
        }
      }
      __syncthreads();

      for (uint32_t width = 2u; width <= sort_size; width <<= 1u) {
        for (uint32_t stride = width >> 1u; stride != 0u; stride >>= 1u) {
          for (uint32_t index = threadIdx.x; index < sort_size;
               index += blockDim.x) {
            const uint32_t other = index ^ stride;
            if (other <= index) {
              continue;
            }
            const bool descending = (index & width) == 0u;
            const float left_score = sort_scores[index];
            const int32_t left_slot = sort_slots[index];
            const float right_score = sort_scores[other];
            const int32_t right_slot = sort_slots[other];
            const bool left_before_right =
                deepseek_session_sparse_score_is_better(
                    left_score, left_slot, right_score, right_slot);
            const bool swap =
                descending ? !left_before_right : left_before_right;
            if (swap) {
              sort_scores[index] = right_score;
              sort_slots[index] = right_slot;
              sort_scores[other] = left_score;
              sort_slots[other] = left_slot;
            }
          }
          __syncthreads();
        }
      }

      if (threadIdx.x == 0) {
        uint32_t selected = 0;
        unsigned long long selection_hash = 0ull;
        for (uint32_t rank = 0; rank < topk_limit; ++rank) {
          if (sort_slots[rank] < 0 || !isfinite(sort_scores[rank])) {
            continue;
          }
          ++selected;
          selection_hash +=
              (static_cast<unsigned long long>(position) + 1ull) *
                  1315423911ull ^
              (static_cast<unsigned long long>(rank) + 1ull) *
                  2654435761ull ^
              (static_cast<unsigned long long>(sort_slots[rank]) + 1ull);
        }
        sort_selected = selected;
        sort_hash = selection_hash;
        *sparse_topk_count = selected;
      }
      __syncthreads();

      for (uint32_t rank = threadIdx.x; rank < sort_selected;
           rank += blockDim.x) {
        sparse_topk_slots[rank] = sort_slots[rank];
      }
      __syncthreads();

      if (threadIdx.x != 0 || sort_selected == 0) {
        return;
      }
      if (deepseek_runtime_counters != nullptr) {
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                deepseek_runtime_counters +
                kDeepSeekRuntimeCounterSparseTopkSelections),
            1ull);
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                deepseek_runtime_counters +
                kDeepSeekRuntimeCounterSparseTopkSlotsSelected),
            static_cast<unsigned long long>(sort_selected));
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                deepseek_runtime_counters +
                kDeepSeekRuntimeCounterSparseTopkCandidatesScored),
            static_cast<unsigned long long>(compressed_attention_tokens));
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                deepseek_runtime_counters +
                kDeepSeekRuntimeCounterSparseTopkSelectionHash),
            sort_hash);
      }
      return;
    }

    unsigned long long selection_hash = 0ull;
    uint32_t selected = 0;
    for (uint32_t rank = 0; rank < topk_limit; ++rank) {
      float thread_best_score = -INFINITY;
      int32_t thread_best_slot = -1;
      for (uint32_t slot = threadIdx.x; slot < compressed_attention_tokens;
           slot += blockDim.x) {
        const float score = sparse_topk_score_workspace[slot];
        const int32_t slot_i32 = static_cast<int32_t>(slot);
        if (deepseek_session_sparse_score_is_better(
                score, slot_i32, thread_best_score, thread_best_slot)) {
          thread_best_score = score;
          thread_best_slot = slot_i32;
        }
      }
      reduce_scores[threadIdx.x] = thread_best_score;
      reduce_slots[threadIdx.x] = thread_best_slot;
      __syncthreads();
      for (uint32_t stride = blockDim.x / 2u; stride != 0; stride >>= 1u) {
        if (threadIdx.x < stride) {
          const float other_score = reduce_scores[threadIdx.x + stride];
          const int32_t other_slot = reduce_slots[threadIdx.x + stride];
          if (deepseek_session_sparse_score_is_better(
                  other_score, other_slot, reduce_scores[threadIdx.x],
                  reduce_slots[threadIdx.x])) {
            reduce_scores[threadIdx.x] = other_score;
            reduce_slots[threadIdx.x] = other_slot;
          }
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        const int32_t best_slot = reduce_slots[0];
        sparse_topk_slots[rank] = best_slot;
        if (best_slot >= 0) {
          sparse_topk_score_workspace[best_slot] = -INFINITY;
          ++selected;
          selection_hash +=
              (static_cast<unsigned long long>(position) + 1ull) *
                  1315423911ull ^
              (static_cast<unsigned long long>(rank) + 1ull) *
                  2654435761ull ^
              (static_cast<unsigned long long>(best_slot) + 1ull);
        }
      }
      __syncthreads();
    }
    if (threadIdx.x != 0 || selected == 0) {
      return;
    }
    *sparse_topk_count = selected;
    if (deepseek_runtime_counters != nullptr) {
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
          static_cast<unsigned long long>(compressed_attention_tokens));
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterSparseTopkSelectionHash),
          selection_hash);
    }
    return;
  }

  if (threadIdx.x != 0) {
    return;
  }
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
      const uint32_t left = head_start + qk_nope + offset * 2u;
      const uint32_t right = left + 1u;
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
