__global__ void hf_deepseek_v32_indexer_kv_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t *step_cursor, uint32_t max_steps,
    float rope_theta, const uint16_t *projection_input,
    uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint64_t *deepseek_runtime_counters) {
  if (blockIdx.x != 0 ||
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
  const uint32_t logical_block = position / kDeepSeekV32IndexerKvBlockTokens;
  uint32_t physical_block = 0;
  if (!deepseek_v32_packed_physical_block(
          kv_block_table, kv_block_count, deepseek_indexer_kv_block_count,
          logical_block, &physical_block)) {
    return;
  }

  const uint32_t head_dim = layout.deepseek_index_head_dim;
  __shared__ float values[kDeepSeekSessionMaxCompressHeadSize];
  __shared__ float mean;
  __shared__ float inv_std;
  __shared__ float scale;
  for (uint32_t row = 0; row < head_dim; ++row) {
    float sum = 0.0f;
    for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
      sum += deepseek_fp8_scaled_weight(
                 arena, layout.deepseek_indexer_k,
                 layout.deepseek_indexer_k_scale, head_dim, hidden, row,
                 col) *
             encoded_to_f32(projection_input[col], dtype);
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
      values[row] = sum;
    }
  }
  __syncthreads();

  float mean_sum = 0.0f;
  for (uint32_t row = threadIdx.x; row < head_dim; row += blockDim.x) {
    mean_sum += values[row];
  }
  mean_sum = block_sum(mean_sum);
  if (threadIdx.x == 0) {
    mean = mean_sum / static_cast<float>(head_dim);
  }
  __syncthreads();

  float variance_sum = 0.0f;
  for (uint32_t row = threadIdx.x; row < head_dim; row += blockDim.x) {
    const float centered = values[row] - mean;
    variance_sum += centered * centered;
  }
  variance_sum = block_sum(variance_sum);
  if (threadIdx.x == 0) {
    inv_std =
        rsqrtf(variance_sum / static_cast<float>(head_dim) + 1.0e-6f);
  }
  __syncthreads();

  for (uint32_t row = threadIdx.x; row < head_dim; row += blockDim.x) {
    const float weight = f32_from_u16_slots(arena + layout.deepseek_indexer_k_norm,
                                            row);
    const float bias = f32_from_u16_slots(
        arena + layout.deepseek_indexer_k_norm_bias, row);
    values[row] = (values[row] - mean) * inv_std * weight + bias;
  }
  __syncthreads();

  const uint32_t rope_dim =
      layout.deepseek_qk_rope_head_dim <= head_dim
          ? layout.deepseek_qk_rope_head_dim
          : 0u;
  const uint32_t rope_half = rope_dim / 2u;
  for (uint32_t offset = threadIdx.x; offset < rope_half; offset += blockDim.x) {
    const uint32_t left = offset;
    const uint32_t right = offset + rope_half;
    const float left_value = values[left];
    const float right_value = values[right];
    values[left] = deepseek_rope_value_serial(
        left_value, right_value, offset, rope_dim, position, rope_theta,
        false, layout);
    values[right] = deepseek_rope_value_serial(
        left_value, right_value, offset, rope_dim, position, rope_theta,
        true, layout);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float absmax = 0.0f;
    for (uint32_t dim = 0; dim < head_dim; ++dim) {
      values[dim] = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(values[dim]));
      absmax = fmaxf(absmax, fabsf(values[dim]));
    }
    scale = exp2f(ceilf(log2f(fmaxf(absmax, 1.0e-4f) / 448.0f)));
  }
  __syncthreads();

  const uint32_t scale_bytes =
      ((head_dim + 127u) / 128u) * sizeof(float);
  const uint64_t page_bytes =
      static_cast<uint64_t>(kDeepSeekV32IndexerKvBlockTokens) *
      (static_cast<uint64_t>(head_dim) + scale_bytes);
  const uint32_t block_offset =
      position % kDeepSeekV32IndexerKvBlockTokens;
  uint8_t *block_ptr = deepseek_indexer_kv +
                       deepseek_indexer_kv_offset_bytes +
                       static_cast<uint64_t>(physical_block) * page_bytes;
  const uint32_t tile_block_id =
      block_offset / kDeepSeekV32IndexerKvTileTokens;
  const uint32_t tile_block_offset =
      block_offset % kDeepSeekV32IndexerKvTileTokens;
  for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
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
  for (uint32_t scale_index = threadIdx.x;
       scale_index < scale_bytes / sizeof(float);
       scale_index += blockDim.x) {
    reinterpret_cast<float *>(scale_ptr)[scale_index] = scale;
  }
  if (threadIdx.x == 0 && deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterIndexerKvWrites),
        1ull);
  }
}

__global__ void hf_deepseek_v32_indexer_kv_encode_tokens_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t chunk_start, uint32_t chunk_tokens,
    uint32_t max_steps, float rope_theta, const uint16_t *projection_input,
    uint32_t projection_input_stride, uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint64_t *deepseek_runtime_counters) {
  const uint32_t local_token = blockIdx.x;
  if (local_token >= chunk_tokens || projection_input_stride < hidden) {
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

  const uint32_t position = chunk_start + local_token;
  if (position >= max_steps) {
    return;
  }
  const uint32_t logical_block = position / kDeepSeekV32IndexerKvBlockTokens;
  uint32_t physical_block = 0;
  if (!deepseek_v32_packed_physical_block(
          kv_block_table, kv_block_count, deepseek_indexer_kv_block_count,
          logical_block, &physical_block)) {
    return;
  }

  const uint16_t *token_projection =
      projection_input +
      static_cast<uint64_t>(local_token) * projection_input_stride;
  const uint32_t head_dim = layout.deepseek_index_head_dim;
  __shared__ float values[kDeepSeekSessionMaxCompressHeadSize];
  __shared__ float mean;
  __shared__ float inv_std;
  __shared__ float scale;
  for (uint32_t row = 0; row < head_dim; ++row) {
    float sum = 0.0f;
    for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
      sum += deepseek_fp8_scaled_weight(
                 arena, layout.deepseek_indexer_k,
                 layout.deepseek_indexer_k_scale, head_dim, hidden, row,
                 col) *
             encoded_to_f32(token_projection[col], dtype);
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
      values[row] = sum;
    }
  }
  __syncthreads();

  float mean_sum = 0.0f;
  for (uint32_t row = threadIdx.x; row < head_dim; row += blockDim.x) {
    mean_sum += values[row];
  }
  mean_sum = block_sum(mean_sum);
  if (threadIdx.x == 0) {
    mean = mean_sum / static_cast<float>(head_dim);
  }
  __syncthreads();

  float variance_sum = 0.0f;
  for (uint32_t row = threadIdx.x; row < head_dim; row += blockDim.x) {
    const float centered = values[row] - mean;
    variance_sum += centered * centered;
  }
  variance_sum = block_sum(variance_sum);
  if (threadIdx.x == 0) {
    inv_std =
        rsqrtf(variance_sum / static_cast<float>(head_dim) + 1.0e-6f);
  }
  __syncthreads();

  for (uint32_t row = threadIdx.x; row < head_dim; row += blockDim.x) {
    const float weight =
        f32_from_u16_slots(arena + layout.deepseek_indexer_k_norm, row);
    const float bias = f32_from_u16_slots(
        arena + layout.deepseek_indexer_k_norm_bias, row);
    values[row] = (values[row] - mean) * inv_std * weight + bias;
  }
  __syncthreads();

  const uint32_t rope_dim =
      layout.deepseek_qk_rope_head_dim <= head_dim
          ? layout.deepseek_qk_rope_head_dim
          : 0u;
  const uint32_t rope_half = rope_dim / 2u;
  for (uint32_t offset = threadIdx.x; offset < rope_half; offset += blockDim.x) {
    const uint32_t left = offset;
    const uint32_t right = offset + rope_half;
    const float left_value = values[left];
    const float right_value = values[right];
    values[left] = deepseek_rope_value_serial(
        left_value, right_value, offset, rope_dim, position, rope_theta,
        false, layout);
    values[right] = deepseek_rope_value_serial(
        left_value, right_value, offset, rope_dim, position, rope_theta,
        true, layout);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float absmax = 0.0f;
    for (uint32_t dim = 0; dim < head_dim; ++dim) {
      values[dim] = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(values[dim]));
      absmax = fmaxf(absmax, fabsf(values[dim]));
    }
    scale = exp2f(ceilf(log2f(fmaxf(absmax, 1.0e-4f) / 448.0f)));
  }
  __syncthreads();

  const uint32_t scale_bytes =
      ((head_dim + 127u) / 128u) * sizeof(float);
  const uint64_t page_bytes =
      static_cast<uint64_t>(kDeepSeekV32IndexerKvBlockTokens) *
      (static_cast<uint64_t>(head_dim) + scale_bytes);
  const uint32_t block_offset =
      position % kDeepSeekV32IndexerKvBlockTokens;
  uint8_t *block_ptr = deepseek_indexer_kv +
                       deepseek_indexer_kv_offset_bytes +
                       static_cast<uint64_t>(physical_block) * page_bytes;
  const uint32_t tile_block_id =
      block_offset / kDeepSeekV32IndexerKvTileTokens;
  const uint32_t tile_block_offset =
      block_offset % kDeepSeekV32IndexerKvTileTokens;
  for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
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
  for (uint32_t scale_index = threadIdx.x;
       scale_index < scale_bytes / sizeof(float);
       scale_index += blockDim.x) {
    reinterpret_cast<float *>(scale_ptr)[scale_index] = scale;
  }
  if (threadIdx.x == 0 && deepseek_runtime_counters != nullptr) {
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
  if ((step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  const uint32_t head = blockIdx.x;
  if (head >= layout.deepseek_index_n_heads) {
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
  float sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    sum += encoded_to_f32(arena[layout.deepseek_indexer_weights +
                                static_cast<uint64_t>(head) * hidden + col],
                          kDTypeBF16) *
           encoded_to_f32(projection_input[col], dtype);
  }
  sum = block_sum(sum);
  if (threadIdx.x == 0) {
    weights[head] = sum;
  }
}

__global__ void hf_deepseek_v32_indexer_weight_state_tokens_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t chunk_start, uint32_t chunk_tokens,
    uint32_t max_steps, const uint16_t *projection_input,
    uint32_t projection_input_stride, uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes) {
  const uint32_t head = blockIdx.x;
  const uint32_t local_token = blockIdx.y;
  if (head >= layout.deepseek_index_n_heads || local_token >= chunk_tokens ||
      projection_input_stride < hidden) {
    return;
  }
  if (arena == nullptr || projection_input == nullptr ||
      deepseek_indexer_state == nullptr ||
      !deepseek_v32_indexer_query_state_supported(layout)) {
    return;
  }
  const uint32_t position = chunk_start + local_token;
  if (position >= max_steps) {
    return;
  }
  const uint16_t *token_projection =
      projection_input +
      static_cast<uint64_t>(local_token) * projection_input_stride;
  const uint64_t token_bytes =
      deepseek_v32_indexer_query_state_token_bytes(layout);
  uint8_t *token_ptr = deepseek_indexer_state +
                       deepseek_indexer_state_offset_bytes +
                       static_cast<uint64_t>(position) * token_bytes;
  auto *weights = reinterpret_cast<float *>(
      token_ptr +
      deepseek_v32_indexer_query_state_weights_offset_bytes(layout));
  float sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    sum += encoded_to_f32(arena[layout.deepseek_indexer_weights +
                                static_cast<uint64_t>(head) * hidden + col],
                          kDTypeBF16) *
           encoded_to_f32(token_projection[col], dtype);
  }
  sum = block_sum(sum);
  if (threadIdx.x == 0) {
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
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  const uint32_t head = blockIdx.x;
  if (head >= layout.deepseek_index_n_heads) {
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

  __shared__ float query_head[kDeepSeekSessionMaxCompressHeadSize];
  __shared__ float q_scale;
  for (uint32_t dim = threadIdx.x; dim < index_head_dim; dim += blockDim.x) {
    const uint32_t row = head * index_head_dim + dim;
    float sum = 0.0f;
    for (uint32_t col = 0; col < q_lora_rank; ++col) {
      sum += deepseek_fp8_scaled_weight(
                 arena, layout.deepseek_indexer_q,
                 layout.deepseek_indexer_q_scale, query_rows, q_lora_rank,
                 row, col) *
             encoded_to_f32(qr_norm[col], dtype);
    }
    query_head[dim] = sum;
  }
  __syncthreads();

  const uint32_t rope_dim =
      layout.deepseek_qk_rope_head_dim <= index_head_dim
          ? layout.deepseek_qk_rope_head_dim
          : 0u;
  const uint32_t rope_half = rope_dim / 2u;
  const float softmax_scale = rsqrtf(static_cast<float>(index_head_dim));
  const float head_scale = rsqrtf(static_cast<float>(index_heads));
  for (uint32_t offset = threadIdx.x; offset < rope_half; offset += blockDim.x) {
    const uint32_t left = offset;
    const uint32_t right = offset + rope_half;
    const float left_value = query_head[left];
    const float right_value = query_head[right];
    query_head[left] = deepseek_rope_value_serial(
        left_value, right_value, offset, rope_dim, position, rope_theta,
        false, layout);
    query_head[right] = deepseek_rope_value_serial(
        left_value, right_value, offset, rope_dim, position, rope_theta,
        true, layout);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float absmax = 0.0f;
    for (uint32_t dim = 0; dim < index_head_dim; ++dim) {
      query_head[dim] = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(query_head[dim]));
      absmax = fmaxf(absmax, fabsf(query_head[dim]));
    }
    q_scale = exp2f(ceilf(log2f(fmaxf(absmax, 1.0e-4f) / 448.0f)));
    q_scales[head] = q_scale;
    weights[head] *= q_scale * softmax_scale * head_scale;
  }
  __syncthreads();

  for (uint32_t dim = threadIdx.x; dim < index_head_dim; dim += blockDim.x) {
    const float scaled =
        fminf(fmaxf(query_head[dim] / q_scale, -448.0f), 448.0f);
    q_fp8[static_cast<uint64_t>(head) * index_head_dim + dim] =
        deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
  }

  if (blockIdx.x == 0 && threadIdx.x == 0 && deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterIndexerStateWrites),
        1ull);
  }
}

__global__ void hf_deepseek_v32_indexer_query_state_projected_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t *step_cursor,
    uint32_t max_steps, float rope_theta, const float *projected_query,
    uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    uint64_t *deepseek_runtime_counters) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  const uint32_t head = blockIdx.x;
  if (head >= layout.deepseek_index_n_heads) {
    return;
  }
  if (arena == nullptr || projected_query == nullptr ||
      deepseek_indexer_state == nullptr ||
      !deepseek_v32_indexer_query_state_supported(layout)) {
    return;
  }

  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t index_heads = layout.deepseek_index_n_heads;
  const uint32_t index_head_dim = layout.deepseek_index_head_dim;
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

  __shared__ float query_head[kDeepSeekSessionMaxCompressHeadSize];
  __shared__ float q_scale;
  for (uint32_t dim = threadIdx.x; dim < index_head_dim; dim += blockDim.x) {
    query_head[dim] =
        projected_query[static_cast<uint64_t>(head) * index_head_dim + dim];
  }
  __syncthreads();

  const uint32_t rope_dim =
      layout.deepseek_qk_rope_head_dim <= index_head_dim
          ? layout.deepseek_qk_rope_head_dim
          : 0u;
  const uint32_t rope_half = rope_dim / 2u;
  const float softmax_scale = rsqrtf(static_cast<float>(index_head_dim));
  const float head_scale = rsqrtf(static_cast<float>(index_heads));
  for (uint32_t offset = threadIdx.x; offset < rope_half; offset += blockDim.x) {
    const uint32_t left = offset;
    const uint32_t right = offset + rope_half;
    const float left_value = query_head[left];
    const float right_value = query_head[right];
    query_head[left] = deepseek_rope_value_serial(
        left_value, right_value, offset, rope_dim, position, rope_theta,
        false, layout);
    query_head[right] = deepseek_rope_value_serial(
        left_value, right_value, offset, rope_dim, position, rope_theta,
        true, layout);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float absmax = 0.0f;
    for (uint32_t dim = 0; dim < index_head_dim; ++dim) {
      query_head[dim] = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(query_head[dim]));
      absmax = fmaxf(absmax, fabsf(query_head[dim]));
    }
    q_scale = exp2f(ceilf(log2f(fmaxf(absmax, 1.0e-4f) / 448.0f)));
    q_scales[head] = q_scale;
    weights[head] *= q_scale * softmax_scale * head_scale;
  }
  __syncthreads();

  for (uint32_t dim = threadIdx.x; dim < index_head_dim; dim += blockDim.x) {
    const float scaled =
        fminf(fmaxf(query_head[dim] / q_scale, -448.0f), 448.0f);
    q_fp8[static_cast<uint64_t>(head) * index_head_dim + dim] =
        deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
  }

  if (blockIdx.x == 0 && threadIdx.x == 0 &&
      deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterIndexerStateWrites),
        1ull);
  }
}

__global__ void hf_deepseek_v32_indexer_query_state_tokens_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t q_lora_rank, uint32_t chunk_start, uint32_t chunk_tokens,
    uint32_t max_steps, float rope_theta, const uint16_t *qr_norm,
    uint32_t qr_norm_stride, uint8_t *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    uint64_t *deepseek_runtime_counters) {
  const uint32_t head = blockIdx.x;
  const uint32_t local_token = blockIdx.y;
  if (head >= layout.deepseek_index_n_heads || local_token >= chunk_tokens ||
      qr_norm_stride < q_lora_rank) {
    return;
  }
  if (arena == nullptr || qr_norm == nullptr ||
      deepseek_indexer_state == nullptr ||
      !deepseek_v32_indexer_query_state_supported(layout) ||
      q_lora_rank != layout.deepseek_q_lora_rank) {
    return;
  }
  const uint32_t position = chunk_start + local_token;
  if (position >= max_steps) {
    return;
  }

  const uint16_t *token_qr_norm =
      qr_norm + static_cast<uint64_t>(local_token) * qr_norm_stride;
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

  __shared__ float query_head[kDeepSeekSessionMaxCompressHeadSize];
  __shared__ float q_scale;
  for (uint32_t dim = threadIdx.x; dim < index_head_dim; dim += blockDim.x) {
    const uint32_t row = head * index_head_dim + dim;
    float sum = 0.0f;
    for (uint32_t col = 0; col < q_lora_rank; ++col) {
      sum += deepseek_fp8_scaled_weight(
                 arena, layout.deepseek_indexer_q,
                 layout.deepseek_indexer_q_scale, query_rows, q_lora_rank,
                 row, col) *
             encoded_to_f32(token_qr_norm[col], dtype);
    }
    query_head[dim] = sum;
  }
  __syncthreads();

  const uint32_t rope_dim =
      layout.deepseek_qk_rope_head_dim <= index_head_dim
          ? layout.deepseek_qk_rope_head_dim
          : 0u;
  const uint32_t rope_half = rope_dim / 2u;
  const float softmax_scale = rsqrtf(static_cast<float>(index_head_dim));
  const float head_scale = rsqrtf(static_cast<float>(index_heads));
  for (uint32_t offset = threadIdx.x; offset < rope_half; offset += blockDim.x) {
    const uint32_t left = offset;
    const uint32_t right = offset + rope_half;
    const float left_value = query_head[left];
    const float right_value = query_head[right];
    query_head[left] = deepseek_rope_value_serial(
        left_value, right_value, offset, rope_dim, position, rope_theta,
        false, layout);
    query_head[right] = deepseek_rope_value_serial(
        left_value, right_value, offset, rope_dim, position, rope_theta,
        true, layout);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float absmax = 0.0f;
    for (uint32_t dim = 0; dim < index_head_dim; ++dim) {
      query_head[dim] = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(query_head[dim]));
      absmax = fmaxf(absmax, fabsf(query_head[dim]));
    }
    q_scale = exp2f(ceilf(log2f(fmaxf(absmax, 1.0e-4f) / 448.0f)));
    q_scales[head] = q_scale;
    weights[head] *= q_scale * softmax_scale * head_scale;
  }
  __syncthreads();

  for (uint32_t dim = threadIdx.x; dim < index_head_dim; dim += blockDim.x) {
    const float scaled =
        fminf(fmaxf(query_head[dim] / q_scale, -448.0f), 448.0f);
    q_fp8[static_cast<uint64_t>(head) * index_head_dim + dim] =
        deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
  }

  if (head == 0 && threadIdx.x == 0 && deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterIndexerStateWrites),
        1ull);
  }
}

__device__ float deepseek_session_read_v32_indexer_kv_raw(
    const uint8_t *kv_cache, uint64_t kv_offset_bytes, uint32_t block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    const SequenceLayerLayout &layout, uint32_t position, uint32_t dim,
    float *scale_out) {
  if (scale_out != nullptr) {
    *scale_out = 0.0f;
  }
  const uint32_t head_dim = layout.deepseek_index_head_dim;
  if (kv_cache == nullptr || head_dim == 0 || dim >= head_dim) {
    return 0.0f;
  }
  const uint32_t logical_block = position / kDeepSeekV32IndexerKvBlockTokens;
  uint32_t physical_block = 0;
  if (!deepseek_v32_packed_physical_block(kv_block_table, kv_block_count,
                                          block_count, logical_block,
                                          &physical_block)) {
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
                             static_cast<uint64_t>(physical_block) * page_bytes;
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
    uint32_t kv_block_count, const uint32_t *kv_block_table,
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
              deepseek_indexer_kv_block_count, kv_block_count, kv_block_table,
              layout, slot, dim, &k_scale);
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
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    int32_t *sparse_topk_slots, uint32_t *sparse_topk_count,
    uint64_t *deepseek_runtime_counters) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (sparse_topk_count != nullptr) {
    *sparse_topk_count = 0;
  }
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  if (sparse_topk_slots == nullptr || sparse_topk_count == nullptr) {
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
      kv_block_count, kv_block_table,
      topk_slots, topk_scores, &candidates_scored, &selection_hash);
  if (selected == 0) {
    return;
  }
  *sparse_topk_count = selected;
  for (uint32_t rank = 0; rank < selected; ++rank) {
    sparse_topk_slots[rank] = topk_slots[rank];
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
