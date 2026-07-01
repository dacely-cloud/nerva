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

__global__ void hf_deepseek_v4_compressed_kv_write_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t head_dim,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps,
    float rope_theta, float *deepseek_compressor_state,
    uint64_t deepseek_compressor_state_offset_bytes,
    uint8_t *deepseek_compressed_kv,
    uint64_t deepseek_compressed_kv_offset_bytes,
    uint32_t deepseek_compressed_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint64_t *deepseek_runtime_counters) {
  if ((step_cursor != nullptr && *step_cursor >= max_steps) ||
      arena == nullptr || deepseek_compressor_state == nullptr ||
      deepseek_compressed_kv == nullptr || head_dim == 0 ||
      head_dim > kDeepSeekSessionMaxCompressHeadSize ||
      layout.deepseek_compress_ratio <= 1 ||
      layout.deepseek_compressor_norm == kMissingOffset ||
      (layout.deepseek_mode != kDeepSeekModeV4Compressed &&
       layout.deepseek_mode != kDeepSeekModeV4CompressedIndexer)) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t compress_ratio = layout.deepseek_compress_ratio;
  if ((position + 1u) % compress_ratio != 0) {
    return;
  }
  const uint32_t nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t rope = layout.deepseek_qk_rope_head_dim;
  if (nope + rope != head_dim) {
    return;
  }
  const uint32_t coff = compress_ratio == 4 ? 2u : 1u;
  const uint32_t state_width = coff * head_dim;
  const uint32_t compressed_slot = position / compress_ratio;
  const uint32_t packed_block_tokens =
      deepseek_v4_packed_kv_block_tokens(compress_ratio);
  const uint32_t logical_compressed_block =
      compressed_slot / packed_block_tokens;
  uint32_t compressed_block = 0;
  if (!deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, deepseek_compressed_kv_block_count,
          logical_compressed_block, packed_block_tokens * compress_ratio,
          &compressed_block)) {
    return;
  }

  __shared__ float compressed[kDeepSeekSessionMaxCompressHeadSize];
  __shared__ float rrms;
  __shared__ float quant_scales[16];
  const uint32_t overlap = compress_ratio == 4 ? 1u : 0u;
  const uint32_t window_tokens = (1u + overlap) * compress_ratio;
  const int64_t start =
      static_cast<int64_t>(position) - static_cast<int64_t>(window_tokens) + 1ll;
  for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float max_score = -INFINITY;
    for (uint32_t window = 0; window < window_tokens; ++window) {
      const int64_t pos = start + static_cast<int64_t>(window);
      if (pos < 0) {
        continue;
      }
      const float score = deepseek_session_compressed_state_value(
          deepseek_compressor_state, deepseek_compressor_state_offset_bytes,
          state_width, head_dim, kv_block_count, kv_block_table,
          static_cast<uint32_t>(pos), dim, compress_ratio, window, true);
      max_score = fmaxf(max_score, score);
    }
    float weighted = 0.0f;
    float denom = 0.0f;
    for (uint32_t window = 0; window < window_tokens; ++window) {
      const int64_t pos = start + static_cast<int64_t>(window);
      if (pos < 0) {
        continue;
      }
      const uint32_t pos_u32 = static_cast<uint32_t>(pos);
      const float score = deepseek_session_compressed_state_value(
          deepseek_compressor_state, deepseek_compressor_state_offset_bytes,
          state_width, head_dim, kv_block_count, kv_block_table, pos_u32, dim,
          compress_ratio, window, true);
      const float weight = expf(score - max_score);
      weighted += deepseek_session_compressed_state_value(
                      deepseek_compressor_state,
                      deepseek_compressor_state_offset_bytes, state_width,
                      head_dim, kv_block_count, kv_block_table, pos_u32, dim,
                      compress_ratio, window, false) *
                  weight;
      denom += weight;
    }
    compressed[dim] = denom > 0.0f ? weighted / denom : 0.0f;
  }
  __syncthreads();

  float variance = 0.0f;
  for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    variance += compressed[dim] * compressed[dim];
  }
  variance = block_sum(variance);
  if (threadIdx.x == 0) {
    rrms = rsqrtf(variance / static_cast<float>(head_dim) + rms_eps);
  }
  __syncthreads();

  const uint32_t token_stride = nope + rope * 2u;
  const uint32_t scale_dim = nope / 64u + 1u;
  const uint32_t kv_pos = compressed_slot % packed_block_tokens;
  const uint64_t block_stride = deepseek_v4_round_up_u64(
      static_cast<uint64_t>(packed_block_tokens) *
          static_cast<uint64_t>(token_stride + scale_dim),
      kDeepSeekV4PackedKvAlignmentBytes);
  uint8_t *block_ptr =
      deepseek_compressed_kv + deepseek_compressed_kv_offset_bytes +
      static_cast<uint64_t>(compressed_block) * block_stride;
  uint8_t *data_ptr = block_ptr + static_cast<uint64_t>(kv_pos) * token_stride;
  uint8_t *scale_ptr =
      block_ptr + static_cast<uint64_t>(packed_block_tokens) * token_stride +
      static_cast<uint64_t>(kv_pos) * scale_dim;
  const uint16_t *norm_weight = arena + layout.deepseek_compressor_norm;
  if (threadIdx.x == 0) {
    constexpr uint32_t quant_block = 64u;
    for (uint32_t scale_index = 0; scale_index < scale_dim; ++scale_index) {
      const uint32_t start_dim = scale_index * quant_block;
      const uint32_t end_dim =
          start_dim + quant_block < nope ? start_dim + quant_block : nope;
      float absmax = 0.0f;
      for (uint32_t dim = start_dim; dim < end_dim; ++dim) {
        const float normed =
            compressed[dim] * rrms *
            encoded_to_f32(norm_weight[dim], kDTypeBF16);
        const float quant_input = deepseek_session_bf16_bits_to_f32(
            deepseek_session_f32_to_bf16_bits(normed));
        absmax = fmaxf(absmax, fabsf(quant_input));
      }
      const float raw = fmaxf(absmax, 1.0e-4f) / 448.0f;
      const float scale = exp2f(ceilf(log2f(raw)));
      quant_scales[scale_index] = scale;
      scale_ptr[scale_index] =
          start_dim < nope ? deepseek_session_encode_e8m0_scale(scale) : 0u;
    }
  }
  __syncthreads();

  for (uint32_t dim = threadIdx.x; dim < nope; dim += blockDim.x) {
    const float normed =
        compressed[dim] * rrms * encoded_to_f32(norm_weight[dim], kDTypeBF16);
    const float quant_input = deepseek_session_bf16_bits_to_f32(
        deepseek_session_f32_to_bf16_bits(normed));
    const float scale = quant_scales[dim / 64u];
    const float scaled = fminf(fmaxf(quant_input / scale, -448.0f), 448.0f);
    data_ptr[dim] = deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
  }
  const uint32_t compressed_pos = compressed_slot * compress_ratio;
  for (uint32_t dim = nope + threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const uint32_t rope_local = dim - nope;
    const uint32_t pair = rope_local / 2u;
    const uint32_t even_dim = nope + pair * 2u;
    const uint32_t odd_dim = even_dim + 1u;
    float rotated =
        compressed[dim] * rrms * encoded_to_f32(norm_weight[dim], kDTypeBF16);
    if (rope_theta > 0.0f && rope >= 2u && odd_dim < head_dim) {
      const float even = compressed[even_dim] * rrms *
                         encoded_to_f32(norm_weight[even_dim], kDTypeBF16);
      const float odd = compressed[odd_dim] * rrms *
                        encoded_to_f32(norm_weight[odd_dim], kDTypeBF16);
      const float angle =
          static_cast<float>(compressed_pos) *
          deepseek_rope_inv_freq(layout, pair, rope, rope_theta);
      float sin_value = 0.0f;
      float cos_value = 0.0f;
      sincosf(angle, &sin_value, &cos_value);
      const float magnitude = deepseek_rope_magnitude(layout);
      rotated = magnitude *
                ((rope_local & 1u) == 0u
                     ? even * cos_value - odd * sin_value
                     : odd * cos_value + even * sin_value);
    }
    const uint16_t bits = deepseek_session_f32_to_bf16_bits(rotated);
    data_ptr[nope + rope_local * 2u] = static_cast<uint8_t>(bits & 0xffu);
    data_ptr[nope + rope_local * 2u + 1u] =
        static_cast<uint8_t>(bits >> 8u);
  }
  if (threadIdx.x == 0 && deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterCompressedKvWrites),
        1ull);
  }
}

__device__ float deepseek_v4_indexer_rotated_value(
    const float *compressed, const uint16_t *norm_weight, uint32_t head_dim,
    uint32_t rope, uint32_t dim, uint32_t compressed_pos, float rrms,
    float rope_theta, const SequenceLayerLayout &layout) {
  const uint32_t nope = head_dim - rope;
  float rotated =
      compressed[dim] * rrms * encoded_to_f32(norm_weight[dim], kDTypeBF16);
  if (rope_theta <= 0.0f || rope < 2u || dim < nope) {
    return rotated;
  }
  const uint32_t rope_local = dim - nope;
  const uint32_t pair = rope_local / 2u;
  const uint32_t even_dim = nope + pair * 2u;
  const uint32_t odd_dim = even_dim + 1u;
  if (odd_dim >= head_dim) {
    return rotated;
  }
  const float even =
      compressed[even_dim] * rrms *
      encoded_to_f32(norm_weight[even_dim], kDTypeBF16);
  const float odd =
      compressed[odd_dim] * rrms *
      encoded_to_f32(norm_weight[odd_dim], kDTypeBF16);
  const float angle =
      static_cast<float>(compressed_pos) *
      deepseek_rope_inv_freq(layout, pair, rope, rope_theta);
  float sin_value = 0.0f;
  float cos_value = 0.0f;
  sincosf(angle, &sin_value, &cos_value);
  const float magnitude = deepseek_rope_magnitude(layout);
  return magnitude *
         ((rope_local & 1u) == 0u ? even * cos_value - odd * sin_value
                                  : odd * cos_value + even * sin_value);
}

__global__ void hf_deepseek_v4_indexer_kv_write_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float rope_theta,
    float *deepseek_indexer_state,
    uint64_t deepseek_indexer_state_offset_bytes,
    uint8_t *deepseek_indexer_kv,
    uint64_t deepseek_indexer_kv_offset_bytes,
    uint32_t deepseek_indexer_kv_block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint64_t *deepseek_runtime_counters) {
  const uint32_t head_dim = layout.deepseek_index_head_dim;
  if ((step_cursor != nullptr && *step_cursor >= max_steps) ||
      arena == nullptr || deepseek_indexer_state == nullptr ||
      deepseek_indexer_kv == nullptr || head_dim == 0 ||
      head_dim > kDeepSeekSessionMaxCompressHeadSize ||
      layout.deepseek_mode != kDeepSeekModeV4CompressedIndexer ||
      layout.deepseek_compress_ratio <= 1 ||
      layout.deepseek_indexer_compressor_norm == kMissingOffset) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t compress_ratio = layout.deepseek_compress_ratio;
  if ((position + 1u) % compress_ratio != 0) {
    return;
  }
  const uint32_t coff = compress_ratio == 4 ? 2u : 1u;
  const uint32_t state_width = coff * head_dim;
  const uint32_t compressed_slot = position / compress_ratio;
  const uint32_t packed_block_tokens =
      deepseek_v4_packed_kv_block_tokens(compress_ratio);
  const uint32_t logical_compressed_block =
      compressed_slot / packed_block_tokens;
  uint32_t compressed_block = 0;
  if (state_width == 0 ||
      !deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, deepseek_indexer_kv_block_count,
          logical_compressed_block, packed_block_tokens * compress_ratio,
          &compressed_block)) {
    return;
  }

  __shared__ float compressed[kDeepSeekSessionMaxCompressHeadSize];
  __shared__ float rrms;
  __shared__ float quant_scales[16];
  const uint32_t overlap = compress_ratio == 4 ? 1u : 0u;
  const uint32_t window_tokens = (1u + overlap) * compress_ratio;
  const int64_t start =
      static_cast<int64_t>(position) - static_cast<int64_t>(window_tokens) + 1ll;
  for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float max_score = -INFINITY;
    for (uint32_t window = 0; window < window_tokens; ++window) {
      const int64_t pos = start + static_cast<int64_t>(window);
      if (pos < 0) {
        continue;
      }
      const float score = deepseek_session_compressed_state_value(
          deepseek_indexer_state, deepseek_indexer_state_offset_bytes,
          state_width, head_dim, kv_block_count, kv_block_table,
          static_cast<uint32_t>(pos), dim, compress_ratio, window, true);
      max_score = fmaxf(max_score, score);
    }
    float weighted = 0.0f;
    float denom = 0.0f;
    for (uint32_t window = 0; window < window_tokens; ++window) {
      const int64_t pos = start + static_cast<int64_t>(window);
      if (pos < 0) {
        continue;
      }
      const uint32_t pos_u32 = static_cast<uint32_t>(pos);
      const float score = deepseek_session_compressed_state_value(
          deepseek_indexer_state, deepseek_indexer_state_offset_bytes,
          state_width, head_dim, kv_block_count, kv_block_table, pos_u32, dim,
          compress_ratio, window, true);
      const float weight = expf(score - max_score);
      weighted += deepseek_session_compressed_state_value(
                      deepseek_indexer_state,
                      deepseek_indexer_state_offset_bytes, state_width,
                      head_dim, kv_block_count, kv_block_table, pos_u32, dim,
                      compress_ratio, window, false) *
                  weight;
      denom += weight;
    }
    compressed[dim] = denom > 0.0f ? weighted / denom : 0.0f;
  }
  __syncthreads();

  float variance = 0.0f;
  for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    variance += compressed[dim] * compressed[dim];
  }
  variance = block_sum(variance);
  if (threadIdx.x == 0) {
    rrms = rsqrtf(variance / static_cast<float>(head_dim) + rms_eps);
  }
  __syncthreads();

  const uint32_t scale_count = deepseek_device_scale_dim(head_dim);
  const uint32_t scale_dim = scale_count * sizeof(float);
  const uint32_t kv_pos = compressed_slot % packed_block_tokens;
  const uint64_t block_stride = deepseek_v4_round_up_u64(
      static_cast<uint64_t>(packed_block_tokens) *
          static_cast<uint64_t>(head_dim + scale_dim),
      kDeepSeekV4PackedKvAlignmentBytes);
  uint8_t *block_ptr =
      deepseek_indexer_kv + deepseek_indexer_kv_offset_bytes +
      static_cast<uint64_t>(compressed_block) * block_stride;
  uint8_t *data_ptr = block_ptr + static_cast<uint64_t>(kv_pos) * head_dim;
  uint8_t *scale_ptr =
      block_ptr + static_cast<uint64_t>(packed_block_tokens) * head_dim +
      static_cast<uint64_t>(kv_pos) * scale_dim;
  float *scale_f32 = reinterpret_cast<float *>(scale_ptr);
  const uint16_t *norm_weight = arena + layout.deepseek_indexer_compressor_norm;
  const uint32_t rope = layout.deepseek_qk_rope_head_dim <= head_dim
                            ? layout.deepseek_qk_rope_head_dim
                            : 0u;
  const uint32_t compressed_pos = compressed_slot * compress_ratio;
  if (threadIdx.x == 0) {
    constexpr uint32_t quant_block = 128u;
    for (uint32_t scale_index = 0; scale_index < scale_count; ++scale_index) {
      const uint32_t start_dim = scale_index * quant_block;
      const uint32_t end_dim =
          start_dim + quant_block < head_dim ? start_dim + quant_block
                                             : head_dim;
      float absmax = 0.0f;
      for (uint32_t dim = start_dim; dim < end_dim; ++dim) {
        const float rotated = deepseek_v4_indexer_rotated_value(
            compressed, norm_weight, head_dim, rope, dim, compressed_pos, rrms,
            rope_theta, layout);
        const float quant_input = deepseek_session_bf16_bits_to_f32(
            deepseek_session_f32_to_bf16_bits(rotated));
        absmax = fmaxf(absmax, fabsf(quant_input));
      }
      const float raw = fmaxf(absmax, 1.0e-4f) / 448.0f;
      const float scale = exp2f(ceilf(log2f(raw)));
      quant_scales[scale_index] = scale;
      scale_f32[scale_index] = scale;
    }
  }
  __syncthreads();

  for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const float rotated = deepseek_v4_indexer_rotated_value(
        compressed, norm_weight, head_dim, rope, dim, compressed_pos, rrms,
        rope_theta, layout);
    const float quant_input = deepseek_session_bf16_bits_to_f32(
        deepseek_session_f32_to_bf16_bits(rotated));
    const float scale = quant_scales[dim / 128u];
    const float scaled = fminf(fmaxf(quant_input / scale, -448.0f), 448.0f);
    data_ptr[dim] = deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
  }
  if (threadIdx.x == 0 && deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterIndexerKvWrites),
        1ull);
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
