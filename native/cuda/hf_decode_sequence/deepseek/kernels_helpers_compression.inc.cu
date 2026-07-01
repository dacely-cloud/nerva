__device__ float deepseek_session_compressed_state_value(
    const float *state_cache, uint64_t state_offset_bytes,
    uint32_t state_width, uint32_t head_dim, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint32_t position, uint32_t dim,
    uint32_t compress_ratio, uint32_t window_index, bool score_half) {
  const uint32_t logical_block = position / kKvCacheBlockTokens;
  if (logical_block >= kv_block_count || dim >= head_dim) {
    return 0.0f;
  }
  const uint32_t physical_block =
      kv_block_table == nullptr ? logical_block : kv_block_table[logical_block];
  const uint32_t pos_in_block = position % kKvCacheBlockTokens;
  const uint32_t head_offset =
      window_index >= compress_ratio ? head_dim : 0u;
  if (head_offset + dim >= state_width) {
    return 0.0f;
  }
  const uint64_t row_stride = static_cast<uint64_t>(state_width) * 2u;
  const uint64_t token_index =
      static_cast<uint64_t>(physical_block) * kKvCacheBlockTokens +
      pos_in_block;
  const uint64_t base =
      state_offset_bytes / sizeof(float) + token_index * row_stride +
      head_offset + dim;
  return score_half ? state_cache[base + state_width] : state_cache[base];
}

__device__ void deepseek_session_compress_state_to_scratch(
    const float *state_cache, uint64_t state_offset_bytes,
    uint32_t state_width, uint32_t head_dim, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint32_t position,
    uint32_t compress_ratio, float *compressed) {
  const uint32_t overlap = compress_ratio == 4 ? 1u : 0u;
  const uint32_t window_tokens = (1u + overlap) * compress_ratio;
  const int64_t start = static_cast<int64_t>(position) -
                        static_cast<int64_t>(window_tokens) + 1ll;
  for (uint32_t dim = 0; dim < head_dim; ++dim) {
    float max_score = -INFINITY;
    for (uint32_t window = 0; window < window_tokens; ++window) {
      const int64_t pos = start + static_cast<int64_t>(window);
      if (pos < 0) {
        continue;
      }
      const float score = deepseek_session_compressed_state_value(
          state_cache, state_offset_bytes, state_width, head_dim,
          kv_block_count, kv_block_table, static_cast<uint32_t>(pos), dim,
          compress_ratio, window, true);
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
          state_cache, state_offset_bytes, state_width, head_dim,
          kv_block_count, kv_block_table, pos_u32, dim, compress_ratio,
          window, true);
      const float weight = expf(score - max_score);
      weighted += deepseek_session_compressed_state_value(
                      state_cache, state_offset_bytes, state_width, head_dim,
                      kv_block_count, kv_block_table, pos_u32, dim,
                      compress_ratio, window, false) *
                  weight;
      denom += weight;
    }
    compressed[dim] = denom > 0.0f ? weighted / denom : 0.0f;
  }
}

__device__ float deepseek_session_normed_compressed_value(
    const float *compressed, const uint16_t *norm_weight, uint32_t head_dim,
    uint32_t dim, float rms_eps) {
  float variance = 0.0f;
  for (uint32_t index = 0; index < head_dim; ++index) {
    variance += compressed[index] * compressed[index];
  }
  const float rrms =
      rsqrtf(variance / static_cast<float>(head_dim) + rms_eps);
  return compressed[dim] * rrms *
         encoded_to_f32(norm_weight[dim], kDTypeBF16);
}

__device__ float deepseek_session_rotated_compressed_value(
    const float *compressed, const uint16_t *norm_weight, uint32_t head_dim,
    uint32_t rope_head_dim, uint32_t dim, uint32_t position, float rms_eps,
    float rope_theta, const SequenceLayerLayout &layout) {
  const uint32_t nope_head_dim = head_dim - rope_head_dim;
  const float value = deepseek_session_normed_compressed_value(
      compressed, norm_weight, head_dim, dim, rms_eps);
  if (rope_theta <= 0.0f || rope_head_dim < 2 || dim < nope_head_dim) {
    return value;
  }
  const uint32_t rope_local = dim - nope_head_dim;
  const uint32_t pair = rope_local / 2u;
  const uint32_t even_dim = nope_head_dim + pair * 2u;
  const uint32_t odd_dim = even_dim + 1u;
  if (odd_dim >= head_dim) {
    return value;
  }
  const float even = deepseek_session_normed_compressed_value(
      compressed, norm_weight, head_dim, even_dim, rms_eps);
  const float odd = deepseek_session_normed_compressed_value(
      compressed, norm_weight, head_dim, odd_dim, rms_eps);
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

__device__ bool deepseek_session_write_fp8_ds_mla_compressed_kv(
    uint16_t *arena, const float *state_cache, uint64_t state_offset_bytes,
    uint8_t *kv_cache, uint64_t kv_offset_bytes,
    uint32_t compressed_block_count, const SequenceLayerLayout &layout,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint32_t position, float rms_eps, float rope_theta, float *compressed) {
  const uint32_t compress_ratio = layout.deepseek_compress_ratio;
  if (state_cache == nullptr || kv_cache == nullptr || compress_ratio <= 1 ||
      (position + 1u) % compress_ratio != 0 ||
      layout.deepseek_compressor_norm == kMissingOffset) {
    return false;
  }
  const uint32_t head_dim =
      layout.deepseek_qk_nope_head_dim + layout.deepseek_qk_rope_head_dim;
  const uint32_t coff = compress_ratio == 4 ? 2u : 1u;
  const uint32_t state_width = coff * head_dim;
  const uint32_t compressed_slot = position / compress_ratio;
  const uint32_t packed_block_tokens =
      deepseek_v4_packed_kv_block_tokens(compress_ratio);
  const uint32_t logical_compressed_block = compressed_slot / packed_block_tokens;
  uint32_t compressed_block = 0;
  if (head_dim == 0 || head_dim > kDeepSeekSessionMaxCompressHeadSize ||
      !deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, compressed_block_count,
          logical_compressed_block, packed_block_tokens * compress_ratio,
          &compressed_block)) {
    return false;
  }
  deepseek_session_compress_state_to_scratch(
      state_cache, state_offset_bytes, state_width, head_dim, kv_block_count,
      kv_block_table, position, compress_ratio, compressed);

  const uint32_t nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t token_stride = nope + rope * 2u;
  const uint32_t scale_dim = nope / 64u + 1u;
  const uint32_t kv_pos = compressed_slot % packed_block_tokens;
  const uint64_t block_stride = deepseek_v4_round_up_u64(
      static_cast<uint64_t>(packed_block_tokens) *
          static_cast<uint64_t>(token_stride + scale_dim),
      kDeepSeekV4PackedKvAlignmentBytes);
  uint8_t *block_ptr = kv_cache + kv_offset_bytes +
                       static_cast<uint64_t>(compressed_block) * block_stride;
  uint8_t *data_ptr = block_ptr + static_cast<uint64_t>(kv_pos) * token_stride;
  uint8_t *scale_ptr =
      block_ptr + static_cast<uint64_t>(packed_block_tokens) * token_stride +
      static_cast<uint64_t>(kv_pos) * scale_dim;
  const uint32_t quant_block = 64u;
  const uint32_t blocks = scale_dim;
  for (uint32_t block = 0; block < blocks; ++block) {
    const uint32_t start = block * quant_block;
    const uint32_t end =
        start + quant_block < nope ? start + quant_block : nope;
    float absmax = 0.0f;
    for (uint32_t dim = start; dim < end; ++dim) {
      const float normed = deepseek_session_normed_compressed_value(
          compressed, arena + layout.deepseek_compressor_norm, head_dim, dim,
          rms_eps);
      const float quant_input = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(normed));
      absmax = fmaxf(absmax, fabsf(quant_input));
    }
    const float raw = fmaxf(absmax, 1.0e-4f) / 448.0f;
    const float scale = exp2f(ceilf(log2f(raw)));
    scale_ptr[block] = block * quant_block < nope
                           ? deepseek_session_encode_e8m0_scale(scale)
                           : 0u;
    for (uint32_t dim = start; dim < end; ++dim) {
      const float normed = deepseek_session_normed_compressed_value(
          compressed, arena + layout.deepseek_compressor_norm, head_dim, dim,
          rms_eps);
      const float quant_input = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(normed));
      const float scaled = fminf(fmaxf(quant_input / scale, -448.0f), 448.0f);
      data_ptr[dim] =
          deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
    }
  }
  const uint32_t compressed_pos = compressed_slot * compress_ratio;
  for (uint32_t dim = nope; dim < head_dim; ++dim) {
    const float rotated = deepseek_session_rotated_compressed_value(
        compressed, arena + layout.deepseek_compressor_norm, head_dim, rope,
        dim, compressed_pos, rms_eps, rope_theta, layout);
    const uint16_t bits = deepseek_session_f32_to_bf16_bits(rotated);
    const uint32_t rope_local = dim - nope;
    data_ptr[nope + rope_local * 2u] = static_cast<uint8_t>(bits & 0xffu);
    data_ptr[nope + rope_local * 2u + 1u] =
        static_cast<uint8_t>(bits >> 8u);
  }
  return true;
}

__device__ bool deepseek_session_write_indexer_fp8_compressed_kv(
    uint16_t *arena, const float *state_cache, uint64_t state_offset_bytes,
    uint8_t *kv_cache, uint64_t kv_offset_bytes,
    uint32_t compressed_block_count, const SequenceLayerLayout &layout,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint32_t position, float rms_eps, float rope_theta, float *compressed) {
  const uint32_t compress_ratio = layout.deepseek_compress_ratio;
  const uint32_t head_dim = layout.deepseek_index_head_dim;
  const uint32_t scale_dim = deepseek_device_scale_dim(head_dim) * sizeof(float);
  if (state_cache == nullptr || kv_cache == nullptr || compress_ratio <= 1 ||
      (position + 1u) % compress_ratio != 0 || head_dim == 0 ||
      scale_dim < sizeof(float) ||
      layout.deepseek_indexer_compressor_norm == kMissingOffset) {
    return false;
  }
  const uint32_t coff = compress_ratio == 4 ? 2u : 1u;
  const uint32_t state_width = coff * head_dim;
  const uint32_t compressed_slot = position / compress_ratio;
  const uint32_t packed_block_tokens =
      deepseek_v4_packed_kv_block_tokens(compress_ratio);
  const uint32_t logical_compressed_block = compressed_slot / packed_block_tokens;
  uint32_t compressed_block = 0;
  if (head_dim > kDeepSeekSessionMaxCompressHeadSize ||
      !deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, compressed_block_count,
          logical_compressed_block, packed_block_tokens * compress_ratio,
          &compressed_block)) {
    return false;
  }
  deepseek_session_compress_state_to_scratch(
      state_cache, state_offset_bytes, state_width, head_dim, kv_block_count,
      kv_block_table, position, compress_ratio, compressed);

  const uint32_t token_stride = head_dim;
  const uint32_t kv_pos = compressed_slot % packed_block_tokens;
  const uint64_t block_stride = deepseek_v4_round_up_u64(
      static_cast<uint64_t>(packed_block_tokens) *
          static_cast<uint64_t>(token_stride + scale_dim),
      kDeepSeekV4PackedKvAlignmentBytes);
  uint8_t *block_ptr = kv_cache + kv_offset_bytes +
                       static_cast<uint64_t>(compressed_block) * block_stride;
  uint8_t *data_ptr = block_ptr + static_cast<uint64_t>(kv_pos) * token_stride;
  uint8_t *scale_ptr =
      block_ptr + static_cast<uint64_t>(packed_block_tokens) * token_stride +
      static_cast<uint64_t>(kv_pos) * scale_dim;

  const uint32_t rope = layout.deepseek_qk_rope_head_dim <= head_dim
                            ? layout.deepseek_qk_rope_head_dim
                            : 0u;
  const uint32_t compressed_pos = compressed_slot * compress_ratio;
  const uint32_t quant_block = 128u;
  float *scales = reinterpret_cast<float *>(scale_ptr);
  for (uint32_t scale_index = 0;
       scale_index < scale_dim / sizeof(float); ++scale_index) {
    const uint32_t start_dim = scale_index * quant_block;
    const uint32_t end_dim =
        start_dim + quant_block < head_dim ? start_dim + quant_block
                                           : head_dim;
    float absmax = 0.0f;
    for (uint32_t dim = start_dim; dim < end_dim; ++dim) {
      const float rotated = deepseek_session_rotated_compressed_value(
          compressed, arena + layout.deepseek_indexer_compressor_norm, head_dim,
          rope, dim, compressed_pos, rms_eps, rope_theta, layout);
      const float quant_input = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(rotated));
      absmax = fmaxf(absmax, fabsf(quant_input));
    }
    const float raw = fmaxf(absmax, 1.0e-4f) / 448.0f;
    const float scale = exp2f(ceilf(log2f(raw)));
    scales[scale_index] = scale;
    for (uint32_t dim = start_dim; dim < end_dim; ++dim) {
      const float rotated = deepseek_session_rotated_compressed_value(
          compressed, arena + layout.deepseek_indexer_compressor_norm, head_dim,
          rope, dim, compressed_pos, rms_eps, rope_theta, layout);
      const float quant_input = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(rotated));
      const float scaled = fminf(fmaxf(quant_input / scale, -448.0f), 448.0f);
      data_ptr[dim] = deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
    }
  }
  return true;
}
