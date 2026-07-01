__device__ float deepseek_fp8_scaled_weight(const uint16_t *arena,
                                            uint64_t weight_offset,
                                            uint64_t scale_offset,
                                            uint32_t rows, uint32_t cols,
                                            uint32_t row, uint32_t col) {
  const auto *weights = reinterpret_cast<const uint8_t *>(arena + weight_offset);
  const uint32_t scale_cols = (cols + 127u) / 128u;
  const uint32_t scale_idx = (row / 128u) * scale_cols + (col / 128u);
  return nerva::deepseek::f8_e4m3fn_bits_to_f32(
             weights[static_cast<uint64_t>(row) * cols + col]) *
         f32_from_u16_slots(arena + scale_offset, scale_idx);
}

__device__ float deepseek_swiglu(float gate, float up, float swiglu_limit) {
  if (isfinite(swiglu_limit) && swiglu_limit > 0.0f) {
    gate = fminf(gate, swiglu_limit);
    up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
  }
  return silu(gate) * up;
}

__device__ float deepseek_fp8_e8m0_scaled_weight(const uint16_t *arena,
                                                 uint64_t weight_offset,
                                                 uint64_t scale_offset,
                                                 uint32_t rows, uint32_t cols,
                                                 uint32_t row, uint32_t col) {
  const auto *weights = reinterpret_cast<const uint8_t *>(arena + weight_offset);
  const auto *scales = reinterpret_cast<const uint8_t *>(arena + scale_offset);
  const uint32_t scale_cols = (cols + 127u) / 128u;
  const uint32_t scale_idx = (row / 128u) * scale_cols + (col / 128u);
  return nerva::deepseek::f8_e4m3fn_bits_to_f32(
             weights[static_cast<uint64_t>(row) * cols + col]) *
         nerva::deepseek::e8m0_exponent_bits_to_f32(scales[scale_idx]);
}

__device__ __forceinline__ uint64_t deepseek_device_fp8_slots(
    uint64_t rows, uint64_t cols) {
  return (rows * cols + 1u) / 2u;
}

__device__ __forceinline__ uint64_t deepseek_device_byte_slots(
    uint64_t values) {
  return (values + 1u) / 2u;
}

__device__ __forceinline__ uint64_t deepseek_device_rank3_slots(
    uint64_t depth, uint64_t rows, uint64_t cols) {
  return deepseek_device_byte_slots(depth * rows * cols);
}

__device__ __forceinline__ uint64_t deepseek_u64_from_u16_slots(
    const uint16_t *slots, uint64_t index) {
  const uint64_t base = index * 4u;
  return static_cast<uint64_t>(slots[base]) |
         (static_cast<uint64_t>(slots[base + 1u]) << 16u) |
         (static_cast<uint64_t>(slots[base + 2u]) << 32u) |
         (static_cast<uint64_t>(slots[base + 3u]) << 48u);
}

__device__ __forceinline__ uint32_t deepseek_device_scale_dim(
    uint32_t value) {
  return (value + 127u) / 128u;
}

__device__ __forceinline__ uint64_t deepseek_device_scale_f32_slots(
    uint32_t rows, uint32_t cols) {
  return static_cast<uint64_t>(deepseek_device_scale_dim(rows)) *
         deepseek_device_scale_dim(cols) * 2u;
}

__device__ float deepseek_fp8_rank3_scaled_weight(
    const uint16_t *arena, uint64_t weight_offset, uint64_t scale_offset,
    uint32_t rows, uint32_t cols, uint32_t expert, uint32_t row,
    uint32_t col) {
  const auto *weights = reinterpret_cast<const uint8_t *>(arena + weight_offset);
  const uint64_t weight_idx =
      (static_cast<uint64_t>(expert) * rows + row) * cols + col;
  const uint32_t scale_rows = deepseek_device_scale_dim(rows);
  const uint32_t scale_cols = deepseek_device_scale_dim(cols);
  const uint64_t scale_idx =
      (static_cast<uint64_t>(expert) * scale_rows + (row / 128u)) *
          scale_cols +
      (col / 128u);
  return nerva::deepseek::f8_e4m3fn_bits_to_f32(weights[weight_idx]) *
         f32_from_u16_slots(arena + scale_offset, scale_idx);
}

__device__ float deepseek_mxfp4_rank3_scaled_weight(
    const uint16_t *arena, uint64_t weight_offset, uint64_t scale_offset,
    uint32_t rows, uint32_t packed_cols, uint32_t expert, uint32_t row,
    uint32_t col) {
  const auto *weights = reinterpret_cast<const uint8_t *>(arena + weight_offset);
  const auto *scales = reinterpret_cast<const uint8_t *>(arena + scale_offset);
  const uint32_t packed_col = col >> 1u;
  const uint8_t byte =
      weights[(static_cast<uint64_t>(expert) * rows + row) * packed_cols +
              packed_col];
  const uint8_t nibble =
      (col & 1u) == 0 ? (byte & 0x0fu) : (byte >> 4u);
  const uint32_t scale_cols = (packed_cols + 15u) / 16u;
  const uint64_t scale_idx =
      (static_cast<uint64_t>(expert) * rows + row) * scale_cols +
      (packed_col / 16u);
  return nerva::deepseek::mxfp4_e2m1_nibble_to_f32(nibble) *
         nerva::deepseek::e8m0_exponent_bits_to_f32(scales[scale_idx]);
}

__device__ float deepseek_yarn_get_mscale(float scale, float mscale) {
  if (scale <= 1.0f) {
    return 1.0f;
  }
  return 0.1f * mscale * logf(scale) + 1.0f;
}

__device__ float deepseek_linear_ramp_mask(float low, float high,
                                           float offset) {
  if (low == high) {
    high += 0.001f;
  }
  return fminf(fmaxf((offset - low) / (high - low), 0.0f), 1.0f);
}

__device__ float deepseek_rope_magnitude(const SequenceLayerLayout &layout) {
  if (layout.deepseek_rope_scaling_type != kDeepSeekRopeScalingDeepSeek ||
      !(layout.deepseek_rope_scaling_factor > 0.0f) ||
      !isfinite(layout.deepseek_rope_scaling_factor)) {
    return 1.0f;
  }
  const float numerator = deepseek_yarn_get_mscale(
      layout.deepseek_rope_scaling_factor, layout.deepseek_rope_mscale);
  const float denominator = deepseek_yarn_get_mscale(
      layout.deepseek_rope_scaling_factor, layout.deepseek_rope_mscale_all_dim);
  const float attn_factor =
      isfinite(layout.deepseek_rope_attn_factor) &&
              layout.deepseek_rope_attn_factor > 0.0f
          ? layout.deepseek_rope_attn_factor
          : 1.0f;
  return denominator == 0.0f ? attn_factor : numerator / denominator * attn_factor;
}

__device__ float deepseek_rope_inv_freq(const SequenceLayerLayout &layout,
                                        uint32_t offset, uint32_t dim,
                                        float theta) {
  const float exponent =
      static_cast<float>(2u * offset) / static_cast<float>(dim);
  const float pos_freq = powf(theta, exponent);
  if (layout.deepseek_rope_scaling_type != kDeepSeekRopeScalingDeepSeek ||
      !(layout.deepseek_rope_scaling_factor > 0.0f) ||
      !isfinite(layout.deepseek_rope_scaling_factor) ||
      layout.deepseek_rope_original_max_position == 0 ||
      !(theta > 0.0f)) {
    return 1.0f / pos_freq;
  }

  constexpr float two_pi = 6.2831853071795864769f;
  const float beta_fast =
      isfinite(layout.deepseek_rope_beta_fast) &&
              layout.deepseek_rope_beta_fast > 0.0f
          ? layout.deepseek_rope_beta_fast
          : 32.0f;
  const float beta_slow =
      isfinite(layout.deepseek_rope_beta_slow) &&
              layout.deepseek_rope_beta_slow > 0.0f
          ? layout.deepseek_rope_beta_slow
          : 1.0f;
  const float original =
      static_cast<float>(layout.deepseek_rope_original_max_position);
  const float denom = 2.0f * logf(theta);
  float low = floorf(static_cast<float>(dim) *
                     logf(original / (beta_fast * two_pi)) / denom);
  float high = ceilf(static_cast<float>(dim) *
                     logf(original / (beta_slow * two_pi)) / denom);
  low = fminf(fmaxf(low, 0.0f), static_cast<float>(dim - 1u));
  high = fminf(fmaxf(high, 0.0f), static_cast<float>(dim - 1u));

  const float ramp =
      deepseek_linear_ramp_mask(low, high, static_cast<float>(offset));
  const float extrapolation_factor =
      isfinite(layout.deepseek_rope_extrapolation_factor) &&
              layout.deepseek_rope_extrapolation_factor > 0.0f
          ? layout.deepseek_rope_extrapolation_factor
          : 1.0f;
  const float inv_freq_mask = (1.0f - ramp) * extrapolation_factor;
  const float inv_freq_extrapolation = 1.0f / pos_freq;
  const float inv_freq_interpolation =
      1.0f / (layout.deepseek_rope_scaling_factor * pos_freq);
  return inv_freq_interpolation * (1.0f - inv_freq_mask) +
         inv_freq_extrapolation * inv_freq_mask;
}

__device__ float deepseek_rope_value_serial(float left, float right,
                                            uint32_t offset, uint32_t dim,
                                            uint32_t position, float theta,
                                            bool second,
                                            const SequenceLayerLayout &layout) {
  if (theta <= 0.0f || dim < 2) {
    return second ? right : left;
  }
  const float angle =
      static_cast<float>(position) *
      deepseek_rope_inv_freq(layout, offset, dim, theta);
  float sin_value = 0.0f;
  float cos_value = 0.0f;
  sincosf(angle, &sin_value, &cos_value);
  const float magnitude = deepseek_rope_magnitude(layout);
  return magnitude *
         (second ? right * cos_value + left * sin_value
                 : left * cos_value - right * sin_value);
}

__device__ __forceinline__ uint16_t deepseek_session_f32_to_bf16_bits(
    float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t lsb = (bits >> 16u) & 1u;
  const uint32_t rounded = bits + 0x7fffu + lsb;
  return static_cast<uint16_t>(rounded >> 16u);
}

__device__ __forceinline__ float deepseek_session_bf16_bits_to_f32(
    uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16u);
}

__device__ uint8_t deepseek_session_f32_to_f8_e4m3fn_bits_nearest(
    float value) {
  if (isnan(value)) {
    return 0x7fu;
  }
  uint8_t best_bits = 0;
  float best_error = INFINITY;
  for (uint32_t bits = 0; bits <= 254u; ++bits) {
    const float candidate =
        nerva::deepseek::f8_e4m3fn_bits_to_f32(static_cast<uint8_t>(bits));
    if (isnan(candidate)) {
      continue;
    }
    const float error = fabsf(candidate - value);
    if (error < best_error ||
        (error == best_error && bits < static_cast<uint32_t>(best_bits))) {
      best_error = error;
      best_bits = static_cast<uint8_t>(bits);
    }
  }
  return best_bits;
}

__device__ __forceinline__ uint8_t deepseek_session_encode_e8m0_scale(
    float scale) {
  int exponent = static_cast<int>(ceilf(log2f(scale)));
  exponent += 127;
  if (exponent < 0) {
    exponent = 0;
  }
  if (exponent > 255) {
    exponent = 255;
  }
  return static_cast<uint8_t>(exponent);
}

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
  float absmax = 0.0f;
  for (uint32_t dim = 0; dim < head_dim; ++dim) {
    const float rotated = deepseek_session_rotated_compressed_value(
        compressed, arena + layout.deepseek_indexer_compressor_norm, head_dim,
        rope, dim, compressed_pos, rms_eps, rope_theta, layout);
    const float quant_input = deepseek_session_bf16_bits_to_f32(
        deepseek_session_f32_to_bf16_bits(rotated));
    absmax = fmaxf(absmax, fabsf(quant_input));
  }
  const float raw = fmaxf(absmax, 1.0e-4f) / 448.0f;
  const float scale = exp2f(ceilf(log2f(raw)));
  *reinterpret_cast<float *>(scale_ptr) = scale;
  for (uint32_t dim = 0; dim < head_dim; ++dim) {
    const float rotated = deepseek_session_rotated_compressed_value(
        compressed, arena + layout.deepseek_indexer_compressor_norm, head_dim,
        rope, dim, compressed_pos, rms_eps, rope_theta, layout);
    const float quant_input = deepseek_session_bf16_bits_to_f32(
        deepseek_session_f32_to_bf16_bits(rotated));
    const float scaled = fminf(fmaxf(quant_input / scale, -448.0f), 448.0f);
    data_ptr[dim] = deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
  }
  return true;
}

__device__ float deepseek_session_read_fp8_ds_mla_compressed_kv(
    const uint8_t *kv_cache, uint64_t kv_offset_bytes,
    uint32_t compressed_block_count, const SequenceLayerLayout &layout,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint32_t compressed_slot, uint32_t dim) {
  const uint32_t nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t head_dim = nope + rope;
  if (kv_cache == nullptr || dim >= head_dim) {
    return 0.0f;
  }
  const uint32_t packed_block_tokens =
      deepseek_v4_packed_kv_block_tokens(layout.deepseek_compress_ratio);
  const uint32_t logical_compressed_block = compressed_slot / packed_block_tokens;
  uint32_t compressed_block = 0;
  if (!deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, compressed_block_count,
          logical_compressed_block,
          packed_block_tokens * layout.deepseek_compress_ratio,
          &compressed_block)) {
    return 0.0f;
  }
  const uint32_t token_stride = nope + rope * 2u;
  const uint32_t scale_dim = nope / 64u + 1u;
  const uint32_t kv_pos = compressed_slot % packed_block_tokens;
  const uint64_t block_stride = deepseek_v4_round_up_u64(
      static_cast<uint64_t>(packed_block_tokens) *
          static_cast<uint64_t>(token_stride + scale_dim),
      kDeepSeekV4PackedKvAlignmentBytes);
  const uint8_t *block_ptr =
      kv_cache + kv_offset_bytes +
      static_cast<uint64_t>(compressed_block) * block_stride;
  const uint8_t *data_ptr =
      block_ptr + static_cast<uint64_t>(kv_pos) * token_stride;
  const uint8_t *scale_ptr =
      block_ptr + static_cast<uint64_t>(packed_block_tokens) * token_stride +
      static_cast<uint64_t>(kv_pos) * scale_dim;
  if (dim < nope) {
    const uint32_t scale_index = dim / 64u;
    const float scale =
        nerva::deepseek::e8m0_exponent_bits_to_f32(scale_ptr[scale_index]);
    return nerva::deepseek::f8_e4m3fn_bits_to_f32(data_ptr[dim]) * scale;
  }
  const uint32_t rope_local = dim - nope;
  const uint32_t offset = nope + rope_local * 2u;
  const uint16_t bits = static_cast<uint16_t>(data_ptr[offset]) |
                        (static_cast<uint16_t>(data_ptr[offset + 1u]) << 8u);
  return deepseek_session_bf16_bits_to_f32(bits);
}

__device__ bool deepseek_session_write_fp8_ds_mla_swa_kv(
    uint8_t *kv_cache, uint64_t kv_offset_bytes, uint32_t block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    const SequenceLayerLayout &layout, uint32_t position, const float *kv) {
  const uint32_t nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t head_dim = nope + rope;
  if (kv_cache == nullptr || kv == nullptr || head_dim == 0) {
    return false;
  }
  const uint32_t logical_block =
      position / kDeepSeekV4PackedKvDefaultBlockTokens;
  uint32_t block = 0;
  if (!deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, block_count, logical_block,
          kDeepSeekV4PackedKvDefaultBlockTokens, &block)) {
    return false;
  }
  const uint32_t token_stride = nope + rope * 2u;
  const uint32_t scale_dim = nope / 64u + 1u;
  const uint32_t kv_pos =
      position % kDeepSeekV4PackedKvDefaultBlockTokens;
  const uint64_t block_stride = deepseek_v4_round_up_u64(
      static_cast<uint64_t>(kDeepSeekV4PackedKvDefaultBlockTokens) *
          static_cast<uint64_t>(token_stride + scale_dim),
      kDeepSeekV4PackedKvAlignmentBytes);
  uint8_t *block_ptr =
      kv_cache + kv_offset_bytes + static_cast<uint64_t>(block) * block_stride;
  uint8_t *data_ptr = block_ptr + static_cast<uint64_t>(kv_pos) * token_stride;
  uint8_t *scale_ptr =
      block_ptr +
      static_cast<uint64_t>(kDeepSeekV4PackedKvDefaultBlockTokens) *
          token_stride +
      static_cast<uint64_t>(kv_pos) * scale_dim;
  const uint32_t quant_block = 64u;
  for (uint32_t scale_index = 0; scale_index < scale_dim; ++scale_index) {
    const uint32_t start = scale_index * quant_block;
    const uint32_t end =
        start + quant_block < nope ? start + quant_block : nope;
    float absmax = 0.0f;
    for (uint32_t dim = start; dim < end; ++dim) {
      const float quant_input = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(kv[dim]));
      absmax = fmaxf(absmax, fabsf(quant_input));
    }
    const float raw = fmaxf(absmax, 1.0e-4f) / 448.0f;
    const float scale = exp2f(ceilf(log2f(raw)));
    scale_ptr[scale_index] =
        start < nope ? deepseek_session_encode_e8m0_scale(scale) : 0u;
    for (uint32_t dim = start; dim < end; ++dim) {
      const float quant_input = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(kv[dim]));
      const float scaled = fminf(fmaxf(quant_input / scale, -448.0f), 448.0f);
      data_ptr[dim] = deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
    }
  }
  for (uint32_t dim = nope; dim < head_dim; ++dim) {
    const uint16_t bits = deepseek_session_f32_to_bf16_bits(kv[dim]);
    const uint32_t rope_local = dim - nope;
    data_ptr[nope + rope_local * 2u] = static_cast<uint8_t>(bits & 0xffu);
    data_ptr[nope + rope_local * 2u + 1u] =
        static_cast<uint8_t>(bits >> 8u);
  }
  return true;
}

__device__ bool deepseek_session_write_v32_fp8_ds_mla_kv(
    uint8_t *kv_cache, uint64_t kv_offset_bytes, uint32_t block_count,
    const uint32_t *kv_block_table, uint32_t kv_block_count,
    const SequenceLayerLayout &layout, uint32_t position, uint32_t dtype,
    const uint16_t *kv_latent_norm, const float *kv_a, float rope_theta) {
  if (kv_cache == nullptr || kv_latent_norm == nullptr || kv_a == nullptr ||
      block_count == 0 ||
      layout.attention_kind != kAttentionKindDeepSeekMla ||
      layout.deepseek_mode != kDeepSeekModeV32MlaIndexer ||
      layout.deepseek_kv_lora_rank != kDeepSeekV32PackedKvNopeBytes ||
      layout.deepseek_qk_rope_head_dim != kDeepSeekV32PackedKvRopeValues) {
    return false;
  }
  const uint32_t logical_block = position / kDeepSeekV32PackedKvBlockTokens;
  uint32_t physical_block = 0;
  if (!deepseek_v32_packed_physical_block(
          kv_block_table, kv_block_count, block_count, logical_block,
          &physical_block)) {
    return false;
  }
  const uint32_t kv_pos = position % kDeepSeekV32PackedKvBlockTokens;
  uint8_t *row =
      kv_cache + kv_offset_bytes +
      (static_cast<uint64_t>(physical_block) * kDeepSeekV32PackedKvBlockTokens +
       kv_pos) *
          kDeepSeekV32PackedKvTokenBytes;
  uint8_t *nope_ptr = row;
  float *scale_ptr = reinterpret_cast<float *>(
      row + kDeepSeekV32PackedKvNopeBytes);
  uint8_t *rope_ptr =
      row + kDeepSeekV32PackedKvNopeBytes + kDeepSeekV32PackedKvScaleBytes;

  constexpr uint32_t kScaleBlockValues = 128;
  constexpr uint32_t kScaleCount =
      kDeepSeekV32PackedKvNopeBytes / kScaleBlockValues;
  for (uint32_t scale_index = 0; scale_index < kScaleCount; ++scale_index) {
    const uint32_t start = scale_index * kScaleBlockValues;
    float absmax = 0.0f;
    for (uint32_t dim = 0; dim < kScaleBlockValues; ++dim) {
      const float value = encoded_to_f32(kv_latent_norm[start + dim], dtype);
      const float quant_input = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(value));
      absmax = fmaxf(absmax, fabsf(quant_input));
    }
    const float raw = fmaxf(absmax, 1.0e-4f) / 448.0f;
    const float scale = exp2f(ceilf(log2f(raw)));
    scale_ptr[scale_index] = scale;
    for (uint32_t dim = 0; dim < kScaleBlockValues; ++dim) {
      const uint32_t source_dim = start + dim;
      const float value = encoded_to_f32(kv_latent_norm[source_dim], dtype);
      const float quant_input = deepseek_session_bf16_bits_to_f32(
          deepseek_session_f32_to_bf16_bits(value));
      const float scaled = fminf(fmaxf(quant_input / scale, -448.0f), 448.0f);
      nope_ptr[source_dim] =
          deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
    }
  }

  const uint32_t rope_half = kDeepSeekV32PackedKvRopeValues / 2u;
  for (uint32_t dim = 0; dim < kDeepSeekV32PackedKvRopeValues; ++dim) {
    float value = kv_a[kDeepSeekV32PackedKvNopeBytes + dim];
    if (rope_half != 0) {
      const uint32_t offset = dim % rope_half;
      value = deepseek_rope_value_serial(
          kv_a[kDeepSeekV32PackedKvNopeBytes + offset],
          kv_a[kDeepSeekV32PackedKvNopeBytes + offset + rope_half], offset,
          kDeepSeekV32PackedKvRopeValues, position, rope_theta,
          dim >= rope_half, layout);
    }
    const uint16_t bits = deepseek_session_f32_to_bf16_bits(value);
    rope_ptr[dim * 2u] = static_cast<uint8_t>(bits & 0xffu);
    rope_ptr[dim * 2u + 1u] = static_cast<uint8_t>(bits >> 8u);
  }
  return true;
}

__device__ float deepseek_session_read_fp8_ds_mla_swa_kv(
    const uint8_t *kv_cache, uint64_t kv_offset_bytes, uint32_t block_count,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    const SequenceLayerLayout &layout, uint32_t position, uint32_t dim) {
  const uint32_t nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t head_dim = nope + rope;
  if (kv_cache == nullptr || dim >= head_dim) {
    return 0.0f;
  }
  const uint32_t logical_block =
      position / kDeepSeekV4PackedKvDefaultBlockTokens;
  uint32_t block = 0;
  if (!deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, block_count, logical_block,
          kDeepSeekV4PackedKvDefaultBlockTokens, &block)) {
    return 0.0f;
  }
  const uint32_t token_stride = nope + rope * 2u;
  const uint32_t scale_dim = nope / 64u + 1u;
  const uint32_t kv_pos =
      position % kDeepSeekV4PackedKvDefaultBlockTokens;
  const uint64_t block_stride = deepseek_v4_round_up_u64(
      static_cast<uint64_t>(kDeepSeekV4PackedKvDefaultBlockTokens) *
          static_cast<uint64_t>(token_stride + scale_dim),
      kDeepSeekV4PackedKvAlignmentBytes);
  const uint8_t *block_ptr =
      kv_cache + kv_offset_bytes + static_cast<uint64_t>(block) * block_stride;
  const uint8_t *data_ptr =
      block_ptr + static_cast<uint64_t>(kv_pos) * token_stride;
  const uint8_t *scale_ptr =
      block_ptr +
      static_cast<uint64_t>(kDeepSeekV4PackedKvDefaultBlockTokens) *
          token_stride +
      static_cast<uint64_t>(kv_pos) * scale_dim;
  if (dim < nope) {
    const uint32_t scale_index = dim / 64u;
    const float scale =
        nerva::deepseek::e8m0_exponent_bits_to_f32(scale_ptr[scale_index]);
    return nerva::deepseek::f8_e4m3fn_bits_to_f32(data_ptr[dim]) * scale;
  }
  const uint32_t rope_local = dim - nope;
  const uint32_t offset = nope + rope_local * 2u;
  const uint16_t bits = static_cast<uint16_t>(data_ptr[offset]) |
                        (static_cast<uint16_t>(data_ptr[offset + 1u]) << 8u);
  return deepseek_session_bf16_bits_to_f32(bits);
}

__device__ float deepseek_session_read_indexer_fp8_compressed_kv(
    const uint8_t *kv_cache, uint64_t kv_offset_bytes,
    uint32_t compressed_block_count, const SequenceLayerLayout &layout,
    uint32_t kv_block_count, const uint32_t *kv_block_table,
    uint32_t compressed_slot, uint32_t dim) {
  const uint32_t head_dim = layout.deepseek_index_head_dim;
  const uint32_t scale_dim = deepseek_device_scale_dim(head_dim) * sizeof(float);
  if (kv_cache == nullptr || head_dim == 0 || dim >= head_dim ||
      scale_dim < sizeof(float)) {
    return 0.0f;
  }
  const uint32_t packed_block_tokens =
      deepseek_v4_packed_kv_block_tokens(layout.deepseek_compress_ratio);
  const uint32_t logical_compressed_block = compressed_slot / packed_block_tokens;
  uint32_t compressed_block = 0;
  if (!deepseek_v4_packed_physical_block(
          kv_block_table, kv_block_count, compressed_block_count,
          logical_compressed_block,
          packed_block_tokens * layout.deepseek_compress_ratio,
          &compressed_block)) {
    return 0.0f;
  }
  const uint32_t kv_pos = compressed_slot % packed_block_tokens;
  const uint64_t block_stride = deepseek_v4_round_up_u64(
      static_cast<uint64_t>(packed_block_tokens) *
          static_cast<uint64_t>(head_dim + scale_dim),
      kDeepSeekV4PackedKvAlignmentBytes);
  const uint8_t *block_ptr =
      kv_cache + kv_offset_bytes +
      static_cast<uint64_t>(compressed_block) * block_stride;
  const uint8_t *data_ptr =
      block_ptr + static_cast<uint64_t>(kv_pos) * head_dim;
  const uint8_t *scale_ptr =
      block_ptr + static_cast<uint64_t>(packed_block_tokens) * head_dim +
      static_cast<uint64_t>(kv_pos) * scale_dim;
  const uint32_t scale_index = dim / 128u;
  const float scale =
      reinterpret_cast<const float *>(scale_ptr)[scale_index];
  return nerva::deepseek::f8_e4m3fn_bits_to_f32(data_ptr[dim]) * scale;
}

__device__ float deepseek_session_indexer_query_rope_value(
    const float *query_head, uint32_t head_dim, uint32_t rope_head_dim,
    uint32_t dim, uint32_t position, float rope_theta,
    const SequenceLayerLayout &layout) {
  if (rope_theta <= 0.0f || rope_head_dim < 2 || rope_head_dim > head_dim) {
    return query_head[dim];
  }
  const uint32_t nope_dim = head_dim - rope_head_dim;
  if (dim < nope_dim) {
    return query_head[dim];
  }
  const uint32_t rope_local = dim - nope_dim;
  const uint32_t pair = rope_local / 2u;
  const uint32_t even_dim = nope_dim + pair * 2u;
  const uint32_t odd_dim = even_dim + 1u;
  if (odd_dim >= head_dim) {
    return query_head[dim];
  }
  const float even = query_head[even_dim];
  const float odd = query_head[odd_dim];
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

__device__ bool deepseek_session_sparse_score_is_better(
    float candidate, int32_t slot, float current, int32_t current_slot) {
  if (!isfinite(candidate)) {
    return false;
  }
  if (current_slot < 0) {
    return true;
  }
  return candidate > current ||
         (candidate == current && slot >= 0 && slot < current_slot);
}

__device__ uint32_t deepseek_session_select_v4_c4_sparse_slots(
    uint16_t *arena, const SequenceLayerLayout &layout, const float *qr_norm,
    const uint16_t *projection_input, uint32_t dtype, uint32_t hidden,
    uint32_t q_lora_rank, uint32_t position, float rope_theta,
    const uint8_t *indexer_kv, uint64_t indexer_kv_offset_bytes,
    uint32_t indexer_kv_block_count, uint32_t kv_block_count,
    const uint32_t *kv_block_table, uint32_t compressed_attention_tokens,
    int32_t *topk_slots, float *topk_scores, float *indexer_query,
    uint32_t *scored_candidates, unsigned long long *selection_hash_out) {
  if (scored_candidates != nullptr) {
    *scored_candidates = 0;
  }
  if (selection_hash_out != nullptr) {
    *selection_hash_out = 0ull;
  }
  if (layout.deepseek_mode != kDeepSeekModeV4CompressedIndexer ||
      (layout.deepseek_flags & kDeepSeekFlagSparseIndexer) == 0 ||
      layout.deepseek_compress_ratio != 4 ||
      layout.deepseek_index_topk == 0 ||
      layout.deepseek_index_n_heads == 0 ||
      layout.deepseek_index_head_dim == 0 ||
      layout.deepseek_index_n_heads > kDeepSeekSessionMaxIndexerHeads ||
      layout.deepseek_index_head_dim > kDeepSeekSessionMaxCompressHeadSize ||
      static_cast<uint64_t>(layout.deepseek_index_n_heads) *
              layout.deepseek_index_head_dim >
          kDeepSeekSessionMaxIndexerQueryValues ||
      compressed_attention_tokens == 0 || indexer_kv == nullptr ||
      indexer_kv_block_count == 0 ||
      layout.deepseek_indexer_q == kMissingOffset ||
      layout.deepseek_indexer_q_scale == kMissingOffset ||
      layout.deepseek_indexer_weights == kMissingOffset) {
    return 0;
  }

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
  if (topk_limit == 0) {
    return 0;
  }
  if (topk_limit >= compressed_attention_tokens) {
    for (uint32_t slot = 0; slot < compressed_attention_tokens; ++slot) {
      topk_slots[slot] = static_cast<int32_t>(slot);
      topk_scores[slot] = INFINITY;
    }
  } else {
    if (scored_candidates != nullptr) {
      *scored_candidates = compressed_attention_tokens;
    }

    float indexer_weights[kDeepSeekSessionMaxIndexerHeads];
    for (uint32_t head = 0; head < index_heads; ++head) {
      float weight_sum = 0.0f;
      for (uint32_t col = 0; col < hidden; ++col) {
        weight_sum +=
            encoded_to_f32(arena[layout.deepseek_indexer_weights +
                                 static_cast<uint64_t>(head) * hidden + col],
                           kDTypeBF16) *
            encoded_to_f32(projection_input[col], dtype);
      }
      indexer_weights[head] = weight_sum;
    }

    for (uint32_t row = 0; row < query_rows; ++row) {
      float sum = 0.0f;
      for (uint32_t col = 0; col < q_lora_rank; ++col) {
        sum += deepseek_fp8_e8m0_scaled_weight(
                   arena, layout.deepseek_indexer_q,
                   layout.deepseek_indexer_q_scale, query_rows, q_lora_rank,
                   row, col) *
               qr_norm[col];
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
        const float scaled = fminf(fmaxf(query_head[dim] / q_scale, -448.0f),
                                   448.0f);
        const uint8_t q_bits =
            deepseek_session_f32_to_f8_e4m3fn_bits_nearest(scaled);
        query_head[dim] =
            nerva::deepseek::f8_e4m3fn_bits_to_f32(q_bits) * q_scale;
      }
    }

    for (uint32_t rank = 0; rank < topk_limit; ++rank) {
      topk_slots[rank] = -1;
      topk_scores[rank] = -INFINITY;
    }

    const float softmax_scale = rsqrtf(static_cast<float>(index_head_dim));
    const float head_scale = rsqrtf(static_cast<float>(index_heads));
    for (uint32_t slot = 0; slot < compressed_attention_tokens; ++slot) {
      float score = 0.0f;
      for (uint32_t head = 0; head < index_heads; ++head) {
        float dot = 0.0f;
        const float *query_head = indexer_query + head * index_head_dim;
        for (uint32_t dim = 0; dim < index_head_dim; ++dim) {
          dot += query_head[dim] *
                 deepseek_session_read_indexer_fp8_compressed_kv(
                     indexer_kv, indexer_kv_offset_bytes,
                     indexer_kv_block_count, layout, kv_block_count,
                     kv_block_table, slot, dim);
        }
        score += indexer_weights[head] * softmax_scale * head_scale * dot;
      }
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
    if (topk_slots[rank] >= 0) {
      ++selected;
      selection_hash +=
          (static_cast<unsigned long long>(position) + 1ull) *
              1315423911ull ^
          (static_cast<unsigned long long>(rank) + 1ull) * 2654435761ull ^
          (static_cast<unsigned long long>(topk_slots[rank]) + 1ull);
    }
  }
  if (selection_hash_out != nullptr) {
    *selection_hash_out = selection_hash;
  }
  return selected;
}

__device__ uint32_t deepseek_session_compressed_attention_count(
    const uint8_t *kv_cache, const SequenceLayerLayout &layout,
    uint32_t compressed_block_count, uint32_t position) {
  if (kv_cache == nullptr ||
      (layout.deepseek_mode != kDeepSeekModeV4Compressed &&
       layout.deepseek_mode != kDeepSeekModeV4CompressedIndexer) ||
      layout.deepseek_compress_ratio <= 1 || compressed_block_count == 0) {
    return 0;
  }
  uint32_t compressed_tokens =
      (position + 1u) / layout.deepseek_compress_ratio;
  const uint32_t capacity =
      compressed_block_count *
      deepseek_v4_packed_kv_block_tokens(layout.deepseek_compress_ratio);
  if (compressed_tokens > capacity) {
    compressed_tokens = capacity;
  }
  return compressed_tokens;
}
