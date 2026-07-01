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

__device__ float deepseek_bf16_weight(const uint16_t *arena,
                                      uint64_t weight_offset,
                                      uint32_t rows, uint32_t cols,
                                      uint32_t row, uint32_t col) {
  (void)rows;
  return encoded_to_f32(
      arena[weight_offset + static_cast<uint64_t>(row) * cols + col],
      kDTypeBF16);
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

__device__ float deepseek_bf16_rank3_weight(
    const uint16_t *arena, uint64_t weight_offset, uint32_t rows,
    uint32_t cols, uint32_t expert, uint32_t row, uint32_t col) {
  const uint64_t weight_idx =
      (static_cast<uint64_t>(expert) * rows + row) * cols + col;
  return encoded_to_f32(arena[weight_offset + weight_idx], kDTypeBF16);
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

__device__ float deepseek_mla_attention_scale(const SequenceLayerLayout &layout,
                                              uint32_t qk_head_dim) {
  if (qk_head_dim == 0) {
    return 0.0f;
  }
  float scale = rsqrtf(static_cast<float>(qk_head_dim));
  if (layout.deepseek_rope_scaling_type == kDeepSeekRopeScalingDeepSeek &&
      layout.deepseek_rope_scaling_factor > 0.0f &&
      isfinite(layout.deepseek_rope_scaling_factor)) {
    const float mscale = deepseek_yarn_get_mscale(
        layout.deepseek_rope_scaling_factor,
        layout.deepseek_rope_mscale_all_dim);
    scale *= mscale * mscale;
  }
  return scale;
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
