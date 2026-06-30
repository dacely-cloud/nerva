__device__ void deepseek_session_apply_v4_mhc_pre_state(
    const uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t position, float rms_eps, const float *layer_input,
    uint32_t initialize_residual, uint64_t hc_base_offset,
    uint64_t hc_fn_offset, uint64_t hc_scale_offset, uint64_t norm_weight_offset,
    float *deepseek_mhc_residual, float *deepseek_mhc_post_mix,
    float *deepseek_mhc_comb_mix, float *temp_layer_input,
    uint16_t *projection_input) {
  const uint32_t hc_mult = layout.deepseek_hc_mult;
  if (arena == nullptr || layer_input == nullptr ||
      deepseek_mhc_residual == nullptr || deepseek_mhc_post_mix == nullptr ||
      deepseek_mhc_comb_mix == nullptr || temp_layer_input == nullptr ||
      projection_input == nullptr || hidden == 0 || hc_mult == 0 ||
      dtype > kDTypeBF16 || hc_mult > kDeepSeekSessionMaxMhcHcMult ||
      layout.deepseek_hc_sinkhorn_iters == 0 ||
      norm_weight_offset == kMissingOffset ||
      hc_base_offset == kMissingOffset || hc_fn_offset == kMissingOffset ||
      hc_scale_offset == kMissingOffset) {
    return;
  }

  const uint64_t hc_hidden =
      static_cast<uint64_t>(hc_mult) * static_cast<uint64_t>(hidden);
  const uint64_t hc_mix_count =
      static_cast<uint64_t>(hc_mult) * (2u + hc_mult);
  if (hc_mix_count > kDeepSeekSessionMaxMhcMixes) {
    return;
  }

  const uint64_t token_residual_offset =
      static_cast<uint64_t>(position) * hc_hidden;
  float sqrsum = 0.0f;
  if (initialize_residual != 0) {
    for (uint32_t dim = 0; dim < hidden; ++dim) {
      const float rounded =
          deepseek_session_bf16_bits_to_f32(deepseek_session_f32_to_bf16_bits(
              layer_input[dim]));
      for (uint32_t channel = 0; channel < hc_mult; ++channel) {
        deepseek_mhc_residual
            [token_residual_offset + static_cast<uint64_t>(channel) * hidden +
             dim] = rounded;
        sqrsum += rounded * rounded;
      }
    }
  } else {
    const uint64_t token_post_offset =
        static_cast<uint64_t>(position) * hc_mult;
    const uint64_t token_comb_offset =
        static_cast<uint64_t>(position) * hc_mult * hc_mult;
    for (uint32_t dim = 0; dim < hidden; ++dim) {
      float old_values[kDeepSeekSessionMaxMhcHcMult];
      float new_values[kDeepSeekSessionMaxMhcHcMult];
      for (uint32_t channel = 0; channel < hc_mult; ++channel) {
        old_values[channel] =
            deepseek_mhc_residual
                [token_residual_offset +
                 static_cast<uint64_t>(channel) * hidden + dim];
      }
      for (uint32_t out_channel = 0; out_channel < hc_mult; ++out_channel) {
        float value =
            deepseek_mhc_post_mix[token_post_offset + out_channel] *
            layer_input[dim];
        for (uint32_t in_channel = 0; in_channel < hc_mult; ++in_channel) {
          value += deepseek_mhc_comb_mix
                       [token_comb_offset +
                        static_cast<uint64_t>(in_channel) * hc_mult +
                        out_channel] *
                   old_values[in_channel];
        }
        new_values[out_channel] =
            deepseek_session_bf16_bits_to_f32(deepseek_session_f32_to_bf16_bits(
                value));
      }
      for (uint32_t channel = 0; channel < hc_mult; ++channel) {
        const float value = new_values[channel];
        deepseek_mhc_residual
            [token_residual_offset + static_cast<uint64_t>(channel) * hidden +
             dim] = value;
        sqrsum += value * value;
      }
    }
  }
  if (!(sqrsum > 0.0f) || !isfinite(sqrsum)) {
    return;
  }
  const float rms_scale =
      rsqrtf(sqrsum / static_cast<float>(hc_hidden) + rms_eps);

  float mixes[kDeepSeekSessionMaxMhcMixes];
  for (uint64_t mix = 0; mix < hc_mix_count; ++mix) {
    float value = 0.0f;
    const uint64_t row_offset = mix * hc_hidden;
    for (uint32_t channel = 0; channel < hc_mult; ++channel) {
      const uint64_t channel_offset =
          static_cast<uint64_t>(channel) * hidden;
      for (uint32_t dim = 0; dim < hidden; ++dim) {
        value += f32_from_u16_slots(arena + hc_fn_offset,
                                    row_offset + channel_offset + dim) *
                 deepseek_mhc_residual[token_residual_offset + channel_offset +
                                        dim];
      }
    }
    mixes[mix] = value * rms_scale;
  }

  const float hc_scale0 = f32_from_u16_slots(arena + hc_scale_offset, 0);
  const float hc_scale1 = f32_from_u16_slots(arena + hc_scale_offset, 1);
  const float hc_scale2 = f32_from_u16_slots(arena + hc_scale_offset, 2);
  const float hc_pre_eps = layout.deepseek_hc_eps;
  const float hc_sinkhorn_eps = layout.deepseek_hc_eps;
  const float hc_post_mult =
      isfinite(layout.deepseek_hc_post_alpha) &&
              layout.deepseek_hc_post_alpha != 0.0f
          ? layout.deepseek_hc_post_alpha
          : 1.0f;

  float pre_mix[kDeepSeekSessionMaxMhcHcMult];
  for (uint32_t channel = 0; channel < hc_mult; ++channel) {
    pre_mix[channel] =
        sigmoid(mixes[channel] * hc_scale0 +
                f32_from_u16_slots(arena + hc_base_offset, channel)) +
        hc_pre_eps;
    deepseek_mhc_post_mix[static_cast<uint64_t>(position) * hc_mult +
                          channel] =
        sigmoid(mixes[hc_mult + channel] * hc_scale1 +
                f32_from_u16_slots(arena + hc_base_offset, hc_mult + channel)) *
        hc_post_mult;
  }

  float comb[kDeepSeekSessionMaxMhcHcMult * kDeepSeekSessionMaxMhcHcMult];
  for (uint32_t row = 0; row < hc_mult; ++row) {
    float row_max = -INFINITY;
    const uint64_t logits_start =
        static_cast<uint64_t>(2u * hc_mult) +
        static_cast<uint64_t>(row) * hc_mult;
    for (uint32_t col = 0; col < hc_mult; ++col) {
      const uint64_t index = static_cast<uint64_t>(row) * hc_mult + col;
      const float logit =
          mixes[logits_start + col] * hc_scale2 +
          f32_from_u16_slots(arena + hc_base_offset, logits_start + col);
      comb[index] = logit;
      row_max = fmaxf(row_max, logit);
    }
    float row_sum = 0.0f;
    for (uint32_t col = 0; col < hc_mult; ++col) {
      const uint64_t index = static_cast<uint64_t>(row) * hc_mult + col;
      const float value = expf(comb[index] - row_max);
      comb[index] = value;
      row_sum += value;
    }
    for (uint32_t col = 0; col < hc_mult; ++col) {
      const uint64_t index = static_cast<uint64_t>(row) * hc_mult + col;
      comb[index] = comb[index] / row_sum + hc_sinkhorn_eps;
    }
  }

  for (uint32_t col = 0; col < hc_mult; ++col) {
    float col_sum = 0.0f;
    for (uint32_t row = 0; row < hc_mult; ++row) {
      col_sum += comb[static_cast<uint64_t>(row) * hc_mult + col];
    }
    for (uint32_t row = 0; row < hc_mult; ++row) {
      comb[static_cast<uint64_t>(row) * hc_mult + col] /=
          col_sum + hc_sinkhorn_eps;
    }
  }
  for (uint32_t iter = 1; iter < layout.deepseek_hc_sinkhorn_iters; ++iter) {
    for (uint32_t row = 0; row < hc_mult; ++row) {
      float row_sum = 0.0f;
      for (uint32_t col = 0; col < hc_mult; ++col) {
        row_sum += comb[static_cast<uint64_t>(row) * hc_mult + col];
      }
      for (uint32_t col = 0; col < hc_mult; ++col) {
        comb[static_cast<uint64_t>(row) * hc_mult + col] /=
            row_sum + hc_sinkhorn_eps;
      }
    }
    for (uint32_t col = 0; col < hc_mult; ++col) {
      float col_sum = 0.0f;
      for (uint32_t row = 0; row < hc_mult; ++row) {
        col_sum += comb[static_cast<uint64_t>(row) * hc_mult + col];
      }
      for (uint32_t row = 0; row < hc_mult; ++row) {
        comb[static_cast<uint64_t>(row) * hc_mult + col] /=
            col_sum + hc_sinkhorn_eps;
      }
    }
  }

  const uint64_t comb_offset =
      static_cast<uint64_t>(position) * hc_mult * hc_mult;
  for (uint32_t index = 0; index < hc_mult * hc_mult; ++index) {
    deepseek_mhc_comb_mix[comb_offset + index] = comb[index];
  }

  float layer_input_sumsq = 0.0f;
  for (uint32_t dim = 0; dim < hidden; ++dim) {
    float value = 0.0f;
    for (uint32_t channel = 0; channel < hc_mult; ++channel) {
      value += pre_mix[channel] *
               deepseek_mhc_residual
                   [token_residual_offset +
                    static_cast<uint64_t>(channel) * hidden + dim];
    }
    const float rounded =
        deepseek_session_bf16_bits_to_f32(deepseek_session_f32_to_bf16_bits(
            value));
    temp_layer_input[dim] = rounded;
    layer_input_sumsq += rounded * rounded;
  }
  const float norm_scale =
      rsqrtf(layer_input_sumsq / static_cast<float>(hidden) + rms_eps);
  for (uint32_t dim = 0; dim < hidden; ++dim) {
    const float normed =
        temp_layer_input[dim] * norm_scale *
        encoded_to_f32(arena[norm_weight_offset + dim], kDTypeBF16);
    temp_layer_input[dim] = normed;
    projection_input[dim] = f32_to_encoded(normed, dtype);
  }
}

__device__ void deepseek_session_finish_v4_mhc_head_norm(
    const uint16_t *arena, SequenceArenaLayout arena_layout,
    SequenceLayerLayout layout, uint32_t dtype, uint32_t final_norm_weight_dtype,
    uint32_t hidden, uint32_t position, float rms_eps, const float *layer_output,
    float *deepseek_mhc_residual, float *deepseek_mhc_post_mix,
    float *deepseek_mhc_comb_mix, float *temp_layer_input,
    uint16_t *projection_input) {
  const uint32_t hc_mult = layout.deepseek_hc_mult;
  if (arena == nullptr || layer_output == nullptr ||
      deepseek_mhc_residual == nullptr || deepseek_mhc_post_mix == nullptr ||
      deepseek_mhc_comb_mix == nullptr || temp_layer_input == nullptr ||
      projection_input == nullptr || hidden == 0 || hc_mult == 0 ||
      hc_mult > kDeepSeekSessionMaxMhcHcMult ||
      arena_layout.deepseek_hc_head_base == kMissingOffset ||
      arena_layout.deepseek_hc_head_fn == kMissingOffset ||
      arena_layout.deepseek_hc_head_scale == kMissingOffset ||
      arena_layout.final_norm == kMissingOffset) {
    return;
  }

  const uint64_t hc_hidden =
      static_cast<uint64_t>(hc_mult) * static_cast<uint64_t>(hidden);
  const uint64_t token_residual_offset =
      static_cast<uint64_t>(position) * hc_hidden;
  const uint64_t token_post_offset =
      static_cast<uint64_t>(position) * hc_mult;
  const uint64_t token_comb_offset =
      static_cast<uint64_t>(position) * hc_mult * hc_mult;

  float sqrsum = 0.0f;
  for (uint32_t dim = 0; dim < hidden; ++dim) {
    float old_values[kDeepSeekSessionMaxMhcHcMult];
    float new_values[kDeepSeekSessionMaxMhcHcMult];
    for (uint32_t channel = 0; channel < hc_mult; ++channel) {
      old_values[channel] =
          deepseek_mhc_residual
              [token_residual_offset + static_cast<uint64_t>(channel) * hidden +
               dim];
    }
    for (uint32_t out_channel = 0; out_channel < hc_mult; ++out_channel) {
      float value =
          deepseek_mhc_post_mix[token_post_offset + out_channel] *
          layer_output[dim];
      for (uint32_t in_channel = 0; in_channel < hc_mult; ++in_channel) {
        value += deepseek_mhc_comb_mix
                     [token_comb_offset +
                      static_cast<uint64_t>(in_channel) * hc_mult +
                      out_channel] *
                 old_values[in_channel];
      }
      new_values[out_channel] =
          deepseek_session_bf16_bits_to_f32(deepseek_session_f32_to_bf16_bits(
              value));
    }
    for (uint32_t channel = 0; channel < hc_mult; ++channel) {
      const float value = new_values[channel];
      deepseek_mhc_residual
          [token_residual_offset + static_cast<uint64_t>(channel) * hidden +
           dim] = value;
      sqrsum += value * value;
    }
  }
  if (!(sqrsum > 0.0f) || !isfinite(sqrsum)) {
    return;
  }

  const float head_rms_scale =
      rsqrtf(sqrsum / static_cast<float>(hc_hidden) + rms_eps);
  const float hc_head_scale =
      f32_from_u16_slots(arena + arena_layout.deepseek_hc_head_scale, 0);
  float gates[kDeepSeekSessionMaxMhcHcMult];
  for (uint32_t channel = 0; channel < hc_mult; ++channel) {
    float mix = 0.0f;
    const uint64_t row_offset = static_cast<uint64_t>(channel) * hc_hidden;
    for (uint32_t in_channel = 0; in_channel < hc_mult; ++in_channel) {
      const uint64_t channel_offset =
          static_cast<uint64_t>(in_channel) * hidden;
      for (uint32_t dim = 0; dim < hidden; ++dim) {
        mix += f32_from_u16_slots(arena + arena_layout.deepseek_hc_head_fn,
                                  row_offset + channel_offset + dim) *
               deepseek_mhc_residual[token_residual_offset + channel_offset +
                                      dim];
      }
    }
    mix *= head_rms_scale;
    gates[channel] =
        sigmoid(mix * hc_head_scale +
                f32_from_u16_slots(arena + arena_layout.deepseek_hc_head_base,
                                   channel)) +
        layout.deepseek_hc_eps;
  }

  float dense_sumsq = 0.0f;
  for (uint32_t dim = 0; dim < hidden; ++dim) {
    float value = 0.0f;
    for (uint32_t channel = 0; channel < hc_mult; ++channel) {
      value += gates[channel] *
               deepseek_mhc_residual
                   [token_residual_offset +
                    static_cast<uint64_t>(channel) * hidden + dim];
    }
    const float rounded =
        deepseek_session_bf16_bits_to_f32(deepseek_session_f32_to_bf16_bits(
            value));
    temp_layer_input[dim] = rounded;
    dense_sumsq += rounded * rounded;
  }
  const float final_norm_scale =
      rsqrtf(dense_sumsq / static_cast<float>(hidden) + rms_eps);
  const uint16_t *norm_weight = arena + arena_layout.final_norm;
  if (final_norm_weight_dtype == kDTypeF32) {
    for (uint32_t dim = 0; dim < hidden; ++dim) {
      const float normed =
          temp_layer_input[dim] * final_norm_scale *
          f32_weight_to_f32_unaligned(norm_weight, dim);
      projection_input[dim] = f32_to_encoded(normed, dtype);
    }
  } else {
    for (uint32_t dim = 0; dim < hidden; ++dim) {
      const float normed =
          temp_layer_input[dim] * final_norm_scale *
          encoded_to_f32(norm_weight[dim], final_norm_weight_dtype);
      projection_input[dim] = f32_to_encoded(normed, dtype);
    }
  }
}

__global__ void hf_deepseek_v4_attn_mhc_pre_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t layer_index, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input, float *deepseek_mhc_residual,
    float *deepseek_mhc_post_mix, float *deepseek_mhc_comb_mix) {
  if (threadIdx.x != 0 ||
      (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, heads * head_dim, head_dim,
                         intermediate);
  deepseek_session_apply_v4_mhc_pre_state(
      arena, layout, dtype, hidden, position, rms_eps, s.input,
      layer_index == 0 ? 1u : 0u, layout.deepseek_hc_attn_base,
      layout.deepseek_hc_attn_fn, layout.deepseek_hc_attn_scale,
      layout.rms_attn, deepseek_mhc_residual, deepseek_mhc_post_mix,
      deepseek_mhc_comb_mix, s.mlp_norm, projection_input);
}
