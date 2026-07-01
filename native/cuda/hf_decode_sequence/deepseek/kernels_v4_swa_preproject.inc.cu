__global__ void hf_deepseek_v4_q_a_norm_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t hidden,
    uint32_t heads, uint32_t head_dim, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps,
    float *scratch) {
  if ((step_cursor != nullptr && *step_cursor >= max_steps) ||
      arena == nullptr || scratch == nullptr || hidden == 0 ||
      heads == 0 || head_dim == 0 || layout.q_norm == kMissingOffset) {
    return;
  }
  const uint32_t q_lora_rank = layout.deepseek_q_lora_rank;
  const uint32_t attention_hidden = heads * head_dim;
  if (q_lora_rank == 0 || q_lora_rank > attention_hidden) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, head_dim,
                         intermediate);
  float norm_sum = 0.0f;
  for (uint32_t index = threadIdx.x; index < q_lora_rank;
       index += blockDim.x) {
    const float value = s.q[index];
    norm_sum += value * value;
  }
  norm_sum = block_sum(norm_sum);
  const float norm_scale =
      rsqrtf(norm_sum / static_cast<float>(q_lora_rank) + rms_eps);
  for (uint32_t index = threadIdx.x; index < q_lora_rank;
       index += blockDim.x) {
    s.q_gate[index] =
        s.q[index] * norm_scale *
        encoded_to_f32(arena[layout.q_norm + index], kDTypeBF16);
  }
}

__global__ void hf_deepseek_v4_finalize_preprojected_qk_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rms_eps, float rope_theta, float *scratch) {
  if ((step_cursor != nullptr && *step_cursor >= max_steps) ||
      arena == nullptr || scratch == nullptr || hidden == 0 ||
      heads == 0 || head_dim == 0 || layout.k_norm == kMissingOffset) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  if (qk_nope + qk_rope != head_dim) {
    return;
  }

  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, heads * head_dim, head_dim,
                         intermediate);
  const uint32_t rope_half = qk_rope / 2u;
  if (blockIdx.x == 0) {
    float norm_sum = 0.0f;
    for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      const float value = s.k[dim];
      norm_sum += value * value;
    }
    norm_sum = block_sum(norm_sum);
    const float norm_scale =
        rsqrtf(norm_sum / static_cast<float>(head_dim) + rms_eps);
    for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      s.k[dim] *= norm_scale *
                  encoded_to_f32(arena[layout.k_norm + dim], kDTypeBF16);
    }
    __syncthreads();
    if (rope_half != 0) {
      for (uint32_t offset = threadIdx.x; offset < rope_half;
           offset += blockDim.x) {
        const uint32_t left = qk_nope + offset;
        const uint32_t right = left + rope_half;
        const float left_value = s.k[left];
        const float right_value = s.k[right];
        s.k[left] = deepseek_rope_value_serial(
            left_value, right_value, offset, qk_rope, position, rope_theta,
            false, layout);
        s.k[right] = deepseek_rope_value_serial(
            left_value, right_value, offset, qk_rope, position, rope_theta,
            true, layout);
      }
    }
    return;
  }

  const uint32_t head = blockIdx.x - 1u;
  if (head >= heads) {
    return;
  }
  const uint32_t head_start = head * head_dim;
  float norm_sum = 0.0f;
  for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const float value = s.q[head_start + dim];
    norm_sum += value * value;
  }
  norm_sum = block_sum(norm_sum);
  const float norm_scale =
      rsqrtf(norm_sum / static_cast<float>(head_dim) + rms_eps);
  for (uint32_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    s.q[head_start + dim] *= norm_scale;
  }
  __syncthreads();
  if (rope_half != 0) {
    for (uint32_t offset = threadIdx.x; offset < rope_half;
         offset += blockDim.x) {
      const uint32_t left = head_start + qk_nope + offset;
      const uint32_t right = left + rope_half;
      const float left_value = s.q[left];
      const float right_value = s.q[right];
      s.q[left] = deepseek_rope_value_serial(
          left_value, right_value, offset, qk_rope, position, rope_theta,
          false, layout);
      s.q[right] = deepseek_rope_value_serial(
          left_value, right_value, offset, qk_rope, position, rope_theta,
          true, layout);
    }
  }
}

__global__ void hf_deepseek_v4_ffn_mhc_pre_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t heads, uint32_t head_dim,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float rms_eps, float *scratch, uint16_t *projection_input,
    float *deepseek_mhc_residual, float *deepseek_mhc_post_mix,
    float *deepseek_mhc_comb_mix) {
  if ((step_cursor != nullptr && *step_cursor >= max_steps) ||
      arena == nullptr || scratch == nullptr || projection_input == nullptr ||
      hidden == 0 || heads == 0 || head_dim == 0) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, heads * head_dim, head_dim,
                         intermediate);
  if (threadIdx.x == 0) {
    deepseek_session_apply_v4_mhc_pre_state(
        arena, layout, dtype, hidden, position, rms_eps, s.residual, 0u,
        layout.deepseek_hc_ffn_base, layout.deepseek_hc_ffn_fn,
        layout.deepseek_hc_ffn_scale, layout.rms_mlp, deepseek_mhc_residual,
        deepseek_mhc_post_mix, deepseek_mhc_comb_mix, s.mlp_norm,
        projection_input);
  }
  __syncthreads();
  for (uint32_t row = threadIdx.x; row < hidden; row += blockDim.x) {
    s.residual[row] = 0.0f;
  }
}
