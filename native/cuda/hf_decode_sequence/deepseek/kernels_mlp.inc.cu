__global__ void hf_deepseek_ff_encode_kernel(
    SequenceLayerLayout layout, uint32_t dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t active_intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch, uint16_t *projection_input) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint32_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = blockDim.x * gridDim.x;
  const uint32_t limit =
      active_intermediate < intermediate ? active_intermediate : intermediate;
  for (uint32_t index = start; index < limit; index += stride) {
    const float value = deepseek_swiglu(
        f32_to_model_dtype(s.gate[index], dtype),
        f32_to_model_dtype(s.up[index], dtype), layout.deepseek_swiglu_limit);
    s.ff[index] = value;
    projection_input[index] = f32_to_encoded(value, dtype);
  }
}

// Attention-residual add + MLP norm with the decode path's rounding: the
// attention output projection is rounded to the model dtype before the
// residual add, mirroring hf_deepseek_residual_mlp_norm_encode_kernel.
__global__ void hf_deepseek_prefill_mlp_norm_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t norm_weight_dtype, uint32_t hidden, uint32_t chunk_start,
    uint32_t chunk_tokens, float rms_eps, const uint16_t *hidden_in,
    float *attn_projection, uint16_t *norm_out) {
  const uint32_t local_token = blockIdx.x;
  if (local_token >= chunk_tokens) {
    return;
  }
  const uint64_t global_token = chunk_start + local_token;
  const uint16_t *input = hidden_in + global_token * hidden;
  float *residual =
      attn_projection + static_cast<uint64_t>(local_token) * hidden;
  uint16_t *out = norm_out + static_cast<uint64_t>(local_token) * hidden;
  float mean_square = 0.0f;
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    const float value = f32_to_model_dtype(residual[index], dtype) +
                        encoded_to_f32(input[index], dtype);
    residual[index] = value;
    mean_square += value * value;
  }
  mean_square = block_sum(mean_square);
  const float scale =
      rsqrtf(mean_square / static_cast<float>(hidden) + rms_eps);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    out[index] = f32_to_encoded(
        residual[index] * scale *
            norm_weight_to_f32(arena + layout.rms_mlp, index,
                               norm_weight_dtype),
        dtype);
  }
}

__global__ void hf_deepseek_prefill_ff_split_kernel(
    SequenceLayerLayout layout, uint32_t dtype, uint32_t intermediate,
    uint32_t chunk_tokens, const float *gate, const float *up,
    uint16_t *ff_out) {
  const uint64_t total = static_cast<uint64_t>(chunk_tokens) * intermediate;
  const uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t cursor = index; cursor < total; cursor += stride) {
    const uint64_t token = cursor / intermediate;
    const uint64_t offset = cursor - token * intermediate;
    const float value = deepseek_swiglu(
        f32_to_model_dtype(gate[cursor], dtype),
        f32_to_model_dtype(up[cursor], dtype), layout.deepseek_swiglu_limit);
    ff_out[token * intermediate + offset] = f32_to_encoded(value, dtype);
  }
}

__global__ void hf_deepseek_accumulate_residual_down_kernel(
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    s.down[index] += s.residual[index];
    s.residual[index] = 0.0f;
  }
}
