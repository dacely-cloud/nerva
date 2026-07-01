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
    const float value =
        deepseek_swiglu(s.gate[index], s.up[index], layout.deepseek_swiglu_limit);
    s.ff[index] = value;
    projection_input[index] = f32_to_encoded(value, dtype);
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
