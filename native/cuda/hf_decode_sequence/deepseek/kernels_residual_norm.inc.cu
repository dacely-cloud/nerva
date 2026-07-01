__global__ void hf_deepseek_residual_mlp_norm_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t norm_weight_dtype, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    s.residual[index] =
        f32_to_model_dtype(s.residual[index], dtype) + s.input[index];
  }
  __syncthreads();
  rms_norm_to_encoded_with_weight_dtype(s.residual, arena + layout.rms_mlp,
                                        hidden, norm_weight_dtype, dtype,
                                        rms_eps, projection_input);
}
