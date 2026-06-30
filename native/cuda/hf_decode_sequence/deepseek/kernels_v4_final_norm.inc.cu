__global__ void hf_deepseek_v4_finish_final_norm_encode_kernel(
    uint16_t *arena, SequenceArenaLayout arena_layout, SequenceLayerLayout layout,
    uint32_t dtype, uint32_t final_norm_weight_dtype, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float rms_eps, float *scratch,
    uint16_t *projection_input, float *deepseek_mhc_residual,
    float *deepseek_mhc_post_mix, float *deepseek_mhc_comb_mix) {
  if (threadIdx.x != 0 || (step_cursor != nullptr && *step_cursor >= max_steps)) {
    return;
  }
  const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  deepseek_session_finish_v4_mhc_head_norm(
      arena, arena_layout, layout, dtype, final_norm_weight_dtype, hidden,
      position, rms_eps, s.down, deepseek_mhc_residual,
      deepseek_mhc_post_mix, deepseek_mhc_comb_mix, s.input, projection_input);
}
