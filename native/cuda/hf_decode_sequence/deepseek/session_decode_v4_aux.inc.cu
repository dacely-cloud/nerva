cudaError_t launch_deepseek_v4_compressor_state_and_kv(
    NervaCudaHfDecodeSequenceSession *session, const SequenceLayerLayout &layout,
    uint32_t layer_index, uint32_t max_steps, float layer_rope_theta,
    cudaStream_t stream) {
  const uint32_t coff = layout.deepseek_compress_ratio == 4 ? 2u : 1u;
  const uint32_t state_width = coff * session->head_dim;
  if (state_width == 0) {
    return cudaSuccess;
  }
  hf_deepseek_v4_compressor_state_kernel<<<state_width, kDecodeThreads, 0,
                                            stream>>>(
      session->device_arena, layout, session->dtype, session->hidden,
      session->head_dim, session->device_step, max_steps,
      session->device_projection_input, session->kv_block_count,
      session->device_kv_block_table, session->device_deepseek_compressor_state,
      deepseek_v4_compressor_state_layer_offset_bytes(session, layer_index),
      session->device_deepseek_runtime_counters);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  hf_deepseek_v4_compressed_kv_write_kernel<<<1, kDecodeThreads, 0, stream>>>(
      session->device_arena, layout, session->head_dim, session->device_step,
      max_steps, session->rms_eps, layer_rope_theta,
      session->device_deepseek_compressor_state,
      deepseek_v4_compressor_state_layer_offset_bytes(session, layer_index),
      session->device_deepseek_compressed_kv,
      deepseek_v4_main_compressed_kv_layer_offset_bytes(session, layer_index),
      deepseek_v4_compressed_kv_block_count(session, layout),
      session->kv_block_count, session->device_kv_block_table,
      session->device_deepseek_runtime_counters);
  return cudaGetLastError();
}

cudaError_t launch_deepseek_v4_indexer_state_and_kv(
    NervaCudaHfDecodeSequenceSession *session, const SequenceLayerLayout &layout,
    uint32_t layer_index, uint32_t max_steps, float layer_rope_theta,
    cudaStream_t stream) {
  const uint32_t coff = layout.deepseek_compress_ratio == 4 ? 2u : 1u;
  const uint32_t state_width = coff * layout.deepseek_index_head_dim;
  if (state_width == 0) {
    return cudaSuccess;
  }
  hf_deepseek_v4_indexer_state_kernel<<<state_width, kDecodeThreads, 0,
                                         stream>>>(
      session->device_arena, layout, session->dtype, session->hidden,
      session->device_step, max_steps, session->device_projection_input,
      session->kv_block_count, session->device_kv_block_table,
      session->device_deepseek_indexer_state,
      deepseek_v4_indexer_state_layer_offset_bytes(session, layer_index),
      session->device_deepseek_runtime_counters);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  hf_deepseek_v4_indexer_kv_write_kernel<<<1, kDecodeThreads, 0, stream>>>(
      session->device_arena, layout, session->device_step, max_steps,
      session->rms_eps, layer_rope_theta,
      session->device_deepseek_indexer_state,
      deepseek_v4_indexer_state_layer_offset_bytes(session, layer_index),
      session->device_deepseek_indexer_kv,
      deepseek_v4_indexer_kv_layer_offset_bytes(session, layer_index),
      deepseek_v4_compressed_kv_block_count(session, layout),
      session->kv_block_count, session->device_kv_block_table,
      session->device_deepseek_runtime_counters);
  return cudaGetLastError();
}
