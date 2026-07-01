uint32_t deepseek_norm_weight_dtype(const SequenceLayerLayout &layout) {
  return layout.deepseek_mode == kDeepSeekModeV32MlaIndexer ? kDTypeF32
                                                            : kDTypeBF16;
}

uint32_t layer_norm_weight_dtype(const SequenceLayerLayout &layout,
                                 uint32_t dtype) {
  if (layout_is_deepseek_v3_mla(layout)) {
    return deepseek_norm_weight_dtype(layout);
  }
  if (layout.attention_kind == kAttentionKindDeepSeekMla) {
    return kDTypeBF16;
  }
  return dtype;
}

const uint8_t *deepseek_fp8_ptr(uint16_t *arena, uint64_t offset) {
  return reinterpret_cast<const uint8_t *>(arena + offset);
}

const float *deepseek_scale_ptr(uint16_t *arena, uint64_t offset) {
  return reinterpret_cast<const float *>(arena + offset);
}

uint64_t deepseek_fp8_slots_u64(uint64_t rows, uint64_t cols) {
  return (rows * cols + 1u) / 2u;
}

uint64_t deepseek_f32_scale_offset(uint64_t matrix_offset, uint64_t rows,
                                   uint64_t cols) {
  return matrix_offset + deepseek_fp8_slots_u64(rows, cols);
}

cudaError_t launch_deepseek_v3_mla_projection_step(
    NervaCudaHfDecodeSequenceSession *session, const SequenceLayerLayout &layout,
    uint32_t layer_index, uint32_t max_steps) {
  if (!layout_is_deepseek_v3_mla(layout) ||
      layout.w_q == kMissingOffset ||
      layout.deepseek_q_a_scale == kMissingOffset ||
      layout.q_norm == kMissingOffset ||
      layout.deepseek_q_b == kMissingOffset ||
      layout.deepseek_q_b_scale == kMissingOffset ||
      layout.w_k == kMissingOffset ||
      layout.deepseek_kv_a_scale == kMissingOffset ||
      layout.k_norm == kMissingOffset ||
      layout.w_v == kMissingOffset ||
      layout.deepseek_kv_b_scale == kMissingOffset ||
      layout.w_o == kMissingOffset ||
      layout.deepseek_o_a_scale == kMissingOffset) {
    return cudaErrorInvalidValue;
  }

  const uint32_t q_lora_rank = layout.deepseek_q_lora_rank;
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t v_head = layout.deepseek_v_head_dim;
  if (q_lora_rank == 0 || kv_lora_rank == 0 || qk_nope == 0 ||
      qk_rope == 0 || v_head == 0) {
    return cudaErrorInvalidValue;
  }
  const uint32_t qk_head_dim = qk_nope + qk_rope;
  if (qk_head_dim == 0 || session->head_dim != qk_head_dim ||
      session->heads == 0) {
    return cudaErrorInvalidValue;
  }
  const uint32_t q_rows = session->heads * qk_head_dim;
  const uint32_t kv_a_rows = kv_lora_rank + qk_rope;
  const uint32_t value_rows = session->heads * v_head;
  const uint32_t attention_rows = static_cast<uint32_t>(
      layer_attention_workspace_rows(layout, session->heads * session->head_dim));
  const uint32_t kv_cache_width = static_cast<uint32_t>(
      layout_deepseek_v3_kv_cache_width(layout,
                                        session->kv_heads * session->head_dim));
  LayerScratch scratch =
      layer_scratch_ptrs(session->device_scratch, session->hidden,
                         attention_rows, kv_cache_width, session->intermediate);
  constexpr uint32_t block_rows = 128;
  constexpr uint32_t block_cols = 128;

  cudaError_t err = launch_deepseek_fp8_f32_scale_encoded_matvec(
      session->stream, deepseek_fp8_ptr(session->device_arena, layout.w_q),
      deepseek_scale_ptr(session->device_arena, layout.deepseek_q_a_scale),
      session->device_projection_input, session->dtype, q_lora_rank,
      session->hidden, block_rows, block_cols, scratch.q);
  if (err != cudaSuccess) return err;

  err = launch_deepseek_fp8_f32_scale_encoded_matvec(
      session->stream, deepseek_fp8_ptr(session->device_arena, layout.w_k),
      deepseek_scale_ptr(session->device_arena, layout.deepseek_kv_a_scale),
      session->device_projection_input, session->dtype, kv_a_rows,
      session->hidden, block_rows, block_cols, scratch.k);
  if (err != cudaSuccess) return err;

  hf_deepseek_v32_indexer_kv_encode_kernel<<<1, 1, 0, session->stream>>>(
      session->device_arena, layout, session->dtype, session->hidden,
      session->device_step, max_steps, session->rope_theta,
      session->device_projection_input, session->device_deepseek_indexer_kv,
      deepseek_v32_indexer_kv_layer_offset_bytes(session, layer_index),
      deepseek_v32_indexer_kv_block_count(session, layout),
      session->device_deepseek_runtime_counters);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  hf_deepseek_v32_indexer_weight_state_kernel<<<1, 1, 0,
                                                 session->stream>>>(
      session->device_arena, layout, session->dtype, session->hidden,
      session->device_step, max_steps, session->device_projection_input,
      reinterpret_cast<uint8_t *>(session->device_deepseek_indexer_state),
      deepseek_v32_indexer_query_state_layer_offset_bytes(session,
                                                          layer_index));
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  hf_decode_rms_norm_f32_to_encoded_kernel<<<1, kDecodeNormThreads, 0,
                                             session->stream>>>(
      session->device_arena, layout.q_norm, scratch.q,
      deepseek_norm_weight_dtype(layout), session->dtype, q_lora_rank,
      session->device_step, max_steps, session->rms_eps,
      session->device_projection_input);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  hf_deepseek_v32_indexer_query_state_kernel<<<1, 1, 0, session->stream>>>(
      session->device_arena, layout, session->dtype, q_lora_rank,
      session->device_step, max_steps, session->rope_theta,
      session->device_projection_input,
      reinterpret_cast<uint8_t *>(session->device_deepseek_indexer_state),
      deepseek_v32_indexer_query_state_layer_offset_bytes(session,
                                                          layer_index),
      session->device_deepseek_runtime_counters);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  err = launch_deepseek_fp8_f32_scale_encoded_matvec(
      session->stream,
      deepseek_fp8_ptr(session->device_arena, layout.deepseek_q_b),
      deepseek_scale_ptr(session->device_arena, layout.deepseek_q_b_scale),
      session->device_projection_input, session->dtype, q_rows, q_lora_rank,
      block_rows, block_cols, scratch.q);
  if (err != cudaSuccess) return err;

  hf_decode_rms_norm_f32_to_encoded_kernel<<<1, kDecodeNormThreads, 0,
                                             session->stream>>>(
      session->device_arena, layout.k_norm, scratch.k,
      deepseek_norm_weight_dtype(layout), session->dtype, kv_lora_rank,
      session->device_step, max_steps, session->rms_eps,
      session->device_projection_input);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  hf_deepseek_v3_mla_attention_encode_kernel<<<1, kDecodeThreads, 0,
                                                session->stream>>>(
      session->device_arena, layout, layer_index, session->dtype, session->heads,
      session->device_step, max_steps, session->rope_theta, scratch.q,
      scratch.k, scratch.v, session->device_projection_input,
      session->device_kv_keys, session->kv_block_count,
      session->device_kv_block_table, session->device_projection_input,
      session->device_deepseek_v32_mla_kv,
      deepseek_v32_mla_kv_layer_offset_bytes(session, layer_index),
      deepseek_v32_mla_kv_block_count(session, layout),
      reinterpret_cast<const uint8_t *>(session->device_deepseek_indexer_state),
      deepseek_v32_indexer_query_state_layer_offset_bytes(session,
                                                          layer_index),
      session->device_deepseek_indexer_kv,
      deepseek_v32_indexer_kv_layer_offset_bytes(session, layer_index),
      deepseek_v32_indexer_kv_block_count(session, layout),
      session->device_deepseek_runtime_counters);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  err = launch_deepseek_fp8_f32_scale_encoded_matvec(
      session->stream, deepseek_fp8_ptr(session->device_arena, layout.w_o),
      deepseek_scale_ptr(session->device_arena, layout.deepseek_o_a_scale),
      session->device_projection_input, session->dtype, session->hidden,
      value_rows, block_rows, block_cols, scratch.residual);
  if (err != cudaSuccess) return err;

  hf_deepseek_residual_mlp_norm_encode_kernel<<<1, kDecodeNormThreads, 0,
                                                session->stream>>>(
      session->device_arena, layout, session->dtype,
      deepseek_norm_weight_dtype(layout), session->hidden, attention_rows,
      kv_cache_width, session->intermediate, session->device_step, max_steps,
      session->rms_eps, session->device_scratch,
      session->device_projection_input);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  if (layout.mlp_kind == kMlpKindSparseMoe) {
    hf_deepseek_v3_sparse_moe_encode_kernel<<<1, kDecodeThreads, 0,
                                              session->stream>>>(
        session->device_arena, layout, session->dtype, session->hidden,
        attention_rows, kv_cache_width, session->intermediate,
        session->device_step, max_steps, session->device_scratch,
        session->device_projection_input,
        session->device_deepseek_runtime_counters);
    return cudaGetLastError();
  }
  if (layout.w_gate == kMissingOffset || layout.w_up == kMissingOffset ||
      layout.w_down == kMissingOffset) {
    return cudaErrorInvalidValue;
  }

  const uint64_t gate_scale =
      deepseek_f32_scale_offset(layout.w_gate, session->intermediate,
                                session->hidden);
  const uint64_t up_scale =
      deepseek_f32_scale_offset(layout.w_up, session->intermediate,
                                session->hidden);
  const uint64_t down_scale =
      deepseek_f32_scale_offset(layout.w_down, session->hidden,
                                session->intermediate);
  err = launch_deepseek_fp8_f32_scale_encoded_matvec(
      session->stream, deepseek_fp8_ptr(session->device_arena, layout.w_gate),
      deepseek_scale_ptr(session->device_arena, gate_scale),
      session->device_projection_input, session->dtype, session->intermediate,
      session->hidden, block_rows, block_cols, scratch.gate);
  if (err != cudaSuccess) return err;
  err = launch_deepseek_fp8_f32_scale_encoded_matvec(
      session->stream, deepseek_fp8_ptr(session->device_arena, layout.w_up),
      deepseek_scale_ptr(session->device_arena, up_scale),
      session->device_projection_input, session->dtype, session->intermediate,
      session->hidden, block_rows, block_cols, scratch.up);
  if (err != cudaSuccess) return err;
  {
    const uint32_t ff_blocks =
        (session->intermediate + kDecodeThreads - 1) / kDecodeThreads;
    hf_layer_ff_encode_kernel<<<ff_blocks, kDecodeThreads, 0,
                                session->stream>>>(
        session->dtype, session->hidden, attention_rows, kv_cache_width,
        session->intermediate, session->device_step, max_steps,
        session->device_scratch, session->device_projection_input);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }
  err = launch_deepseek_fp8_f32_scale_encoded_matvec(
      session->stream, deepseek_fp8_ptr(session->device_arena, layout.w_down),
      deepseek_scale_ptr(session->device_arena, down_scale),
      session->device_projection_input, session->dtype, session->hidden,
      session->intermediate, block_rows, block_cols, scratch.down);
  if (err != cudaSuccess) return err;

  return cudaSuccess;
}

cudaError_t launch_deepseek_v4_swa_dense_projection_step(
    NervaCudaHfDecodeSequenceSession *session, const SequenceLayerLayout &layout,
    uint32_t layer_index, uint32_t max_steps, uint32_t prompt_token_count) {
  if (!layout_is_deepseek_v4_native(layout)) {
    return cudaErrorInvalidValue;
  }
  if (layout.w_q == kMissingOffset ||
      layout.deepseek_q_a_scale == kMissingOffset ||
      layout.q_norm == kMissingOffset ||
      layout.deepseek_q_b == kMissingOffset ||
      layout.deepseek_q_b_scale == kMissingOffset ||
      layout.w_k == kMissingOffset ||
      layout.deepseek_kv_a_scale == kMissingOffset ||
      layout.k_norm == kMissingOffset) {
    return cudaErrorInvalidValue;
  }
  const uint32_t q_lora_rank = layout.deepseek_q_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  if (q_lora_rank == 0 || qk_rope == 0 ||
      qk_nope + qk_rope != session->head_dim || session->heads == 0) {
    return cudaErrorInvalidValue;
  }
  const uint32_t attention_hidden = session->heads * session->head_dim;
  if (q_lora_rank > attention_hidden) {
    return cudaErrorInvalidValue;
  }
  LayerScratch scratch =
      layer_scratch_ptrs(session->device_scratch, session->hidden,
                         attention_hidden, session->head_dim,
                         session->intermediate);
  constexpr uint32_t block_rows = 128;
  constexpr uint32_t block_cols = 128;

  hf_deepseek_v4_attn_mhc_pre_kernel<<<1, 1, 0, session->stream>>>(
      session->device_arena, layout, session->dtype, session->hidden,
      session->heads, session->head_dim, session->intermediate, layer_index,
      session->device_step, max_steps, session->rms_eps,
      session->device_scratch, session->device_projection_input,
      session->device_deepseek_mhc_residual,
      session->device_deepseek_mhc_post_mix,
      session->device_deepseek_mhc_comb_mix);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  err = launch_deepseek_fp8_e8m0_scale_encoded_matvec(
      session->stream, deepseek_fp8_ptr(session->device_arena, layout.w_q),
      deepseek_fp8_ptr(session->device_arena, layout.deepseek_q_a_scale),
      session->device_projection_input, session->dtype, q_lora_rank,
      session->hidden, block_rows, block_cols, scratch.q);
  if (err != cudaSuccess) return err;

  err = launch_deepseek_fp8_e8m0_scale_encoded_matvec(
      session->stream, deepseek_fp8_ptr(session->device_arena, layout.w_k),
      deepseek_fp8_ptr(session->device_arena, layout.deepseek_kv_a_scale),
      session->device_projection_input, session->dtype, session->head_dim,
      session->hidden, block_rows, block_cols, scratch.k);
  if (err != cudaSuccess) return err;

  hf_deepseek_v4_q_a_norm_kernel<<<1, kDecodeNormThreads, 0,
                                   session->stream>>>(
      session->device_arena, layout, session->hidden, session->heads,
      session->head_dim, session->intermediate, session->device_step,
      max_steps, session->rms_eps, session->device_scratch);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  err = launch_deepseek_fp8_e8m0_scale_matvec(
      session->stream,
      deepseek_fp8_ptr(session->device_arena, layout.deepseek_q_b),
      deepseek_fp8_ptr(session->device_arena, layout.deepseek_q_b_scale),
      scratch.q_gate, attention_hidden, q_lora_rank, block_rows, block_cols,
      scratch.q);
  if (err != cudaSuccess) return err;

  hf_deepseek_v4_finalize_preprojected_qk_kernel<<<session->heads + 1u,
                                                    kDecodeThreads, 0,
                                                    session->stream>>>(
      session->device_arena, layout, session->dtype, session->hidden,
      session->heads, session->head_dim, session->intermediate,
      session->device_step, max_steps, session->rms_eps, session->rope_theta,
      session->device_scratch);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  uint32_t precomputed_compressor_state = 0;
  uint32_t precomputed_indexer_state = 0;
  if ((layout.deepseek_mode == kDeepSeekModeV4Compressed ||
       layout.deepseek_mode == kDeepSeekModeV4CompressedIndexer) &&
      layout.deepseek_compress_ratio > 1 &&
      session->device_deepseek_compressor_state != nullptr &&
      layout.deepseek_compressor_wkv != kMissingOffset &&
      layout.deepseek_compressor_wgate != kMissingOffset &&
      layout.deepseek_compressor_ape != kMissingOffset) {
    const uint32_t coff =
        layout.deepseek_compress_ratio == 4 ? 2u : 1u;
    const uint32_t state_width = coff * session->head_dim;
    if (state_width != 0) {
      hf_deepseek_v4_compressor_state_kernel<<<state_width, kDecodeThreads, 0,
                                                session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          session->head_dim, session->device_step, max_steps,
          session->device_projection_input, session->kv_block_count,
          session->device_kv_block_table,
          session->device_deepseek_compressor_state,
          deepseek_v4_compressor_state_layer_offset_bytes(session, layer_index),
          session->device_deepseek_runtime_counters);
      err = cudaGetLastError();
      if (err != cudaSuccess) return err;
      precomputed_compressor_state = 1;
    }
  }
  if (layout.deepseek_mode == kDeepSeekModeV4CompressedIndexer &&
      layout.deepseek_compress_ratio > 1 &&
      layout.deepseek_index_head_dim > 0 &&
      session->device_deepseek_indexer_state != nullptr &&
      layout.deepseek_indexer_compressor_wkv != kMissingOffset &&
      layout.deepseek_indexer_compressor_wgate != kMissingOffset &&
      layout.deepseek_indexer_compressor_ape != kMissingOffset) {
    const uint32_t coff =
        layout.deepseek_compress_ratio == 4 ? 2u : 1u;
    const uint32_t state_width = coff * layout.deepseek_index_head_dim;
    if (state_width != 0) {
      hf_deepseek_v4_indexer_state_kernel<<<state_width, kDecodeThreads, 0,
                                             session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          session->device_step, max_steps, session->device_projection_input,
          session->kv_block_count, session->device_kv_block_table,
          session->device_deepseek_indexer_state,
          deepseek_v4_indexer_state_layer_offset_bytes(session, layer_index),
          session->device_deepseek_runtime_counters);
      err = cudaGetLastError();
      if (err != cudaSuccess) return err;
      precomputed_indexer_state = 1;
    }
  }

  hf_deepseek_v4_swa_dense_layer_kernel<<<1, 1, 0, session->stream>>>(
      session->device_arena, layout, layer_index, session->dtype,
      session->hidden, session->heads, session->head_dim,
      session->intermediate, session->device_step, max_steps,
      session->rms_eps, session->rope_theta, session->device_scratch,
      session->device_kv_keys, session->device_kv_values,
      session->kv_block_count, session->device_kv_block_table,
      session->vocab_size, session->device_prompt_tokens, prompt_token_count,
      session->device_slots,
      session->device_projection_input, session->device_deepseek_swa_kv,
      deepseek_v4_swa_kv_layer_offset_bytes(session, layer_index),
      deepseek_v4_swa_kv_block_count(session, layout),
      session->device_deepseek_compressor_state,
      deepseek_v4_compressor_state_layer_offset_bytes(session, layer_index),
      session->device_deepseek_compressed_kv,
      deepseek_v4_main_compressed_kv_layer_offset_bytes(session, layer_index),
      deepseek_v4_compressed_kv_block_count(session, layout),
      session->device_deepseek_indexer_state,
      deepseek_v4_indexer_state_layer_offset_bytes(session, layer_index),
      session->device_deepseek_indexer_kv,
      deepseek_v4_indexer_kv_layer_offset_bytes(session, layer_index),
      deepseek_v4_compressed_kv_block_count(session, layout),
      session->device_deepseek_mhc_residual,
      session->device_deepseek_mhc_post_mix,
      session->device_deepseek_mhc_comb_mix,
      session->device_deepseek_runtime_counters,
      session->experimental_rt_local_window_tokens, 2u,
      precomputed_compressor_state, precomputed_indexer_state);
  return cudaGetLastError();
}
