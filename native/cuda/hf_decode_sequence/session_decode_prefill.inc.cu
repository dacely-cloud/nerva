cudaError_t pack_session_weight_replicas(
    NervaCudaHfDecodeSequenceSession *session) {
  if (!use_cublas_layer_path(session)) {
    return cudaSuccess;
  }
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const PackedProjectionShape shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  hf_pack_qkv_weights_kernel<<<
      static_cast<uint32_t>(shape.qkv_rows * session->layer_count),
      kDecodeThreads, 0, session->stream>>>(
      session->device_qkv_packed, session->device_arena,
      session->device_layouts, session->layer_count, session->hidden,
      attention_hidden, kv_hidden);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  if (session->packed_gate_up_bytes != 0 &&
      session->device_gate_up_packed != nullptr) {
    hf_pack_gate_up_weights_kernel<<<
        static_cast<uint32_t>(shape.gate_up_rows * session->layer_count),
        kDecodeThreads, 0, session->stream>>>(
        session->device_gate_up_packed, session->device_arena,
        session->device_layouts, session->layer_count, session->hidden,
        session->intermediate);
    return cudaGetLastError();
  }
  return cudaSuccess;
}

cudaError_t launch_monolithic_session_step(
    NervaCudaHfDecodeSequenceSession *session, uint32_t max_steps,
    uint32_t prompt_token_count, uint32_t has_eos_token, uint32_t eos_token) {
  hf_decode_sequence_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
      session->device_arena, session->arena_layout, session->device_layouts,
      session->layer_count, session->dtype, session->hidden, session->heads,
      session->kv_heads, session->head_dim, session->intermediate, 0,
      session->device_step, max_steps, session->device_prompt_tokens,
      prompt_token_count, session->rms_eps, session->rope_theta,
      session->device_scratch, session->device_kv_keys,
      session->device_kv_values, session->kv_block_count,
      session->device_kv_block_table,
      session->device_slots, session->device_linear_gdn_conv_state,
      session->device_linear_gdn_recurrent_state);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = final_head_gemv(session->cublas, session->device_arena,
                          session->arena_layout, session->dtype,
                          session->hidden, session->vocab_size, device_logits);
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        session->stream, session->device_step, max_steps, has_eos_token,
        eos_token, device_logits, session->vocab_size, session->device_slots,
        session->active_sampler);
  }
  return err;
}

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

  hf_decode_rms_norm_f32_to_encoded_kernel<<<1, kDecodeNormThreads, 0,
                                             session->stream>>>(
      session->device_arena, layout.q_norm, scratch.q,
      deepseek_norm_weight_dtype(layout), session->dtype, q_lora_rank,
      session->device_step, max_steps, session->rms_eps,
      session->device_projection_input);
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
      session->device_kv_block_table, session->device_projection_input);
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
        session->device_projection_input);
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
  hf_deepseek_v4_swa_dense_layer_kernel<<<1, 1, 0, session->stream>>>(
      session->device_arena, layout, layer_index, session->dtype,
      session->hidden, session->heads, session->head_dim,
      session->intermediate, session->device_step, max_steps,
      session->rms_eps, session->rope_theta, session->device_scratch,
      session->device_kv_keys, session->device_kv_values,
      session->kv_block_count, session->device_kv_block_table,
      session->vocab_size, session->device_prompt_tokens, prompt_token_count,
      session->device_slots,
      session->device_projection_input, session->device_deepseek_compressor_state,
      deepseek_v4_compressor_state_layer_offset_bytes(session, layer_index),
      session->device_deepseek_compressed_kv,
      deepseek_v4_main_compressed_kv_layer_offset_bytes(session, layer_index),
      deepseek_v4_compressed_kv_block_count(session, layout),
      session->device_deepseek_indexer_state,
      deepseek_v4_indexer_state_layer_offset_bytes(session, layer_index),
      session->device_deepseek_indexer_kv,
      deepseek_v4_indexer_kv_layer_offset_bytes(session, layer_index),
      deepseek_v4_compressed_kv_block_count(session, layout),
      session->device_deepseek_runtime_counters);
  return cudaGetLastError();
}

cudaError_t launch_cublas_layer_session_step(
    NervaCudaHfDecodeSequenceSession *session, uint32_t max_steps,
    uint32_t prompt_token_count, uint32_t has_eos_token, uint32_t eos_token,
    uint32_t attention_chunks) {
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const uint32_t decode_head_threads = decode_head_threads_for_session(session);
  cudaError_t err = cudaSuccess;
  uint64_t input_offset = session->arena_layout.input;
  uint64_t output_offset = session->arena_layout.scratch;
  LayerScratch scratch = layer_scratch_ptrs(
      session->device_scratch, session->hidden, attention_hidden, kv_hidden,
      session->intermediate);
  const PackedProjectionShape packed_shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  if (err == cudaSuccess && session->layer_count > 0) {
    const SequenceLayerLayout first_layout = session->host_layouts[0];
    const uint32_t first_attention_hidden = static_cast<uint32_t>(
        layer_attention_workspace_rows(first_layout, attention_hidden));
    const uint32_t first_kv_hidden = static_cast<uint32_t>(
        layout_deepseek_kv_cache_width(first_layout, kv_hidden));
    hf_decode_prepare_first_attn_norm_encode_kernel<<<
        1, kDecodeNormThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, first_layout,
        session->dtype, layer_norm_weight_dtype(first_layout, session->dtype),
        session->hidden, first_attention_hidden, first_kv_hidden,
        session->intermediate, session->device_step, max_steps,
        session->device_prompt_tokens, prompt_token_count, session->device_slots,
        session->rms_eps, session->device_scratch,
        session->device_projection_input);
    err = cudaGetLastError();
  } else if (err == cudaSuccess) {
    hf_decode_prepare_input_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, session->hidden,
        session->device_step, max_steps, session->device_prompt_tokens,
        prompt_token_count, session->device_slots);
    err = cudaGetLastError();
  }
  for (uint32_t layer_index = 0;
       err == cudaSuccess && layer_index < session->layer_count;
       ++layer_index) {
    const SequenceLayerLayout layout = session->host_layouts[layer_index];
    const bool is_deepseek_v3 = layout_is_deepseek_v3_mla(layout);
    const bool is_deepseek_v4_native = layout_is_deepseek_v4_native(layout);
    const uint32_t layer_attention_hidden = static_cast<uint32_t>(
        layer_attention_workspace_rows(layout, attention_hidden));
    const uint32_t layer_kv_hidden = static_cast<uint32_t>(
        layout_deepseek_kv_cache_width(layout, kv_hidden));
    if (is_deepseek_v3) {
      err = launch_deepseek_v3_mla_projection_step(
          session, layout, layer_index, max_steps);
    } else if (is_deepseek_v4_native) {
      err = launch_deepseek_v4_swa_dense_projection_step(
          session, layout, layer_index, max_steps, prompt_token_count);
    } else {
      err = project_encoded_rows(
          session, &session->qkv_plan,
          session->device_qkv_packed +
              packed_shape.qkv_elements_per_layer * layer_index,
          session->device_projection_input,
          static_cast<uint32_t>(packed_shape.qkv_rows), session->hidden, 1,
          session->dtype, 0.0f, scratch.q);
    }
    if (!is_deepseek_v3 && !is_deepseek_v4_native) {
    if (err == cudaSuccess && layout.w_q_gate != kMissingOffset) {
      err = project_encoded_rows(
          session, nullptr, session->device_arena + layout.w_q_gate,
          session->device_projection_input, attention_hidden, session->hidden,
          1, session->dtype, 0.0f, scratch.q_gate);
    }
    if (err == cudaSuccess && attention_chunks == 0) {
      hf_layer_qkv_attention_encode_kernel<<<session->heads, decode_head_threads, 0,
                                             session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
#if NERVA_HAVE_CUDNN_FRONTEND
      const bool use_cudnn_decode_sdpa =
          can_use_cudnn_decode_sdpa(session, attention_chunks);
#else
      const bool use_cudnn_decode_sdpa = false;
#endif
      hf_layer_qkv_prepare_kernel<<<session->heads, decode_head_threads, 0,
                                    session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          use_cudnn_decode_sdpa ? session->device_decode_q : nullptr,
          use_cudnn_decode_sdpa ? session->device_decode_seq_len_q : nullptr,
          use_cudnn_decode_sdpa ? session->device_decode_seq_len_kv : nullptr);
      err = cudaGetLastError();
      const uint32_t query_group = session->heads / session->kv_heads;
      const bool use_shared_warp_gqa =
          query_group == kGroupedGqaHeads &&
          session->heads % session->kv_heads == 0 &&
          session->head_dim <= kSharedWarpGqaHeadDimMax;
      const bool use_grouped_gqa =
          query_group == kGroupedGqaHeads &&
          session->heads % session->kv_heads == 0 &&
          session->head_dim <= kGroupedGqaHeadDimMax;
      const bool use_fused_qk_selector =
          use_shared_warp_gqa &&
          experimental_rt_qk_fused_selector_active(session, attention_chunks);
      if (err == cudaSuccess && !use_fused_qk_selector) {
        err = launch_experimental_rt_qk_page_selector(
            session, layer_index, attention_chunks, max_steps, session->stream);
      }
      bool ran_cudnn_decode_sdpa = false;
#if NERVA_HAVE_CUDNN_FRONTEND
      if (err == cudaSuccess && use_cudnn_decode_sdpa) {
        err = execute_cudnn_decode_sdpa(session, layer_index);
        if (err == cudaSuccess) {
          ran_cudnn_decode_sdpa = true;
        }
      }
#endif
      if (err == cudaSuccess && !ran_cudnn_decode_sdpa) {
        const dim3 grid((use_shared_warp_gqa || use_grouped_gqa)
                            ? session->kv_heads
                            : session->heads,
                        attention_chunks);
        launch_hf_layer_attention_chunk_kernel(
            session->stream, grid, session->dtype, use_shared_warp_gqa,
            use_grouped_gqa, decode_head_threads, layer_index, session->hidden,
            session->heads, session->kv_heads, session->head_dim,
            session->intermediate, session->device_step, max_steps,
            attention_chunks, session->device_scratch,
            session->device_kv_keys, session->device_kv_values,
            session->device_decode_attention_values,
            session->device_decode_attention_m,
            session->device_decode_attention_l, session->kv_block_count,
            session->device_kv_block_table,
            session->experimental_rt_sparse_attention_active == 0
                ? nullptr
                : session->device_experimental_rt_candidate_pages,
            use_fused_qk_selector ? 1u : 0u,
            session->experimental_rt_local_window_tokens,
            session->experimental_rt_sink_tokens);

        err = cudaGetLastError();
      }
      if (err == cudaSuccess && !ran_cudnn_decode_sdpa) {
        const size_t reduce_shared_bytes =
            static_cast<size_t>(attention_chunks) * sizeof(float);
        hf_layer_attention_reduce_kernel<<<session->heads, decode_head_threads,
                                           reduce_shared_bytes,
                                           session->stream>>>(
            session->dtype, session->hidden, session->heads, session->kv_heads,
            session->head_dim, session->intermediate, session->device_step,
            max_steps, attention_chunks, session->device_scratch,
            session->device_decode_attention_values,
            session->device_decode_attention_m,
            session->device_decode_attention_l,
            session->device_projection_input);
        err = cudaGetLastError();
      }
    }
    if (err == cudaSuccess && layout.w_q_gate != kMissingOffset) {
      const uint32_t blocks =
          (attention_hidden + kDecodeThreads - 1) / kDecodeThreads;
      hf_layer_query_gate_attention_encode_kernel<<<blocks, kDecodeThreads, 0,
                                                    session->stream>>>(
          session->dtype, session->hidden, attention_hidden, kv_hidden,
          session->intermediate, session->device_step, max_steps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->attention_output_plan,
          session->device_arena + layout.w_o, session->device_projection_input,
          session->hidden, attention_hidden, 1, session->dtype, 0.0f,
          scratch.residual);
    if (err == cudaSuccess) {
      hf_layer_mlp_norm_encode_kernel<<<1, kDecodeNormThreads, 0,
                                        session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess && layout.mlp_kind == kMlpKindSparseMoe) {
      hf_layer_sparse_moe_encode_kernel<<<1, kDecodeThreads, 0,
                                          session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->device_scratch,
          session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
      err = project_encoded_rows(
          session, &session->gate_up_plan,
          session->device_gate_up_packed +
              packed_shape.gate_up_elements_per_layer * layer_index,
          session->device_projection_input,
          static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden, 1,
          session->dtype, 0.0f, scratch.gate);
      if (err == cudaSuccess) {
        const uint32_t ff_blocks =
            (session->intermediate + kDecodeThreads - 1) / kDecodeThreads;
        hf_layer_ff_encode_kernel<<<ff_blocks, kDecodeThreads, 0,
                                    session->stream>>>(
            session->dtype, session->hidden, attention_hidden, kv_hidden,
            session->intermediate, session->device_step, max_steps,
            session->device_scratch, session->device_projection_input);
        err = cudaGetLastError();
      }
      if (err == cudaSuccess)
        err = project_encoded_rows(
            session, &session->down_plan,
            session->device_arena + layout.w_down, session->device_projection_input,
            session->hidden, session->intermediate, 1, session->dtype, 0.0f,
            scratch.down);
    }
    }
    if (err == cudaSuccess && layer_index + 1 < session->layer_count) {
      const SequenceLayerLayout next_layout =
          session->host_layouts[layer_index + 1];
      hf_layer_finish_next_attn_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, output_offset, next_layout, session->dtype,
          layer_norm_weight_dtype(next_layout, session->dtype), session->hidden,
          layer_attention_hidden, layer_kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
      hf_layer_finish_final_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, session->arena_layout, session->dtype,
          session->dtype, session->hidden, layer_attention_hidden,
          layer_kv_hidden, session->intermediate, session->device_step,
          max_steps, session->rms_eps, session->device_scratch,
          session->device_projection_input);
      err = cudaGetLastError();
    }
    const uint64_t next_input = output_offset;
    output_offset = input_offset;
    input_offset = next_input;
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = project_encoded_rows(
        session, &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        1, session->dtype, 0.0f, device_logits);
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        session->stream, session->device_step, max_steps, has_eos_token,
        eos_token, device_logits, session->vocab_size, session->device_slots,
        session->active_sampler);
  }
  return err;
}

cudaError_t profile_cublas_layer_session_step(
    NervaCudaHfDecodeSequenceSession *session, uint32_t max_steps,
    uint32_t prompt_token_count, uint32_t has_eos_token, uint32_t eos_token,
    uint32_t attention_chunks, uint32_t cursor) {
  uint64_t projection_ns = 0;
  uint64_t qkv_projection_ns = 0;
  uint64_t attention_output_projection_ns = 0;
  uint64_t gate_up_projection_ns = 0;
  uint64_t down_projection_ns = 0;
  uint64_t lm_head_projection_ns = 0;
  uint64_t attention_ns = 0;
  uint64_t mlp_ns = 0;
  uint64_t norm_ns = 0;
  uint64_t sampling_ns = 0;
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const uint32_t decode_head_threads = decode_head_threads_for_session(session);
  const PackedProjectionShape packed_shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  LayerScratch scratch = layer_scratch_ptrs(
      session->device_scratch, session->hidden, attention_hidden, kv_hidden,
      session->intermediate);

  hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(session->device_step,
                                                          cursor);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) err = profile_begin(session);
  if (err == cudaSuccess && session->layer_count > 0) {
    const SequenceLayerLayout first_layout = session->host_layouts[0];
    hf_decode_prepare_first_attn_norm_encode_kernel<<<
        1, kDecodeNormThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, first_layout,
        session->dtype, layer_norm_weight_dtype(first_layout, session->dtype),
        session->hidden, attention_hidden, kv_hidden, session->intermediate,
        session->device_step, max_steps, session->device_prompt_tokens,
        prompt_token_count, session->device_slots, session->rms_eps,
        session->device_scratch,
        session->device_projection_input);
    err = cudaGetLastError();
  } else if (err == cudaSuccess) {
    hf_decode_prepare_input_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, session->hidden,
        session->device_step, max_steps, session->device_prompt_tokens,
        prompt_token_count, session->device_slots);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = profile_end(session, &norm_ns);

  uint64_t input_offset = session->arena_layout.input;
  uint64_t output_offset = session->arena_layout.scratch;
  for (uint32_t layer_index = 0;
       err == cudaSuccess && layer_index < session->layer_count;
       ++layer_index) {
    const SequenceLayerLayout layout = session->host_layouts[layer_index];
    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->qkv_plan,
          session->device_qkv_packed +
              packed_shape.qkv_elements_per_layer * layer_index,
          session->device_projection_input,
          static_cast<uint32_t>(packed_shape.qkv_rows), session->hidden, 1,
          session->dtype, 0.0f, scratch.q);
    if (err == cudaSuccess && layout.w_q_gate != kMissingOffset) {
      err = project_encoded_rows(
          session, nullptr, session->device_arena + layout.w_q_gate,
          session->device_projection_input, attention_hidden, session->hidden,
          1, session->dtype, 0.0f, scratch.q_gate);
    }
    if (err == cudaSuccess) err = profile_end(session, &qkv_projection_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess && attention_chunks == 0) {
      hf_layer_qkv_attention_encode_kernel<<<session->heads, decode_head_threads, 0,
                                             session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
#if NERVA_HAVE_CUDNN_FRONTEND
      const bool use_cudnn_decode_sdpa =
          can_use_cudnn_decode_sdpa(session, attention_chunks);
#else
      const bool use_cudnn_decode_sdpa = false;
#endif
      hf_layer_qkv_prepare_kernel<<<session->heads, decode_head_threads, 0,
                                    session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          use_cudnn_decode_sdpa ? session->device_decode_q : nullptr,
          use_cudnn_decode_sdpa ? session->device_decode_seq_len_q : nullptr,
          use_cudnn_decode_sdpa ? session->device_decode_seq_len_kv : nullptr);
      err = cudaGetLastError();
      const uint32_t query_group = session->heads / session->kv_heads;
      const bool use_shared_warp_gqa =
          query_group == kGroupedGqaHeads &&
          session->heads % session->kv_heads == 0 &&
          session->head_dim <= kSharedWarpGqaHeadDimMax;
      const bool use_grouped_gqa =
          query_group == kGroupedGqaHeads &&
          session->heads % session->kv_heads == 0 &&
          session->head_dim <= kGroupedGqaHeadDimMax;
      const bool use_fused_qk_selector =
          use_shared_warp_gqa &&
          experimental_rt_qk_fused_selector_active(session, attention_chunks);
      if (err == cudaSuccess && !use_fused_qk_selector) {
        err = launch_experimental_rt_qk_page_selector(
            session, layer_index, attention_chunks, max_steps, session->stream);
      }
      bool ran_cudnn_decode_sdpa = false;
#if NERVA_HAVE_CUDNN_FRONTEND
      if (err == cudaSuccess && use_cudnn_decode_sdpa) {
        err = execute_cudnn_decode_sdpa(session, layer_index);
        if (err == cudaSuccess) {
          ran_cudnn_decode_sdpa = true;
        }
      }
#endif
      if (err == cudaSuccess && !ran_cudnn_decode_sdpa) {
        const dim3 grid((use_shared_warp_gqa || use_grouped_gqa)
                            ? session->kv_heads
                            : session->heads,
                        attention_chunks);
        launch_hf_layer_attention_chunk_kernel(
            session->stream, grid, session->dtype, use_shared_warp_gqa,
            use_grouped_gqa, decode_head_threads, layer_index, session->hidden,
            session->heads, session->kv_heads, session->head_dim,
            session->intermediate, session->device_step, max_steps,
            attention_chunks, session->device_scratch,
            session->device_kv_keys, session->device_kv_values,
            session->device_decode_attention_values,
            session->device_decode_attention_m,
            session->device_decode_attention_l, session->kv_block_count,
            session->device_kv_block_table,
            session->experimental_rt_sparse_attention_active == 0
                ? nullptr
                : session->device_experimental_rt_candidate_pages,
            use_fused_qk_selector ? 1u : 0u,
            session->experimental_rt_local_window_tokens,
            session->experimental_rt_sink_tokens);

        err = cudaGetLastError();
      }
      if (err == cudaSuccess && !ran_cudnn_decode_sdpa) {
        const size_t reduce_shared_bytes =
            static_cast<size_t>(attention_chunks) * sizeof(float);
        hf_layer_attention_reduce_kernel<<<session->heads, decode_head_threads,
                                           reduce_shared_bytes,
                                           session->stream>>>(
            session->dtype, session->hidden, session->heads, session->kv_heads,
            session->head_dim, session->intermediate, session->device_step,
            max_steps, attention_chunks, session->device_scratch,
            session->device_decode_attention_values,
            session->device_decode_attention_m,
            session->device_decode_attention_l,
            session->device_projection_input);
        err = cudaGetLastError();
      }
    }
    if (err == cudaSuccess && layout.w_q_gate != kMissingOffset) {
      const uint32_t blocks =
          (attention_hidden + kDecodeThreads - 1) / kDecodeThreads;
      hf_layer_query_gate_attention_encode_kernel<<<blocks, kDecodeThreads, 0,
                                                    session->stream>>>(
          session->dtype, session->hidden, attention_hidden, kv_hidden,
          session->intermediate, session->device_step, max_steps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = profile_end(session, &attention_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->attention_output_plan,
          session->device_arena + layout.w_o, session->device_projection_input,
          session->hidden, attention_hidden, 1, session->dtype, 0.0f,
          scratch.residual);
    if (err == cudaSuccess) {
      err = profile_end(session, &attention_output_projection_ns);
    }

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess) {
      hf_layer_mlp_norm_encode_kernel<<<1, kDecodeNormThreads, 0,
                                        session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = profile_end(session, &norm_ns);

    if (layout.mlp_kind == kMlpKindSparseMoe) {
      if (err == cudaSuccess) err = profile_begin(session);
      if (err == cudaSuccess) {
        hf_layer_sparse_moe_encode_kernel<<<1, kDecodeThreads, 0,
                                            session->stream>>>(
            session->device_arena, layout, session->dtype, session->hidden,
            attention_hidden, kv_hidden, session->intermediate,
            session->device_step, max_steps, session->device_scratch,
            session->device_projection_input);
        err = cudaGetLastError();
      }
      if (err == cudaSuccess) err = profile_end(session, &mlp_ns);
    } else {
      if (err == cudaSuccess) err = profile_begin(session);
      if (err == cudaSuccess)
        err = project_encoded_rows(
            session, &session->gate_up_plan,
            session->device_gate_up_packed +
                packed_shape.gate_up_elements_per_layer * layer_index,
            session->device_projection_input,
            static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden, 1,
            session->dtype, 0.0f, scratch.gate);
      if (err == cudaSuccess) err = profile_end(session, &gate_up_projection_ns);

      if (err == cudaSuccess) err = profile_begin(session);
      if (err == cudaSuccess) {
        const uint32_t ff_blocks =
            (session->intermediate + kDecodeThreads - 1) / kDecodeThreads;
        hf_layer_ff_encode_kernel<<<ff_blocks, kDecodeThreads, 0,
                                    session->stream>>>(
            session->dtype, session->hidden, attention_hidden, kv_hidden,
            session->intermediate, session->device_step, max_steps,
            session->device_scratch, session->device_projection_input);
        err = cudaGetLastError();
      }
      if (err == cudaSuccess) err = profile_end(session, &mlp_ns);

      if (err == cudaSuccess) err = profile_begin(session);
      if (err == cudaSuccess)
        err = project_encoded_rows(
            session, &session->down_plan,
            session->device_arena + layout.w_down, session->device_projection_input,
            session->hidden, session->intermediate, 1, session->dtype, 0.0f,
            scratch.down);
      if (err == cudaSuccess) err = profile_end(session, &down_projection_ns);
    }

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess && layer_index + 1 < session->layer_count) {
      const SequenceLayerLayout next_layout =
          session->host_layouts[layer_index + 1];
      hf_layer_finish_next_attn_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, output_offset, next_layout, session->dtype,
          layer_norm_weight_dtype(next_layout, session->dtype), session->hidden,
          attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
      hf_layer_finish_final_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, session->arena_layout, session->dtype,
          session->dtype, session->hidden, attention_hidden, kv_hidden,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->device_scratch,
          session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = profile_end(session, &norm_ns);
    const uint64_t next_input = output_offset;
    output_offset = input_offset;
    input_offset = next_input;
  }

  if (err == cudaSuccess) err = profile_begin(session);
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = project_encoded_rows(
        session, &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        1, session->dtype, 0.0f, device_logits);
  }
  if (err == cudaSuccess) err = profile_end(session, &lm_head_projection_ns);

  if (err == cudaSuccess) err = profile_begin(session);
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        session->stream, session->device_step, max_steps, has_eos_token,
        eos_token, device_logits, session->vocab_size, session->device_slots,
        session->active_sampler);
  }
  if (err == cudaSuccess) err = profile_end(session, &sampling_ns);

  if (err == cudaSuccess) {
    hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(
        session->device_step, cursor);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(session->stream);
  if (err == cudaSuccess) {
    projection_ns = qkv_projection_ns + attention_output_projection_ns +
                    gate_up_projection_ns + down_projection_ns +
                    lm_head_projection_ns;
    session->cached_projection_ns = projection_ns;
    session->cached_qkv_projection_ns = qkv_projection_ns;
    session->cached_attention_output_projection_ns =
        attention_output_projection_ns;
    session->cached_gate_up_projection_ns = gate_up_projection_ns;
    session->cached_down_projection_ns = down_projection_ns;
    session->cached_lm_head_projection_ns = lm_head_projection_ns;
    session->cached_attention_ns = attention_ns;
    session->cached_mlp_ns = mlp_ns;
    session->cached_norm_ns = norm_ns;
    session->cached_sampling_ns = sampling_ns;
  }
  return err;
}

cudaError_t ensure_session_graph(NervaCudaHfDecodeSequenceSession *session,
                                 uint32_t max_steps,
                                 uint32_t prompt_token_count,
                                 uint32_t has_eos_token,
                                 uint32_t eos_token,
                                 uint32_t attention_chunks,
                                 uint32_t profile_cursor,
                                 NervaCudaHfDecodeSequenceResult *out);

cudaError_t encoded_row_major_gemm_tokens_cached(
    NervaCudaHfDecodeSequenceSession *session, const uint16_t *matrix,
    const uint16_t *input, uint32_t rows, uint32_t cols, uint32_t tokens,
    uint32_t dtype, float beta, float *output) {
  if (session == nullptr) {
    return cudaErrorInvalidValue;
  }
  return encoded_row_major_gemm_tokens(session->cublas, matrix, input, rows,
                                       cols, tokens, dtype, beta, output);
}

cudaError_t launch_cublas_session_prefill(
    NervaCudaHfDecodeSequenceSession *session, uint32_t prompt_token_count,
    uint32_t has_eos_token, uint32_t eos_token,
    NervaCudaHfDecodeSequenceResult *out) {
  if (prompt_token_count == 0 || prompt_token_count > session->max_context_tokens ||
      !use_cublas_prefill_path(session)) {
    return cudaErrorInvalidValue;
  }
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const PackedProjectionShape packed_shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  const bool collect_profile = session->detailed_profile != 0;
  uint64_t qkv_projection_ns = 0;
  uint64_t attention_output_projection_ns = 0;
  uint64_t gate_up_projection_ns = 0;
  uint64_t down_projection_ns = 0;
  uint64_t lm_head_projection_ns = 0;
  uint64_t attention_ns = 0;
  uint64_t mlp_ns = 0;
  uint64_t norm_ns = 0;
  uint64_t sampling_ns = 0;
  auto profile_stage_begin = [&]() -> cudaError_t {
    return collect_profile ? profile_begin(session) : cudaSuccess;
  };
  auto profile_stage_end = [&](uint64_t *bucket) -> cudaError_t {
    return collect_profile ? profile_end(session, bucket) : cudaSuccess;
  };
  cudaError_t err = cudaEventRecord(session->device_start, session->stream);
  if (err == cudaSuccess) {
    hf_prefill_embed_kernel<<<prompt_token_count, kDecodeThreads, 0,
                              session->stream>>>(
        session->device_arena, session->arena_layout, session->hidden,
        session->device_prompt_tokens, prompt_token_count,
        session->device_prefill_hidden_a);
    err = cudaGetLastError();
    out->kernel_launches += 1;
  }
  uint16_t *hidden_in = session->device_prefill_hidden_a;
  uint16_t *hidden_out = session->device_prefill_hidden_b;
  for (uint32_t layer_index = 0;
       err == cudaSuccess && layer_index < session->layer_count;
       ++layer_index) {
    const SequenceLayerLayout layout = session->host_layouts[layer_index];
    for (uint32_t chunk_start = 0;
         err == cudaSuccess && chunk_start < prompt_token_count;
         chunk_start += session->prefill_chunk_tokens) {
      const uint32_t chunk_tokens =
          std::min(session->prefill_chunk_tokens, prompt_token_count - chunk_start);
      if (err == cudaSuccess) err = profile_stage_begin();
      if (err == cudaSuccess) {
        hf_prefill_attn_norm_kernel<<<chunk_tokens, kDecodeThreads, 0,
                                      session->stream>>>(
            session->device_arena, layout, session->dtype, session->hidden,
            chunk_start, chunk_tokens, session->rms_eps, hidden_in,
            session->device_prefill_norm);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&norm_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        err = project_encoded_rows(
            session, &session->qkv_plan,
            session->device_qkv_packed +
                packed_shape.qkv_elements_per_layer * layer_index,
            session->device_prefill_norm,
            static_cast<uint32_t>(packed_shape.qkv_rows), session->hidden,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_qkv);
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess && layout.w_q_gate != kMissingOffset) {
        err = project_encoded_rows(
            session, nullptr, session->device_arena + layout.w_q_gate,
            session->device_prefill_norm, attention_hidden, session->hidden,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_q_gate);
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&qkv_projection_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      const uint32_t query_group =
          session->kv_heads == 0 ? 0 : session->heads / session->kv_heads;
      const bool use_grouped_gqa =
          query_group == kGroupedGqaHeads &&
          session->heads % session->kv_heads == 0 &&
          session->head_dim <= kSharedWarpGqaHeadDimMax;
      const uint32_t prefill_local_window_tokens =
          session->experimental_prefill_local_window_tokens;
#if NERVA_HAVE_CUDNN_FRONTEND
      const bool use_cudnn_sdpa =
          prefill_local_window_tokens == 0 &&
          session->cudnn_prefill_sdpa_disabled == 0 &&
          session->cudnn != nullptr &&
          session->device_prefill_qkv_encoded != nullptr &&
          session->dtype == kDTypeBF16 && use_grouped_gqa &&
          chunk_start == 0 && chunk_tokens == prompt_token_count &&
          session->head_dim <= 128;
#endif
      if (err == cudaSuccess) {
        const dim3 grid(chunk_tokens, std::max(session->heads, session->kv_heads));
        hf_prefill_qkv_publish_kernel<<<grid, session->head_threads, 0,
                                      session->stream>>>(
            session->device_arena, layout, layer_index, session->dtype,
            session->heads, session->kv_heads, session->head_dim,
            session->max_context_tokens, chunk_start, chunk_tokens,
            session->rms_eps, session->rope_theta, session->device_prefill_qkv,
            session->device_kv_keys, session->device_kv_values,
#if NERVA_HAVE_CUDNN_FRONTEND
            use_cudnn_sdpa ? session->device_prefill_qkv_encoded : nullptr,
#else
            nullptr,
#endif
            session->kv_block_count, session->device_kv_block_table);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&norm_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
#if NERVA_HAVE_CUDNN_FRONTEND
        bool ran_cudnn_sdpa = false;
        if (use_cudnn_sdpa) {
          err = execute_cudnn_prefill_sdpa(session, chunk_tokens);
          if (err == cudaSuccess) {
            out->kernel_launches += 1;
            ran_cudnn_sdpa = true;
          } else if (err == cudaErrorNotSupported ||
                     err == cudaErrorMemoryAllocation) {
            session->cudnn_prefill_sdpa_disabled = 1;
            err = cudaSuccess;
          }
        }
        if (!ran_cudnn_sdpa) {
#endif
        if (use_grouped_gqa) {
          const dim3 grid(chunk_tokens, session->kv_heads);
          launch_hf_prefill_grouped_gqa_attention_direct_kernel(
              session->stream, grid, session->dtype, layer_index,
              session->heads, session->kv_heads, session->head_dim,
              session->max_context_tokens, chunk_start, chunk_tokens,
              session->device_prefill_qkv, session->device_kv_keys,
              session->device_kv_values, session->kv_block_count,
              session->device_kv_block_table, session->device_prefill_attn,
              prefill_local_window_tokens);
        } else {
          const dim3 grid(chunk_tokens, session->heads);
          hf_prefill_attention_kernel<<<grid, session->head_threads,
                                        session->head_dim * sizeof(float),
                                        session->stream>>>(
              layer_index, session->dtype, session->heads, session->kv_heads,
              session->head_dim, session->max_context_tokens, chunk_start,
              chunk_tokens, session->device_prefill_qkv, session->device_kv_keys,
              session->device_kv_values, session->kv_block_count,
              session->device_kv_block_table, session->device_prefill_attn,
              prefill_local_window_tokens);
        }
          err = cudaGetLastError();
          out->kernel_launches += 1;
#if NERVA_HAVE_CUDNN_FRONTEND
        }
#endif
      }
      if (err == cudaSuccess && layout.w_q_gate != kMissingOffset) {
        const uint32_t blocks =
            static_cast<uint32_t>(
                (static_cast<uint64_t>(chunk_tokens) * attention_hidden +
                 kDecodeThreads - 1) /
                kDecodeThreads);
        hf_prefill_query_gate_attention_kernel<<<blocks, kDecodeThreads, 0,
                                                 session->stream>>>(
            session->dtype, attention_hidden, chunk_tokens,
            session->device_prefill_q_gate, session->device_prefill_attn);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&attention_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        err = project_encoded_rows(
            session, &session->attention_output_plan,
            session->device_arena + layout.w_o,
            session->device_prefill_attn, session->hidden, attention_hidden,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_o);
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) {
        err = profile_stage_end(&attention_output_projection_ns);
      }
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        hf_prefill_mlp_norm_kernel<<<chunk_tokens, kDecodeThreads, 0,
                                     session->stream>>>(
            session->device_arena, layout, session->dtype, session->hidden,
            chunk_start, chunk_tokens, session->rms_eps, hidden_in,
            session->device_prefill_o, session->device_prefill_norm);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&norm_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (layout.mlp_kind == kMlpKindSparseMoe) {
        if (err == cudaSuccess) {
          hf_prefill_sparse_moe_kernel<<<chunk_tokens, kDecodeThreads, 0,
                                         session->stream>>>(
              session->device_arena, layout, session->dtype, session->hidden,
              session->intermediate, chunk_tokens, session->device_prefill_norm,
              session->device_prefill_gate_up, session->device_prefill_down);
          err = cudaGetLastError();
          out->kernel_launches += 1;
        }
        if (err == cudaSuccess) err = profile_stage_end(&mlp_ns);
      } else {
        if (err == cudaSuccess) {
          err = project_encoded_rows(
              session, &session->gate_up_plan,
              session->device_gate_up_packed +
                  packed_shape.gate_up_elements_per_layer * layer_index,
              session->device_prefill_norm,
              static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden,
              chunk_tokens, session->dtype, 0.0f,
              session->device_prefill_gate_up);
          out->kernel_launches += 1;
        }
        if (err == cudaSuccess) err = profile_stage_end(&gate_up_projection_ns);
        if (err == cudaSuccess) {
          err = profile_stage_begin();
        }
        if (err == cudaSuccess) {
          const uint32_t blocks =
              static_cast<uint32_t>(
                  (static_cast<uint64_t>(chunk_tokens) * session->intermediate +
                   kDecodeThreads - 1) /
                  kDecodeThreads);
          hf_prefill_ff_kernel<<<blocks, kDecodeThreads, 0, session->stream>>>(
              session->dtype, session->intermediate, chunk_tokens,
              session->device_prefill_gate_up, session->device_prefill_ff);
          err = cudaGetLastError();
          out->kernel_launches += 1;
        }
        if (err == cudaSuccess) err = profile_stage_end(&mlp_ns);
        if (err == cudaSuccess) {
          err = profile_stage_begin();
        }
        if (err == cudaSuccess) {
          err = project_encoded_rows(
              session, &session->down_plan,
              session->device_arena + layout.w_down,
              session->device_prefill_ff, session->hidden, session->intermediate,
              chunk_tokens, session->dtype, 0.0f, session->device_prefill_down);
          out->kernel_launches += 1;
        }
        if (err == cudaSuccess) err = profile_stage_end(&down_projection_ns);
      }
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        const uint32_t blocks =
            static_cast<uint32_t>(
                (static_cast<uint64_t>(chunk_tokens) * session->hidden +
                 kDecodeThreads - 1) /
                kDecodeThreads);
        hf_prefill_finish_kernel<<<blocks, kDecodeThreads, 0, session->stream>>>(
            session->dtype, session->hidden, chunk_start, chunk_tokens,
            session->device_prefill_o, session->device_prefill_down, hidden_out);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&mlp_ns);
    }
    std::swap(hidden_in, hidden_out);
  }
  if (err == cudaSuccess) {
    err = profile_stage_begin();
  }
  if (err == cudaSuccess) {
    hf_prefill_final_norm_last_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, session->dtype,
        session->hidden, prompt_token_count, session->rms_eps, hidden_in,
        session->device_projection_input);
    err = cudaGetLastError();
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) err = profile_stage_end(&norm_ns);
  if (err == cudaSuccess) {
    hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(
        session->device_step, prompt_token_count - 1u);
    err = cudaGetLastError();
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) {
    err = profile_stage_begin();
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = project_encoded_rows(
        session, &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        1, session->dtype, 0.0f, device_logits);
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) err = profile_stage_end(&lm_head_projection_ns);
  if (err == cudaSuccess) {
    err = profile_stage_begin();
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        session->stream, session->device_step, session->max_context_tokens,
        has_eos_token, eos_token, device_logits, session->vocab_size,
        session->device_slots, session->active_sampler);
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) err = profile_stage_end(&sampling_ns);
  if (err == cudaSuccess) {
    err = cudaEventRecord(session->device_stop, session->stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(session->stream);
    out->sync_calls += 1;
  }
  if (err == cudaSuccess) {
    float device_ms = 0.0f;
    err = cudaEventElapsedTime(&device_ms, session->device_start,
                               session->device_stop);
    if (err == cudaSuccess && device_ms > 0.0f) {
      out->device_elapsed_ns = static_cast<uint64_t>(device_ms * 1000000.0f);
      if (out->device_elapsed_ns == 0) out->device_elapsed_ns = 1;
    }
  }
  if (err == cudaSuccess && collect_profile) {
    out->qkv_projection_ns = qkv_projection_ns;
    out->attention_output_projection_ns = attention_output_projection_ns;
    out->gate_up_projection_ns = gate_up_projection_ns;
    out->down_projection_ns = down_projection_ns;
    out->lm_head_projection_ns = lm_head_projection_ns;
    out->projection_ns = qkv_projection_ns + attention_output_projection_ns +
                         gate_up_projection_ns + down_projection_ns +
                         lm_head_projection_ns;
    out->attention_ns = attention_ns;
    out->mlp_ns = mlp_ns;
    out->norm_ns = norm_ns;
    out->sampling_ns = sampling_ns;
  }
  return err;
}

cudaError_t launch_serial_session_prefill(
    NervaCudaHfDecodeSequenceSession *session, uint32_t prompt_token_count,
    uint32_t has_eos_token, uint32_t eos_token,
    NervaCudaHfDecodeSequenceResult *out) {
  cudaError_t err =
      ensure_session_graph(session, session->max_context_tokens, prompt_token_count,
                           has_eos_token, eos_token, 0, 0, out);
  if (err == cudaSuccess) {
    err = cudaEventRecord(session->device_start, session->stream);
  }
  for (uint32_t step = 0; err == cudaSuccess && step < prompt_token_count; ++step) {
    err = cudaGraphLaunch(session->cached_graph_exec, session->stream);
    if (err == cudaSuccess) {
      out->graph_replays += 1;
      out->graph_launches += 1;
      out->kernel_launches += out->graph_nodes == 0 ? 1 : out->graph_nodes;
    }
  }
  if (err == cudaSuccess) {
    err = cudaEventRecord(session->device_stop, session->stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(session->stream);
    out->sync_calls += 1;
  }
  if (err == cudaSuccess) {
    float device_ms = 0.0f;
    err = cudaEventElapsedTime(&device_ms, session->device_start,
                               session->device_stop);
    if (err == cudaSuccess && device_ms > 0.0f) {
      out->device_elapsed_ns = static_cast<uint64_t>(device_ms * 1000000.0f);
      if (out->device_elapsed_ns == 0) out->device_elapsed_ns = 1;
    }
  }
  return err;
}
