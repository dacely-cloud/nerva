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
    uint32_t prompt_token_count, uint32_t has_eos_token, uint32_t eos_token,
    uint32_t sample_final_head) {
  hf_decode_sequence_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
      session->device_arena, session->arena_layout, session->device_layouts,
      session->layer_count, session->dtype,
      final_norm_weight_dtype_for_layouts(session->host_layouts.data(),
                                          session->layer_count, session->dtype),
      session->hidden, session->heads, session->kv_heads, session->head_dim,
      session->intermediate, 0, session->device_step, max_steps,
      session->device_prompt_tokens, prompt_token_count, session->rms_eps,
      session->rope_theta, session->device_scratch, session->device_kv_keys,
      session->device_kv_values, session->kv_block_count,
      session->device_kv_block_table,
      session->device_slots, session->device_linear_gdn_conv_state,
      session->device_linear_gdn_recurrent_state);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess && sample_final_head != 0) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = final_head_gemv(session->cublas, session->device_arena,
                          session->arena_layout, session->dtype,
                          session->hidden, session->vocab_size, device_logits);
  }
  if (err == cudaSuccess && sample_final_head != 0) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        session->stream, session->device_step, max_steps, has_eos_token,
        eos_token, device_logits, session->vocab_size, session->device_slots,
        session->active_sampler);
  }
  if (err == cudaSuccess && sample_final_head == 0) {
    hf_decode_advance_step_kernel<<<1, 1, 0, session->stream>>>(
        session->device_step, max_steps);
    err = cudaGetLastError();
  }
  return err;
}

#include "hf_decode_sequence/deepseek/session_decode.inc.cu"

cudaError_t launch_cublas_layer_session_step(
    NervaCudaHfDecodeSequenceSession *session, uint32_t max_steps,
    uint32_t prompt_token_count, uint32_t has_eos_token, uint32_t eos_token,
    uint32_t attention_chunks, uint32_t sample_final_head) {
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
        session->dtype, session->hidden,
        layer_norm_weight_dtype(first_layout, session->dtype),
        first_attention_hidden, first_kv_hidden,
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
          session, layout, layer_index, max_steps, attention_chunks);
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
      if (err == cudaSuccess) {
        err = launch_experimental_rt_query_descriptor_selector(
            session, layer_index, attention_chunks, session->stream);
      }
      if (err == cudaSuccess && !use_fused_qk_selector &&
          !experimental_rt_query_descriptor_selector_active(session,
                                                           attention_chunks)) {
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
    } else if (err == cudaSuccess && is_deepseek_v4_native) {
      const uint32_t final_norm_weight_dtype =
          final_norm_weight_dtype_for_layouts(session->host_layouts.data(),
                                              session->layer_count,
                                              session->dtype);
      hf_deepseek_v4_finish_final_norm_encode_kernel<<<
          1, 1, 0, session->stream>>>(
          session->device_arena, session->arena_layout, layout, session->dtype,
          final_norm_weight_dtype, session->hidden, layer_attention_hidden,
          layer_kv_hidden, session->intermediate, session->device_step,
          max_steps, session->rms_eps, session->device_scratch,
          session->device_projection_input,
          session->device_deepseek_mhc_residual,
          session->device_deepseek_mhc_post_mix,
          session->device_deepseek_mhc_comb_mix);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
      const uint32_t final_norm_weight_dtype =
          final_norm_weight_dtype_for_layouts(session->host_layouts.data(),
                                              session->layer_count,
                                              session->dtype);
      hf_layer_finish_final_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, session->arena_layout, session->dtype,
          final_norm_weight_dtype, session->hidden, layer_attention_hidden,
          layer_kv_hidden, session->intermediate, session->device_step,
          max_steps, session->rms_eps, session->device_scratch,
          session->device_projection_input);
      err = cudaGetLastError();
    }
    const uint64_t next_input = output_offset;
    output_offset = input_offset;
    input_offset = next_input;
  }
  if (err == cudaSuccess && sample_final_head != 0) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = project_encoded_rows(
        session, &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        1, session->dtype, 0.0f, device_logits);
  }
  if (err == cudaSuccess && sample_final_head != 0) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        session->stream, session->device_step, max_steps, has_eos_token,
        eos_token, device_logits, session->vocab_size, session->device_slots,
        session->active_sampler);
  }
  if (err == cudaSuccess && sample_final_head == 0) {
    hf_decode_advance_step_kernel<<<1, 1, 0, session->stream>>>(
        session->device_step, max_steps);
    err = cudaGetLastError();
  }
  return err;
}

cudaError_t profile_cublas_layer_session_step(
    NervaCudaHfDecodeSequenceSession *session, uint32_t max_steps,
    uint32_t prompt_token_count, uint32_t has_eos_token, uint32_t eos_token,
    uint32_t attention_chunks, uint32_t cursor, uint32_t sample_final_head) {
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
    const uint32_t first_attention_hidden = static_cast<uint32_t>(
        layer_attention_workspace_rows(first_layout, attention_hidden));
    const uint32_t first_kv_hidden = static_cast<uint32_t>(
        layout_deepseek_kv_cache_width(first_layout, kv_hidden));
    hf_decode_prepare_first_attn_norm_encode_kernel<<<
        1, kDecodeNormThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, first_layout,
        session->dtype, session->hidden,
        layer_norm_weight_dtype(first_layout, session->dtype),
        first_attention_hidden, first_kv_hidden, session->intermediate,
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
    const bool is_deepseek_v3 = layout_is_deepseek_v3_mla(layout);
    const bool is_deepseek_v4_native = layout_is_deepseek_v4_native(layout);
    const uint32_t layer_attention_hidden = static_cast<uint32_t>(
        layer_attention_workspace_rows(layout, attention_hidden));
    const uint32_t layer_kv_hidden = static_cast<uint32_t>(
        layout_deepseek_kv_cache_width(layout, kv_hidden));
    if (is_deepseek_v3 || is_deepseek_v4_native) {
      DeepseekDecodeProfileBuckets deepseek_profile{
          &qkv_projection_ns,
          &attention_output_projection_ns,
          &gate_up_projection_ns,
          &down_projection_ns,
          &attention_ns,
          &mlp_ns,
          &norm_ns};
      if (err == cudaSuccess && is_deepseek_v3) {
        err = launch_deepseek_v3_mla_projection_step(
            session, layout, layer_index, max_steps, attention_chunks,
            &deepseek_profile);
      } else if (err == cudaSuccess) {
        err = launch_deepseek_v4_swa_dense_projection_step(
            session, layout, layer_index, max_steps, prompt_token_count,
            &deepseek_profile);
      }

      if (err == cudaSuccess) err = profile_begin(session);
      if (err == cudaSuccess && layer_index + 1 < session->layer_count) {
        const SequenceLayerLayout next_layout =
            session->host_layouts[layer_index + 1];
        hf_layer_finish_next_attn_norm_encode_kernel<<<
            1, kDecodeNormThreads, 0, session->stream>>>(
            session->device_arena, output_offset, next_layout, session->dtype,
            layer_norm_weight_dtype(next_layout, session->dtype),
            session->hidden, layer_attention_hidden, layer_kv_hidden,
            session->intermediate, session->device_step, max_steps,
            session->rms_eps, session->device_scratch,
            session->device_projection_input);
        err = cudaGetLastError();
      } else if (err == cudaSuccess && is_deepseek_v4_native) {
        const uint32_t final_norm_weight_dtype =
            final_norm_weight_dtype_for_layouts(session->host_layouts.data(),
                                                session->layer_count,
                                                session->dtype);
        hf_deepseek_v4_finish_final_norm_encode_kernel<<<
            1, 1, 0, session->stream>>>(
            session->device_arena, session->arena_layout, layout,
            session->dtype, final_norm_weight_dtype, session->hidden,
            layer_attention_hidden, layer_kv_hidden, session->intermediate,
            session->device_step, max_steps, session->rms_eps,
            session->device_scratch, session->device_projection_input,
            session->device_deepseek_mhc_residual,
            session->device_deepseek_mhc_post_mix,
            session->device_deepseek_mhc_comb_mix);
        err = cudaGetLastError();
      } else if (err == cudaSuccess) {
        const uint32_t final_norm_weight_dtype =
            final_norm_weight_dtype_for_layouts(session->host_layouts.data(),
                                                session->layer_count,
                                                session->dtype);
        hf_layer_finish_final_norm_encode_kernel<<<
            1, kDecodeNormThreads, 0, session->stream>>>(
            session->device_arena, session->arena_layout, session->dtype,
            final_norm_weight_dtype, session->hidden, layer_attention_hidden,
            layer_kv_hidden, session->intermediate, session->device_step,
            max_steps, session->rms_eps, session->device_scratch,
            session->device_projection_input);
        err = cudaGetLastError();
      }
      if (err == cudaSuccess) err = profile_end(session, &norm_ns);
      const uint64_t next_input = output_offset;
      output_offset = input_offset;
      input_offset = next_input;
      continue;
    }
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
      if (err == cudaSuccess) {
        err = launch_experimental_rt_query_descriptor_selector(
            session, layer_index, attention_chunks, session->stream);
      }
      if (err == cudaSuccess && !use_fused_qk_selector &&
          !experimental_rt_query_descriptor_selector_active(session,
                                                           attention_chunks)) {
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
    } else if (err == cudaSuccess && layout_is_deepseek_v4_native(layout)) {
      const uint32_t final_norm_weight_dtype =
          final_norm_weight_dtype_for_layouts(session->host_layouts.data(),
                                              session->layer_count,
                                              session->dtype);
      hf_deepseek_v4_finish_final_norm_encode_kernel<<<
          1, 1, 0, session->stream>>>(
          session->device_arena, session->arena_layout, layout, session->dtype,
          final_norm_weight_dtype, session->hidden, attention_hidden, kv_hidden,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->device_scratch,
          session->device_projection_input,
          session->device_deepseek_mhc_residual,
          session->device_deepseek_mhc_post_mix,
          session->device_deepseek_mhc_comb_mix);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
      const uint32_t final_norm_weight_dtype =
          final_norm_weight_dtype_for_layouts(session->host_layouts.data(),
                                              session->layer_count,
                                              session->dtype);
      hf_layer_finish_final_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, session->arena_layout, session->dtype,
          final_norm_weight_dtype, session->hidden, attention_hidden, kv_hidden,
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

  if (err == cudaSuccess && sample_final_head != 0) err = profile_begin(session);
  if (err == cudaSuccess && sample_final_head != 0) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = project_encoded_rows(
        session, &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        1, session->dtype, 0.0f, device_logits);
  }
  if (err == cudaSuccess && sample_final_head != 0) {
    err = profile_end(session, &lm_head_projection_ns);
  }

  if (err == cudaSuccess && sample_final_head != 0) err = profile_begin(session);
  if (err == cudaSuccess && sample_final_head != 0) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        session->stream, session->device_step, max_steps, has_eos_token,
        eos_token, device_logits, session->vocab_size, session->device_slots,
        session->active_sampler);
  }
  if (err == cudaSuccess && sample_final_head != 0) {
    err = profile_end(session, &sampling_ns);
  }

  if (err == cudaSuccess && sample_final_head == 0) {
    hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(
        session->device_step, cursor + 1u);
    err = cudaGetLastError();
  } else if (err == cudaSuccess) {
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
                                 uint32_t sample_final_head,
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

bool use_deepseek_v3_single_layer_prefill_cache_path(
    const NervaCudaHfDecodeSequenceSession *session, uint32_t prompt_token_count) {
  if (session == nullptr || prompt_token_count == 0 ||
      prompt_token_count > session->max_context_tokens ||
      session->layer_count != 1 || session->host_layouts.empty() ||
      session->device_prefill_hidden_a == nullptr ||
      session->device_prefill_norm == nullptr ||
      session->device_prefill_qkv == nullptr ||
      session->device_kv_keys == nullptr) {
    return false;
  }
  const SequenceLayerLayout layout = session->host_layouts[0];
  if (!layout_is_deepseek_v3_mla(layout) ||
      layout.rms_attn == kMissingOffset || layout.w_k == kMissingOffset ||
      layout.deepseek_kv_a_scale == kMissingOffset ||
      layout.k_norm == kMissingOffset || layout.deepseek_kv_lora_rank == 0 ||
      layout.deepseek_qk_rope_head_dim == 0) {
    return false;
  }
  const bool active_sparse_indexer =
      layout_is_deepseek_v32_indexer_query_native(layout) &&
      layout.deepseek_index_topk != 0;
  if (active_sparse_indexer &&
      (layout.w_q == kMissingOffset ||
       layout.deepseek_q_a_scale == kMissingOffset ||
       layout.q_norm == kMissingOffset ||
       layout.deepseek_indexer_q == kMissingOffset ||
       layout.deepseek_indexer_q_scale == kMissingOffset ||
       session->device_prefill_o == nullptr ||
       session->device_prefill_attn == nullptr ||
       layout.deepseek_q_lora_rank > session->hidden ||
       layout.deepseek_q_lora_rank > session->heads * session->head_dim)) {
    return false;
  }
  const uint32_t kv_a_rows =
      layout.deepseek_kv_lora_rank + layout.deepseek_qk_rope_head_dim;
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const PackedProjectionShape packed_shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  return packed_shape.qkv_rows >= kv_a_rows;
}

cudaError_t launch_deepseek_v3_single_layer_prefill_cache_path(
    NervaCudaHfDecodeSequenceSession *session, uint32_t prompt_token_count,
    uint32_t has_eos_token, uint32_t eos_token,
    NervaCudaHfDecodeSequenceResult *out) {
  if (!use_deepseek_v3_single_layer_prefill_cache_path(session,
                                                       prompt_token_count)) {
    return cudaErrorInvalidValue;
  }
  const SequenceLayerLayout layout = session->host_layouts[0];
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t q_lora_rank = layout.deepseek_q_lora_rank;
  const uint32_t kv_a_rows =
      layout.deepseek_kv_lora_rank + layout.deepseek_qk_rope_head_dim;
  const bool active_sparse_indexer =
      layout_is_deepseek_v32_indexer_query_native(layout) &&
      layout.deepseek_index_topk != 0;
  const uint32_t prefix_tokens =
      prompt_token_count == 0 ? 0 : prompt_token_count - 1u;
  constexpr uint32_t block_rows = 128;
  constexpr uint32_t block_cols = 128;
  cudaError_t err = cudaEventRecord(session->device_start, session->stream);
  for (uint32_t chunk_start = 0;
       err == cudaSuccess && chunk_start < prompt_token_count;
       chunk_start += session->prefill_chunk_tokens) {
    const uint32_t chunk_tokens = std::min(session->prefill_chunk_tokens,
                                          prompt_token_count - chunk_start);
    const uint32_t indexer_tokens =
        active_sparse_indexer && chunk_start < prefix_tokens
            ? std::min(chunk_tokens, prefix_tokens - chunk_start)
            : 0u;
    hf_prefill_embed_range_kernel<<<chunk_tokens, kDecodeThreads, 0,
                                    session->stream>>>(
        session->device_arena, session->arena_layout, session->hidden,
        session->device_prompt_tokens + chunk_start, chunk_tokens, chunk_start,
        session->device_prefill_hidden_a);
    err = cudaGetLastError();
    if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;

    if (err == cudaSuccess) {
      hf_deepseek_rms_norm_encoded_tokens_kernel<<<chunk_tokens,
                                                   kDecodeNormThreads, 0,
                                                   session->stream>>>(
          session->device_arena, layout.rms_attn,
          session->device_prefill_hidden_a +
              static_cast<uint64_t>(chunk_start) * session->hidden,
          deepseek_norm_weight_dtype(layout), session->dtype, session->dtype,
          session->hidden, session->hidden, session->hidden, chunk_tokens,
          session->rms_eps, session->device_prefill_norm);
      err = cudaGetLastError();
      if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
    }

    if (err == cudaSuccess && indexer_tokens != 0) {
      hf_deepseek_v32_indexer_kv_encode_tokens_kernel<<<
          indexer_tokens, kDecodeThreads, 0, session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          chunk_start, indexer_tokens, session->max_context_tokens,
          session->rope_theta, session->device_prefill_norm, session->hidden,
          session->device_deepseek_indexer_kv,
          deepseek_v32_indexer_kv_layer_offset_bytes(session, 0),
          deepseek_v32_indexer_kv_block_count(session, layout),
          session->kv_block_count, session->device_kv_block_table,
          session->device_deepseek_runtime_counters);
      err = cudaGetLastError();
      if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
    }

    if (err == cudaSuccess && indexer_tokens != 0) {
      const dim3 weight_grid(layout.deepseek_index_n_heads, indexer_tokens);
      hf_deepseek_v32_indexer_weight_state_tokens_kernel<<<
          weight_grid, kDecodeThreads, 0, session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          chunk_start, indexer_tokens, session->max_context_tokens,
          session->device_prefill_norm, session->hidden,
          reinterpret_cast<uint8_t *>(session->device_deepseek_indexer_state),
          deepseek_v32_indexer_query_state_layer_offset_bytes(session, 0));
      err = cudaGetLastError();
      if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
    }

    if (err == cudaSuccess) {
      err = launch_deepseek_fp8_f32_scale_encoded_gemm_tokens(
          session->stream, deepseek_fp8_ptr(session->device_arena, layout.w_k),
          deepseek_scale_ptr(session->device_arena,
                             layout.deepseek_kv_a_scale),
          session->device_prefill_norm, session->dtype, kv_a_rows,
          session->hidden, chunk_tokens, block_rows, block_cols,
          session->device_prefill_qkv);
      if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
    }

    if (err == cudaSuccess && indexer_tokens != 0) {
      err = launch_deepseek_fp8_f32_scale_encoded_gemm_tokens(
          session->stream, deepseek_fp8_ptr(session->device_arena, layout.w_q),
          deepseek_scale_ptr(session->device_arena,
                             layout.deepseek_q_a_scale),
          session->device_prefill_norm, session->dtype, q_lora_rank,
          session->hidden, indexer_tokens, block_rows, block_cols,
          session->device_prefill_o);
      if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
    }

    if (err == cudaSuccess && indexer_tokens != 0) {
      hf_deepseek_rms_norm_f32_tokens_kernel<<<indexer_tokens,
                                               kDecodeNormThreads, 0,
                                               session->stream>>>(
          session->device_arena, layout.q_norm, session->device_prefill_o,
          deepseek_norm_weight_dtype(layout), session->dtype, q_lora_rank,
          q_lora_rank, q_lora_rank, indexer_tokens, session->rms_eps,
          session->device_prefill_attn);
      err = cudaGetLastError();
      if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
    }

    if (err == cudaSuccess && indexer_tokens != 0) {
      const uint32_t query_heads_per_block =
          deepseek_v32_indexer_query_heads_per_block(
              layout.deepseek_index_head_dim, kDecodeThreads);
      const uint32_t query_head_blocks =
          (layout.deepseek_index_n_heads + query_heads_per_block - 1u) /
          query_heads_per_block;
      const size_t query_shared_bytes =
          q_lora_rank <= kDeepSeekV32IndexerQueryStageMaxCols
              ? static_cast<size_t>(q_lora_rank) * sizeof(float)
              : 0u;
      const dim3 query_grid(query_head_blocks, indexer_tokens);
      hf_deepseek_v32_indexer_query_state_tokens_kernel<<<
          query_grid, kDecodeThreads, query_shared_bytes,
          session->stream>>>(
          session->device_arena, layout, session->dtype, q_lora_rank,
          chunk_start, indexer_tokens, session->max_context_tokens,
          session->rope_theta, session->device_prefill_attn, q_lora_rank,
          reinterpret_cast<uint8_t *>(session->device_deepseek_indexer_state),
          deepseek_v32_indexer_query_state_layer_offset_bytes(session, 0),
          session->device_deepseek_runtime_counters);
      err = cudaGetLastError();
      if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
    }

    if (err == cudaSuccess) {
      hf_deepseek_rms_norm_f32_tokens_kernel<<<chunk_tokens,
                                               kDecodeNormThreads, 0,
                                               session->stream>>>(
          session->device_arena, layout.k_norm, session->device_prefill_qkv,
          deepseek_norm_weight_dtype(layout), session->dtype, kv_lora_rank,
          kv_a_rows, kv_lora_rank, chunk_tokens, session->rms_eps,
          session->device_prefill_norm);
      err = cudaGetLastError();
      if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
    }

    if (err == cudaSuccess) {
      hf_deepseek_v3_mla_cache_encode_tokens_kernel<<<
          chunk_tokens, kDecodeThreads, 0, session->stream>>>(
          session->device_arena, layout, 0, session->dtype, chunk_start,
          chunk_tokens, session->max_context_tokens, session->rope_theta,
          session->device_prefill_qkv, kv_a_rows, session->device_prefill_norm,
          kv_lora_rank, session->device_kv_keys, session->kv_block_count,
          session->device_kv_block_table, session->device_deepseek_v32_mla_kv,
          deepseek_v32_mla_kv_layer_offset_bytes(session, 0),
          deepseek_v32_mla_kv_block_count(session, layout));
      err = cudaGetLastError();
      if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
    }
  }

  if (err == cudaSuccess) {
    hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(
        session->device_step, prompt_token_count - 1u);
    err = cudaGetLastError();
    if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
  }
  if (err == cudaSuccess) {
    err = ensure_session_graph(session, session->max_context_tokens,
                               prompt_token_count, has_eos_token, eos_token, 0,
                               1, prompt_token_count - 1u, out);
  }
  if (err == cudaSuccess) {
    err = cudaGraphLaunch(session->cached_graph_exec, session->stream);
    if (err == cudaSuccess && out != nullptr) {
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
    if (out != nullptr) out->sync_calls += 1;
  }
  if (err == cudaSuccess && out != nullptr) {
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

bool deepseek_v3_prefill_layer_supported(
    const NervaCudaHfDecodeSequenceSession *session,
    const SequenceLayerLayout &layout, uint32_t prompt_token_count) {
  if (session == nullptr || !layout_is_deepseek_v3_mla(layout) ||
      layout.w_q == kMissingOffset || layout.q_norm == kMissingOffset ||
      layout.deepseek_q_b == kMissingOffset || layout.w_k == kMissingOffset ||
      layout.k_norm == kMissingOffset || layout.w_v == kMissingOffset ||
      layout.w_o == kMissingOffset || layout.rms_attn == kMissingOffset ||
      layout.rms_mlp == kMissingOffset || layout.deepseek_q_lora_rank == 0 ||
      layout.deepseek_kv_lora_rank == 0 ||
      layout.deepseek_qk_nope_head_dim == 0 ||
      layout.deepseek_qk_rope_head_dim == 0 ||
      layout.deepseek_v_head_dim == 0) {
    return false;
  }
  const uint32_t qk_head_dim =
      layout.deepseek_qk_nope_head_dim + layout.deepseek_qk_rope_head_dim;
  if (qk_head_dim == 0 || session->head_dim != qk_head_dim ||
      session->heads == 0) {
    return false;
  }
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  if (!bf16_storage &&
      (layout.deepseek_q_a_scale == kMissingOffset ||
       layout.deepseek_q_b_scale == kMissingOffset ||
       layout.deepseek_kv_a_scale == kMissingOffset ||
       layout.deepseek_kv_b_scale == kMissingOffset ||
       layout.deepseek_o_a_scale == kMissingOffset)) {
    return false;
  }
  const bool active_sparse_indexer =
      layout_is_deepseek_v32_indexer_query_native(layout) &&
      layout.deepseek_index_topk != 0;
  if (active_sparse_indexer &&
      (layout.deepseek_index_n_heads == 0 ||
       layout.deepseek_index_head_dim == 0 ||
       session->device_deepseek_indexer_state == nullptr ||
       session->device_deepseek_indexer_kv == nullptr)) {
    return false;
  }
  if (active_sparse_indexer &&
      layout.deepseek_index_topk < prompt_token_count) {
    const uint64_t topk_stride =
        std::min(layout.deepseek_index_topk, kDeepSeekSparseTopKSlotCapacity);
    const uint64_t chunk_tokens =
        std::min<uint64_t>(prompt_token_count, session->prefill_chunk_tokens);
    const uint64_t slot_capacity =
        session->prefill_gate_up_bytes / sizeof(int32_t);
    const uint64_t count_capacity =
        session->prefill_down_bytes / sizeof(uint32_t);
    if (topk_stride == 0 || chunk_tokens == 0 ||
        topk_stride > UINT64_MAX / chunk_tokens ||
        topk_stride * chunk_tokens > slot_capacity ||
        chunk_tokens > count_capacity) {
      return false;
    }
  }
  if (layout.mlp_kind == kMlpKindSparseMoe) {
    return layout.w_router != kMissingOffset &&
           layout.w_expert_gate_up != kMissingOffset &&
           layout.w_expert_down != kMissingOffset &&
           layout.num_experts != 0 && layout.experts_per_token != 0 &&
           layout.moe_intermediate != 0 &&
           layout.moe_intermediate <= session->intermediate;
  }
  if (layout.w_gate == kMissingOffset || layout.w_up == kMissingOffset ||
      layout.w_down == kMissingOffset) {
    return false;
  }
  if (!bf16_storage) {
    const uint64_t gate_scale =
        deepseek_f32_scale_offset(layout.w_gate, session->intermediate,
                                  session->hidden);
    const uint64_t up_scale =
        deepseek_f32_scale_offset(layout.w_up, session->intermediate,
                                  session->hidden);
    const uint64_t down_scale =
        deepseek_f32_scale_offset(layout.w_down, session->hidden,
                                  session->intermediate);
    if (gate_scale == kMissingOffset || up_scale == kMissingOffset ||
        down_scale == kMissingOffset) {
      return false;
    }
  }
  return true;
}

bool use_deepseek_v3_prefill_path(
    const NervaCudaHfDecodeSequenceSession *session,
    uint32_t prompt_token_count) {
  if (session == nullptr || prompt_token_count == 0 ||
      prompt_token_count > session->max_context_tokens ||
      session->host_layouts.size() != session->layer_count ||
      session->layer_count == 0 || session->device_prefill_hidden_a == nullptr ||
      session->device_prefill_hidden_b == nullptr ||
      session->device_prefill_norm == nullptr ||
      session->device_prefill_qkv == nullptr ||
      session->device_prefill_qkv_encoded == nullptr ||
      session->device_prefill_attn == nullptr ||
      session->device_prefill_o == nullptr ||
      session->device_prefill_gate_up == nullptr ||
      session->device_prefill_ff == nullptr ||
      session->device_prefill_down == nullptr ||
      session->device_kv_keys == nullptr || session->device_arena == nullptr ||
      session->cublas == nullptr || session->cublas_lt == nullptr) {
    return false;
  }
  for (const SequenceLayerLayout &layout : session->host_layouts) {
    if (!deepseek_v3_prefill_layer_supported(session, layout,
                                             prompt_token_count)) {
      return false;
    }
  }
  return true;
}

cudaError_t deepseek_prefill_project_tokens(
    NervaCudaHfDecodeSequenceSession *session,
    const SequenceLayerLayout &layout, uint64_t weight_offset,
    uint64_t scale_offset, const uint16_t *input, uint32_t rows,
    uint32_t cols, uint32_t tokens, float *output) {
  constexpr uint32_t block_rows = 128;
  constexpr uint32_t block_cols = 128;
  if (layout.deepseek_storage == kDeepSeekStorageBf16) {
    return project_encoded_rows(session, nullptr,
                                session->device_arena + weight_offset, input,
                                rows, cols, tokens, kDTypeBF16, 0.0f, output);
  }
  if (scale_offset == kMissingOffset) {
    return cudaErrorInvalidValue;
  }
  return launch_deepseek_fp8_f32_scale_encoded_gemm_tokens(
      session->stream, deepseek_fp8_ptr(session->device_arena, weight_offset),
      deepseek_scale_ptr(session->device_arena, scale_offset), input,
      session->dtype, rows, cols, tokens, block_rows, block_cols, output);
}

static bool deepseek_prefill_sparse_moe_has_shared_expert(
    const SequenceLayerLayout &layout) {
  return layout.shared_expert_intermediate != 0 &&
         layout.w_shared_expert_gate != kMissingOffset &&
         layout.w_shared_expert_up != kMissingOffset &&
         layout.w_shared_expert_down != kMissingOffset;
}

// Slab stride (rows of hidden per down-staging pass) for the expert-grouped
// tiled sparse-MoE prefill path; 0 when the prefill FF buffer cannot host
// the pair list, tile metadata, and at least one 64-row staging slab after
// the route ids/weights.
static uint32_t deepseek_prefill_sparse_moe_tiled_slab_stride(
    const NervaCudaHfDecodeSequenceSession *session,
    const SequenceLayerLayout &layout, uint32_t chunk_tokens) {
  if (layout.num_experts > kSparseMoeExpertsMax) {
    return 0;
  }
  const bool has_shared =
      deepseek_prefill_sparse_moe_has_shared_expert(layout);
  const DeepSeekPrefillMoeTileScratch tile_scratch =
      deepseek_prefill_moe_tile_scratch(chunk_tokens,
                                        layout.experts_per_token,
                                        layout.num_experts,
                                        has_shared ? 1u : 0u);
  if (tile_scratch.tile_capacity > 65535u) {
    return 0;
  }
  const uint64_t staging_slots =
      static_cast<uint64_t>(tile_scratch.routed_pairs) +
      tile_scratch.shared_pairs;
  const uint64_t ff_u32 = session->prefill_ff_bytes / sizeof(uint32_t);
  if (staging_slots == 0 || ff_u32 <= tile_scratch.staging_offset) {
    return 0;
  }
  uint64_t rows = (ff_u32 - tile_scratch.staging_offset) / staging_slots;
  rows = std::min<uint64_t>(rows, session->hidden);
  rows -= rows % kDeepSeekPrefillMoePairTile;
  return rows >= kDeepSeekPrefillMoePairTile ? static_cast<uint32_t>(rows)
                                             : 0u;
}

bool deepseek_prefill_sparse_moe_split_supported(
    const NervaCudaHfDecodeSequenceSession *session,
    const SequenceLayerLayout &layout, uint32_t chunk_tokens) {
  if (session == nullptr || chunk_tokens == 0 ||
      session->device_prefill_ff == nullptr ||
      session->device_prefill_gate_up == nullptr ||
      session->device_prefill_down == nullptr ||
      layout.mlp_kind != kMlpKindSparseMoe || layout.num_experts == 0 ||
      layout.experts_per_token == 0 || layout.moe_intermediate == 0 ||
      layout.moe_intermediate > session->intermediate ||
      layout.w_router == kMissingOffset ||
      layout.w_expert_gate_up == kMissingOffset ||
      layout.w_expert_down == kMissingOffset) {
    return false;
  }
  const uint64_t top_k = layout.experts_per_token;
  const uint64_t moe_intermediate = layout.moe_intermediate;
  const uint64_t shared_intermediate =
      layout.shared_expert_intermediate == 0 ? 0 : layout.shared_expert_intermediate;
  if (shared_intermediate > session->intermediate) {
    return false;
  }
  const uint64_t rank_ff_rows =
      sat_add_u64(sat_mul_u64(top_k, moe_intermediate), shared_intermediate);
  const uint64_t rank_ff_bytes =
      sat_mul_u64(sat_mul_u64(rank_ff_rows, chunk_tokens), sizeof(float));
  if (rank_ff_bytes > session->prefill_gate_up_bytes) {
    return false;
  }
  const uint64_t route_slots = sat_mul_u64(chunk_tokens, top_k);
  const uint64_t route_bytes =
      sat_mul_u64(route_slots, sizeof(uint32_t) + sizeof(float));
  if (route_bytes > session->prefill_ff_bytes) {
    return false;
  }
  return deepseek_prefill_sparse_moe_tiled_slab_stride(session, layout,
                                                       chunk_tokens) != 0;
}

cudaError_t launch_deepseek_prefill_sparse_moe_split(
    NervaCudaHfDecodeSequenceSession *session,
    const SequenceLayerLayout &layout, uint32_t chunk_tokens,
    NervaCudaHfDecodeSequenceResult *out) {
  if (!deepseek_prefill_sparse_moe_split_supported(session, layout,
                                                   chunk_tokens)) {
    return cudaErrorInvalidValue;
  }
  constexpr uint32_t threads = kDecodeThreads;
  const bool has_shared_expert =
      deepseek_prefill_sparse_moe_has_shared_expert(layout);
  const DeepSeekPrefillMoeTileScratch tile_scratch =
      deepseek_prefill_moe_tile_scratch(chunk_tokens,
                                        layout.experts_per_token,
                                        layout.num_experts,
                                        has_shared_expert ? 1u : 0u);
  const uint32_t slab_stride = deepseek_prefill_sparse_moe_tiled_slab_stride(
      session, layout, chunk_tokens);
  if (slab_stride == 0) {
    return cudaErrorInvalidValue;
  }
  const uint32_t tile_grid_y =
      static_cast<uint32_t>(tile_scratch.tile_capacity);

  float *router_logits_tokens = session->device_prefill_gate_up;
  const dim3 router_grid(layout.num_experts, chunk_tokens);
  hf_deepseek_prefill_sparse_moe_router_logits_kernel<<<
      router_grid, threads, 0, session->stream>>>(
      session->device_arena, layout, session->dtype, session->hidden,
      chunk_tokens, session->device_prefill_norm, router_logits_tokens);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;

  if (err == cudaSuccess) {
    hf_deepseek_prefill_sparse_moe_route_kernel<<<
        chunk_tokens, threads, 0, session->stream>>>(
        session->device_arena, layout, session->dtype, session->hidden,
        session->intermediate, chunk_tokens, router_logits_tokens,
        session->device_prefill_ff, session->device_deepseek_runtime_counters);
    err = cudaGetLastError();
    if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
  }

  if (err == cudaSuccess) {
    hf_deepseek_prefill_sparse_moe_build_pairs_kernel<<<
        1, threads, 0, session->stream>>>(
        layout, chunk_tokens, has_shared_expert ? 1u : 0u,
        session->device_prefill_ff);
    err = cudaGetLastError();
    if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
  }
  if (err == cudaSuccess) {
    const uint32_t gate_rows =
        std::max(layout.moe_intermediate,
                 has_shared_expert ? layout.shared_expert_intermediate : 0u);
    const dim3 gate_grid(
        ceil_div_u32(gate_rows, kDeepSeekPrefillMoePairTile), tile_grid_y);
    hf_deepseek_prefill_sparse_moe_gate_up_tiles_kernel<<<
        gate_grid, threads, 0, session->stream>>>(
        session->device_arena, layout, session->dtype, session->hidden,
        session->intermediate, chunk_tokens, has_shared_expert ? 1u : 0u,
        session->device_prefill_norm, session->device_prefill_ff,
        session->device_prefill_gate_up);
    err = cudaGetLastError();
    if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
  }
  for (uint32_t slab_base = 0;
       err == cudaSuccess && slab_base < session->hidden;
       slab_base += slab_stride) {
    const uint32_t slab_rows =
        std::min(slab_stride, session->hidden - slab_base);
    const dim3 down_grid(
        ceil_div_u32(slab_rows, kDeepSeekPrefillMoePairTile), tile_grid_y);
    hf_deepseek_prefill_sparse_moe_down_tiles_kernel<<<
        down_grid, threads, 0, session->stream>>>(
        session->device_arena, layout, session->hidden,
        session->intermediate, chunk_tokens, has_shared_expert ? 1u : 0u,
        slab_base, slab_rows, slab_stride, session->device_prefill_ff,
        session->device_prefill_gate_up);
    err = cudaGetLastError();
    if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
    if (err == cudaSuccess) {
      const dim3 combine_grid(ceil_div_u32(slab_rows, threads),
                              chunk_tokens);
      hf_deepseek_prefill_sparse_moe_down_combine_kernel<<<
          combine_grid, threads, 0, session->stream>>>(
          layout, session->hidden, chunk_tokens,
          has_shared_expert ? 1u : 0u, slab_base, slab_rows, slab_stride,
          session->device_prefill_ff, session->device_prefill_down);
      err = cudaGetLastError();
      if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
    }
  }
  return err;
}

cudaError_t launch_deepseek_v3_session_prefill(
    NervaCudaHfDecodeSequenceSession *session, uint32_t prompt_token_count,
    uint32_t has_eos_token, uint32_t eos_token,
    NervaCudaHfDecodeSequenceResult *out) {
  if (!use_deepseek_v3_prefill_path(session, prompt_token_count)) {
    return cudaErrorInvalidValue;
  }
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
    if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
  }

  uint16_t *hidden_in = session->device_prefill_hidden_a;
  uint16_t *hidden_out = session->device_prefill_hidden_b;
  for (uint32_t layer_index = 0;
       err == cudaSuccess && layer_index < session->layer_count;
       ++layer_index) {
    const SequenceLayerLayout layout = session->host_layouts[layer_index];
    const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
    const uint32_t q_lora_rank = layout.deepseek_q_lora_rank;
    const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
    const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
    const uint32_t qk_head_dim =
        layout.deepseek_qk_nope_head_dim + layout.deepseek_qk_rope_head_dim;
    const uint32_t q_rows = session->heads * qk_head_dim;
    const uint32_t kv_a_rows = kv_lora_rank + qk_rope;
    const uint32_t value_rows = session->heads * layout.deepseek_v_head_dim;
    const bool active_sparse_indexer =
        layout_is_deepseek_v32_indexer_query_native(layout) &&
        layout.deepseek_index_topk != 0;
    const bool use_sparse_prefill_topk =
        active_sparse_indexer &&
        prompt_token_count > layout.deepseek_index_topk;
    const uint32_t sparse_topk_stride =
        use_sparse_prefill_topk
            ? std::min(layout.deepseek_index_topk,
                       kDeepSeekSparseTopKSlotCapacity)
            : 0u;

    for (uint32_t chunk_start = 0;
         err == cudaSuccess && chunk_start < prompt_token_count;
         chunk_start += session->prefill_chunk_tokens) {
      const uint32_t chunk_tokens =
          std::min(session->prefill_chunk_tokens,
                   prompt_token_count - chunk_start);
      uint16_t *q_norm_tokens = session->device_prefill_qkv_encoded;
      uint16_t *k_norm_tokens =
          q_norm_tokens + static_cast<uint64_t>(q_lora_rank) * chunk_tokens;

      if (err == cudaSuccess) err = profile_stage_begin();
      if (err == cudaSuccess) {
        hf_prefill_attn_norm_kernel<<<chunk_tokens, kDecodeThreads, 0,
                                      session->stream>>>(
            session->device_arena, layout, session->dtype,
            deepseek_norm_weight_dtype(layout), session->hidden,
            chunk_start, chunk_tokens, session->rms_eps, hidden_in,
            session->device_prefill_norm);
        err = cudaGetLastError();
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&norm_ns);

      if (err == cudaSuccess && active_sparse_indexer) {
        hf_deepseek_v32_indexer_kv_encode_tokens_kernel<<<
            chunk_tokens, kDecodeThreads, 0, session->stream>>>(
            session->device_arena, layout, session->dtype, session->hidden,
            chunk_start, chunk_tokens, session->max_context_tokens,
            session->rope_theta, session->device_prefill_norm,
            session->hidden, session->device_deepseek_indexer_kv,
            deepseek_v32_indexer_kv_layer_offset_bytes(session, layer_index),
            deepseek_v32_indexer_kv_block_count(session, layout),
            session->kv_block_count, session->device_kv_block_table,
            session->device_deepseek_runtime_counters);
        err = cudaGetLastError();
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      if (err == cudaSuccess && use_sparse_prefill_topk) {
        const dim3 weight_grid(layout.deepseek_index_n_heads, chunk_tokens);
        hf_deepseek_v32_indexer_weight_state_tokens_kernel<<<
            weight_grid, kDecodeThreads, 0, session->stream>>>(
            session->device_arena, layout, session->dtype, session->hidden,
            chunk_start, chunk_tokens, session->max_context_tokens,
            session->device_prefill_norm, session->hidden,
            reinterpret_cast<uint8_t *>(session->device_deepseek_indexer_state),
            deepseek_v32_indexer_query_state_layer_offset_bytes(session,
                                                                layer_index));
        err = cudaGetLastError();
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }

      if (err == cudaSuccess) err = profile_stage_begin();
      if (err == cudaSuccess) {
        err = deepseek_prefill_project_tokens(
            session, layout, layout.w_q, layout.deepseek_q_a_scale,
            session->device_prefill_norm, q_lora_rank, session->hidden,
            chunk_tokens, session->device_prefill_o);
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      if (err == cudaSuccess) {
        err = deepseek_prefill_project_tokens(
            session, layout, layout.w_k, layout.deepseek_kv_a_scale,
            session->device_prefill_norm, kv_a_rows, session->hidden,
            chunk_tokens, session->device_prefill_gate_up);
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&qkv_projection_ns);

      if (err == cudaSuccess) err = profile_stage_begin();
      if (err == cudaSuccess) {
        hf_deepseek_rms_norm_f32_tokens_kernel<<<chunk_tokens,
                                                 kDecodeNormThreads, 0,
                                                 session->stream>>>(
            session->device_arena, layout.q_norm, session->device_prefill_o,
            deepseek_norm_weight_dtype(layout), session->dtype, q_lora_rank,
            q_lora_rank, q_lora_rank, chunk_tokens, session->rms_eps,
            q_norm_tokens);
        err = cudaGetLastError();
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      if (err == cudaSuccess) {
        hf_deepseek_rms_norm_f32_tokens_kernel<<<chunk_tokens,
                                                 kDecodeNormThreads, 0,
                                                 session->stream>>>(
            session->device_arena, layout.k_norm,
            session->device_prefill_gate_up,
            deepseek_norm_weight_dtype(layout), session->dtype, kv_lora_rank,
            kv_a_rows, kv_lora_rank, chunk_tokens, session->rms_eps,
            k_norm_tokens);
        err = cudaGetLastError();
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      if (err == cudaSuccess && use_sparse_prefill_topk) {
        const uint32_t query_heads_per_block =
            deepseek_v32_indexer_query_heads_per_block(
                layout.deepseek_index_head_dim, kDecodeThreads);
        const uint32_t query_head_blocks =
            (layout.deepseek_index_n_heads + query_heads_per_block - 1u) /
            query_heads_per_block;
        const size_t query_shared_bytes =
            q_lora_rank <= kDeepSeekV32IndexerQueryStageMaxCols
                ? static_cast<size_t>(q_lora_rank) * sizeof(float)
                : 0u;
        const dim3 query_grid(query_head_blocks, chunk_tokens);
        hf_deepseek_v32_indexer_query_state_tokens_kernel<<<
            query_grid, kDecodeThreads, query_shared_bytes,
            session->stream>>>(
            session->device_arena, layout, session->dtype, q_lora_rank,
            chunk_start, chunk_tokens, session->max_context_tokens,
            session->rope_theta, q_norm_tokens, q_lora_rank,
            reinterpret_cast<uint8_t *>(session->device_deepseek_indexer_state),
            deepseek_v32_indexer_query_state_layer_offset_bytes(session,
                                                                layer_index),
            session->device_deepseek_runtime_counters);
        err = cudaGetLastError();
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&norm_ns);

      if (err == cudaSuccess) err = profile_stage_begin();
      if (err == cudaSuccess) {
        err = deepseek_prefill_project_tokens(
            session, layout, layout.deepseek_q_b,
            layout.deepseek_q_b_scale, q_norm_tokens, q_rows, q_lora_rank,
            chunk_tokens, session->device_prefill_qkv);
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&qkv_projection_ns);

      if (err == cudaSuccess) err = profile_stage_begin();
      if (err == cudaSuccess) {
        hf_deepseek_v3_mla_cache_encode_tokens_kernel<<<
            chunk_tokens, kDecodeThreads, 0, session->stream>>>(
            session->device_arena, layout, layer_index, session->dtype,
            chunk_start, chunk_tokens, session->max_context_tokens,
            session->rope_theta, session->device_prefill_gate_up, kv_a_rows,
            k_norm_tokens, kv_lora_rank, session->device_kv_keys,
            session->kv_block_count, session->device_kv_block_table,
            session->device_deepseek_v32_mla_kv,
            deepseek_v32_mla_kv_layer_offset_bytes(session, layer_index),
            deepseek_v32_mla_kv_block_count(session, layout));
        err = cudaGetLastError();
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      int32_t *sparse_topk_slots =
          use_sparse_prefill_topk
              ? reinterpret_cast<int32_t *>(session->device_prefill_gate_up)
              : nullptr;
      uint32_t *sparse_topk_counts =
          use_sparse_prefill_topk
              ? reinterpret_cast<uint32_t *>(session->device_prefill_down)
              : nullptr;
      if (err == cudaSuccess && use_sparse_prefill_topk) {
        const uint64_t prefill_down_floats =
            session->prefill_down_bytes / sizeof(float);
        const uint64_t score_workspace_offset = chunk_tokens;
        const uint64_t score_workspace_floats =
            prefill_down_floats > score_workspace_offset
                ? prefill_down_floats - score_workspace_offset
                : 0u;
        const uint64_t score_workspace_needed =
            static_cast<uint64_t>(chunk_tokens) * session->max_context_tokens;
        if (score_workspace_floats >= score_workspace_needed) {
          hf_deepseek_v32_sparse_topk_select_tokens_parallel_kernel<<<
              chunk_tokens, kDecodeThreads, 0, session->stream>>>(
              layout, chunk_start, chunk_tokens, session->max_context_tokens,
              reinterpret_cast<const uint8_t *>(
                  session->device_deepseek_indexer_state),
              deepseek_v32_indexer_query_state_layer_offset_bytes(session,
                                                                  layer_index),
              session->device_deepseek_indexer_kv,
              deepseek_v32_indexer_kv_layer_offset_bytes(session, layer_index),
              deepseek_v32_indexer_kv_block_count(session, layout),
              session->kv_block_count, session->device_kv_block_table,
              sparse_topk_slots, sparse_topk_stride, sparse_topk_counts,
              session->device_prefill_down + score_workspace_offset,
              session->max_context_tokens,
              session->device_deepseek_runtime_counters);
        } else {
          hf_deepseek_v32_sparse_topk_select_tokens_kernel<<<
              chunk_tokens, 1, 0, session->stream>>>(
              layout, chunk_start, chunk_tokens, session->max_context_tokens,
              reinterpret_cast<const uint8_t *>(
                  session->device_deepseek_indexer_state),
              deepseek_v32_indexer_query_state_layer_offset_bytes(session,
                                                                  layer_index),
              session->device_deepseek_indexer_kv,
              deepseek_v32_indexer_kv_layer_offset_bytes(session, layer_index),
              deepseek_v32_indexer_kv_block_count(session, layout),
              session->kv_block_count, session->device_kv_block_table,
              sparse_topk_slots, sparse_topk_stride, sparse_topk_counts,
              session->device_deepseek_runtime_counters);
        }
        err = cudaGetLastError();
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      if (err == cudaSuccess) {
        const bool grouped_attention =
            kv_lora_rank <=
                kDeepSeekMlaPrefillMaxLatentSlots * kDecodeThreads &&
            qk_rope <= kDecodeThreads;
        if (grouped_attention) {
          const size_t shared_bytes =
              static_cast<size_t>(kDeepSeekMlaPrefillHeadGroup) *
              kv_lora_rank * sizeof(float);
          const dim3 grid(chunk_tokens,
                          (session->heads + kDeepSeekMlaPrefillHeadGroup -
                           1u) /
                              kDeepSeekMlaPrefillHeadGroup);
          hf_deepseek_v3_mla_attention_tokens_grouped_kernel<<<
              grid, kDecodeThreads, shared_bytes, session->stream>>>(
              session->device_arena, layout, layer_index, session->dtype,
              session->heads, session->max_context_tokens,
              session->rope_theta, chunk_start, chunk_tokens,
              session->device_prefill_qkv, q_rows, session->device_kv_keys,
              session->kv_block_count, session->device_kv_block_table,
              session->device_prefill_attn, value_rows, sparse_topk_slots,
              sparse_topk_stride, sparse_topk_counts,
              session->device_deepseek_runtime_counters);
        } else {
          const size_t shared_bytes =
              (static_cast<size_t>(kv_lora_rank) * 2u + qk_rope) *
              sizeof(float);
          const dim3 grid(chunk_tokens, session->heads);
          hf_deepseek_v3_mla_attention_tokens_kernel<<<
              grid, kDecodeThreads, shared_bytes, session->stream>>>(
              session->device_arena, layout, layer_index, session->dtype,
              session->heads, session->max_context_tokens,
              session->rope_theta, chunk_start, chunk_tokens,
              session->device_prefill_qkv, q_rows, session->device_kv_keys,
              session->kv_block_count, session->device_kv_block_table,
              session->device_prefill_attn, value_rows, sparse_topk_slots,
              sparse_topk_stride, sparse_topk_counts,
              session->device_deepseek_runtime_counters);
        }
        err = cudaGetLastError();
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&attention_ns);

      if (err == cudaSuccess) err = profile_stage_begin();
      if (err == cudaSuccess) {
        err = deepseek_prefill_project_tokens(
            session, layout, layout.w_o, layout.deepseek_o_a_scale,
            session->device_prefill_attn, session->hidden, value_rows,
            chunk_tokens, session->device_prefill_o);
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      if (err == cudaSuccess)
        err = profile_stage_end(&attention_output_projection_ns);

      if (err == cudaSuccess) err = profile_stage_begin();
      if (err == cudaSuccess) {
        hf_deepseek_prefill_mlp_norm_kernel<<<chunk_tokens, kDecodeThreads, 0,
                                              session->stream>>>(
            session->device_arena, layout, session->dtype,
            deepseek_norm_weight_dtype(layout), session->hidden,
            chunk_start, chunk_tokens, session->rms_eps, hidden_in,
            session->device_prefill_o, session->device_prefill_norm);
        err = cudaGetLastError();
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&norm_ns);

      if (layout.mlp_kind == kMlpKindSparseMoe) {
        if (err == cudaSuccess) err = profile_stage_begin();
        if (err == cudaSuccess) {
          if (deepseek_prefill_sparse_moe_split_supported(session, layout,
                                                          chunk_tokens)) {
            err = launch_deepseek_prefill_sparse_moe_split(session, layout,
                                                           chunk_tokens, out);
          } else {
            hf_deepseek_prefill_sparse_moe_kernel<<<
                chunk_tokens, kDecodeThreads, 0, session->stream>>>(
                session->device_arena, layout, session->dtype, session->hidden,
                session->intermediate, chunk_tokens,
                session->device_prefill_norm, session->device_prefill_gate_up,
                session->device_prefill_down,
                session->device_deepseek_runtime_counters);
            err = cudaGetLastError();
            if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
          }
        }
        if (err == cudaSuccess) err = profile_stage_end(&mlp_ns);
      } else {
        const uint64_t gate_scale =
            bf16_storage
                ? kMissingOffset
                : deepseek_f32_scale_offset(layout.w_gate,
                                            session->intermediate,
                                            session->hidden);
        const uint64_t up_scale =
            bf16_storage
                ? kMissingOffset
                : deepseek_f32_scale_offset(layout.w_up,
                                            session->intermediate,
                                            session->hidden);
        const uint64_t down_scale =
            bf16_storage
                ? kMissingOffset
                : deepseek_f32_scale_offset(layout.w_down, session->hidden,
                                            session->intermediate);
        if (err == cudaSuccess) err = profile_stage_begin();
        if (err == cudaSuccess) {
          err = deepseek_prefill_project_tokens(
              session, layout, layout.w_gate, gate_scale,
              session->device_prefill_norm, session->intermediate,
              session->hidden, chunk_tokens, session->device_prefill_gate_up);
          if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
        }
        if (err == cudaSuccess) {
          err = deepseek_prefill_project_tokens(
              session, layout, layout.w_up, up_scale,
              session->device_prefill_norm, session->intermediate,
              session->hidden, chunk_tokens, session->device_prefill_qkv);
          if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
        }
        if (err == cudaSuccess) err = profile_stage_end(&gate_up_projection_ns);
        if (err == cudaSuccess) err = profile_stage_begin();
        if (err == cudaSuccess) {
          const uint32_t blocks = static_cast<uint32_t>(
              (static_cast<uint64_t>(chunk_tokens) * session->intermediate +
               kDecodeThreads - 1u) /
              kDecodeThreads);
          hf_deepseek_prefill_ff_split_kernel<<<blocks, kDecodeThreads, 0,
                                                session->stream>>>(
              layout, session->dtype, session->intermediate, chunk_tokens,
              session->device_prefill_gate_up, session->device_prefill_qkv,
              session->device_prefill_ff);
          err = cudaGetLastError();
          if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
        }
        if (err == cudaSuccess) err = profile_stage_end(&mlp_ns);
        if (err == cudaSuccess) err = profile_stage_begin();
        if (err == cudaSuccess) {
          err = deepseek_prefill_project_tokens(
              session, layout, layout.w_down, down_scale,
              session->device_prefill_ff, session->hidden,
              session->intermediate, chunk_tokens,
              session->device_prefill_down);
          if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
        }
        if (err == cudaSuccess) err = profile_stage_end(&down_projection_ns);
      }

      if (err == cudaSuccess) err = profile_stage_begin();
      if (err == cudaSuccess) {
        const uint32_t blocks = static_cast<uint32_t>(
            (static_cast<uint64_t>(chunk_tokens) * session->hidden +
             kDecodeThreads - 1u) /
            kDecodeThreads);
        hf_prefill_finish_kernel<<<blocks, kDecodeThreads, 0,
                                   session->stream>>>(
            session->dtype, session->hidden, chunk_start, chunk_tokens,
            session->device_prefill_o, session->device_prefill_down,
            hidden_out);
        err = cudaGetLastError();
        if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&mlp_ns);
    }
    std::swap(hidden_in, hidden_out);
  }

  if (err == cudaSuccess) err = profile_stage_begin();
  if (err == cudaSuccess) {
    const uint32_t final_norm_weight_dtype =
        final_norm_weight_dtype_for_layouts(session->host_layouts.data(),
                                            session->layer_count,
                                            session->dtype);
    hf_prefill_final_norm_last_kernel<<<1, kDecodeThreads, 0,
                                        session->stream>>>(
        session->device_arena, session->arena_layout, session->dtype,
        final_norm_weight_dtype, session->hidden, prompt_token_count,
        session->rms_eps, hidden_in, session->device_projection_input);
    err = cudaGetLastError();
    if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
  }
  if (err == cudaSuccess) err = profile_stage_end(&norm_ns);
  if (err == cudaSuccess) {
    hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(
        session->device_step, prompt_token_count - 1u);
    err = cudaGetLastError();
    if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
  }
  if (err == cudaSuccess) err = profile_stage_begin();
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = project_encoded_rows(
        session, &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        1, session->dtype, 0.0f, device_logits);
    if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
  }
  if (err == cudaSuccess) err = profile_stage_end(&lm_head_projection_ns);
  if (err == cudaSuccess) err = profile_stage_begin();
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        session->stream, session->device_step, session->max_context_tokens,
        has_eos_token, eos_token, device_logits, session->vocab_size,
        session->device_slots, session->active_sampler);
    if (err == cudaSuccess && out != nullptr) out->kernel_launches += 1;
  }
  if (err == cudaSuccess) err = profile_stage_end(&sampling_ns);
  if (err == cudaSuccess) {
    err = cudaEventRecord(session->device_stop, session->stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(session->stream);
    if (out != nullptr) out->sync_calls += 1;
  }
  if (err == cudaSuccess && out != nullptr) {
    float device_ms = 0.0f;
    err = cudaEventElapsedTime(&device_ms, session->device_start,
                               session->device_stop);
    if (err == cudaSuccess && device_ms > 0.0f) {
      out->device_elapsed_ns = static_cast<uint64_t>(device_ms * 1000000.0f);
      if (out->device_elapsed_ns == 0) out->device_elapsed_ns = 1;
    }
  }
  if (err == cudaSuccess && collect_profile && out != nullptr) {
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
            session->device_arena, layout, session->dtype,
            layer_norm_weight_dtype(layout, session->dtype), session->hidden,
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
            session->device_arena, layout, session->dtype,
            layer_norm_weight_dtype(layout, session->dtype), session->hidden,
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
    const uint32_t final_norm_weight_dtype =
        final_norm_weight_dtype_for_layouts(session->host_layouts.data(),
                                            session->layer_count,
                                            session->dtype);
    hf_prefill_final_norm_last_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, session->dtype,
        final_norm_weight_dtype, session->hidden, prompt_token_count,
        session->rms_eps, hidden_in, session->device_projection_input);
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
  cudaError_t err = cudaSuccess;
  if (err == cudaSuccess) {
    err = cudaEventRecord(session->device_start, session->stream);
  }
  if (prompt_token_count > 1) {
    err = ensure_session_graph(session, session->max_context_tokens,
                               prompt_token_count, has_eos_token, eos_token, 0,
                               0, 0, out);
  }
  for (uint32_t step = 0;
       err == cudaSuccess && step + 1u < prompt_token_count; ++step) {
    err = cudaGraphLaunch(session->cached_graph_exec, session->stream);
    if (err == cudaSuccess) {
      out->graph_replays += 1;
      out->graph_launches += 1;
      out->kernel_launches += out->graph_nodes == 0 ? 1 : out->graph_nodes;
    }
  }
  if (err == cudaSuccess) {
    err = ensure_session_graph(session, session->max_context_tokens,
                               prompt_token_count, has_eos_token, eos_token, 0,
                               1, prompt_token_count - 1u, out);
  }
  if (err == cudaSuccess) {
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
