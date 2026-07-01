cudaError_t launch_deepseek_v3_mla_projection_step(
    NervaCudaHfDecodeSequenceSession *session, const SequenceLayerLayout &layout,
    uint32_t layer_index, uint32_t max_steps, uint32_t attention_chunks,
    const DeepseekDecodeProfileBuckets *profile = nullptr) {
  if (!layout_is_deepseek_v3_mla(layout) ||
      layout.w_q == kMissingOffset ||
      layout.q_norm == kMissingOffset ||
      layout.deepseek_q_b == kMissingOffset ||
      layout.w_k == kMissingOffset ||
      layout.k_norm == kMissingOffset ||
      layout.w_v == kMissingOffset ||
      layout.w_o == kMissingOffset) {
    return cudaErrorInvalidValue;
  }
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  if (!bf16_storage &&
      (layout.deepseek_q_a_scale == kMissingOffset ||
       layout.deepseek_q_b_scale == kMissingOffset ||
       layout.deepseek_kv_a_scale == kMissingOffset ||
       layout.deepseek_kv_b_scale == kMissingOffset ||
       layout.deepseek_o_a_scale == kMissingOffset)) {
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

  cudaError_t err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  if (bf16_storage) {
    err = project_encoded_rows(
        session, nullptr, session->device_arena + layout.w_q,
        session->device_projection_input, q_lora_rank, session->hidden, 1,
        kDTypeBF16, 0.0f, scratch.q);
    if (err == cudaSuccess) {
      err = project_encoded_rows(
          session, nullptr, session->device_arena + layout.w_k,
          session->device_projection_input, kv_a_rows, session->hidden, 1,
          kDTypeBF16, 0.0f, scratch.k);
    }
  } else {
    err = launch_deepseek_fp8_f32_scale_dual_encoded_matvec_varrows(
        session->stream, deepseek_fp8_ptr(session->device_arena, layout.w_q),
        deepseek_scale_ptr(session->device_arena, layout.deepseek_q_a_scale),
        deepseek_fp8_ptr(session->device_arena, layout.w_k),
        deepseek_scale_ptr(session->device_arena, layout.deepseek_kv_a_scale),
        session->device_projection_input, session->dtype, q_lora_rank,
        kv_a_rows, session->hidden, block_rows, block_cols, scratch.q,
        scratch.k);
  }
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->qkv_projection_ns);
  if (err != cudaSuccess) return err;

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  hf_deepseek_v32_indexer_kv_encode_kernel<<<1, kDecodeThreads, 0,
                                              session->stream>>>(
      session->device_arena, layout, session->dtype, session->hidden,
      session->device_step, max_steps, session->rope_theta,
      session->device_projection_input, session->device_deepseek_indexer_kv,
      deepseek_v32_indexer_kv_layer_offset_bytes(session, layer_index),
      deepseek_v32_indexer_kv_block_count(session, layout),
      session->kv_block_count, session->device_kv_block_table,
      session->device_deepseek_runtime_counters);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  const uint32_t indexer_weight_blocks =
      layout.deepseek_index_n_heads == 0 ? 1u : layout.deepseek_index_n_heads;
  hf_deepseek_v32_indexer_weight_state_kernel<<<indexer_weight_blocks,
                                                 kDecodeThreads, 0,
                                                 session->stream>>>(
      session->device_arena, layout, session->dtype, session->hidden,
      session->device_step, max_steps, session->device_projection_input,
      reinterpret_cast<uint8_t *>(session->device_deepseek_indexer_state),
      deepseek_v32_indexer_query_state_layer_offset_bytes(session,
                                                          layer_index));
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->attention_ns);
  if (err != cudaSuccess) return err;

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  hf_decode_rms_norm_f32_to_encoded_kernel<<<1, kDecodeNormThreads, 0,
                                             session->stream>>>(
      session->device_arena, layout.q_norm, scratch.q,
      deepseek_norm_weight_dtype(layout), session->dtype, q_lora_rank,
      session->device_step, max_steps, session->rms_eps,
      session->device_projection_input);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->norm_ns);
  if (err != cudaSuccess) return err;

  const uint32_t indexer_query_blocks =
      layout.deepseek_index_n_heads == 0 ? 1u : layout.deepseek_index_n_heads;
  const bool has_v32_indexer_query =
      layout.attention_kind == kAttentionKindDeepSeekMla &&
      layout.deepseek_mode == kDeepSeekModeV32MlaIndexer &&
      (layout.deepseek_flags & kDeepSeekFlagSparseIndexer) != 0 &&
      layout.deepseek_q_lora_rank == q_lora_rank &&
      layout.deepseek_index_n_heads != 0 &&
      layout.deepseek_index_head_dim != 0 &&
      layout.deepseek_indexer_q != kMissingOffset &&
      layout.deepseek_indexer_q_scale != kMissingOffset &&
      layout.deepseek_indexer_weights != kMissingOffset &&
      session->device_deepseek_indexer_state != nullptr;
  const uint32_t indexer_query_rows =
      layout.deepseek_index_n_heads * layout.deepseek_index_head_dim;
  if (has_v32_indexer_query) {
    err = deepseek_profile_begin_if(session, profile);
    if (err != cudaSuccess) return err;
    if (bf16_storage) {
      err = project_encoded_rows(
          session, nullptr, session->device_arena + layout.deepseek_q_b,
          session->device_projection_input, q_rows, q_lora_rank, 1,
          kDTypeBF16, 0.0f, scratch.q);
      if (err == cudaSuccess) {
        err = project_encoded_rows(
            session, nullptr,
            session->device_arena + layout.deepseek_indexer_q,
            session->device_projection_input, indexer_query_rows,
            q_lora_rank, 1, kDTypeBF16, 0.0f, scratch.attn);
      }
    } else {
      err = launch_deepseek_fp8_f32_scale_dual_encoded_matvec_varrows(
          session->stream,
          deepseek_fp8_ptr(session->device_arena, layout.deepseek_q_b),
          deepseek_scale_ptr(session->device_arena, layout.deepseek_q_b_scale),
          deepseek_fp8_ptr(session->device_arena, layout.deepseek_indexer_q),
          deepseek_scale_ptr(session->device_arena,
                             layout.deepseek_indexer_q_scale),
          session->device_projection_input, session->dtype, q_rows,
          indexer_query_rows, q_lora_rank, block_rows, block_cols, scratch.q,
          scratch.attn);
    }
    if (err != cudaSuccess) return err;
    err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                      ? nullptr
                                                      : profile->qkv_projection_ns);
    if (err != cudaSuccess) return err;
  }

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  if (has_v32_indexer_query) {
    hf_deepseek_v32_indexer_query_state_projected_kernel<<<
        indexer_query_blocks, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, layout, session->device_step, max_steps,
        session->rope_theta, scratch.attn,
        reinterpret_cast<uint8_t *>(session->device_deepseek_indexer_state),
        deepseek_v32_indexer_query_state_layer_offset_bytes(session,
                                                            layer_index),
        session->device_deepseek_runtime_counters);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  } else {
    hf_deepseek_v32_indexer_query_state_kernel<<<indexer_query_blocks,
                                                  kDecodeThreads, 0,
                                                  session->stream>>>(
        session->device_arena, layout, session->dtype, q_lora_rank,
        session->device_step, max_steps, session->rope_theta,
        session->device_projection_input,
        reinterpret_cast<uint8_t *>(session->device_deepseek_indexer_state),
        deepseek_v32_indexer_query_state_layer_offset_bytes(session,
                                                            layer_index),
        session->device_deepseek_runtime_counters);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->attention_ns);
  if (err != cudaSuccess) return err;

  if (!has_v32_indexer_query) {
    err = deepseek_profile_begin_if(session, profile);
    if (err != cudaSuccess) return err;
    if (bf16_storage) {
      err = project_encoded_rows(
          session, nullptr, session->device_arena + layout.deepseek_q_b,
          session->device_projection_input, q_rows, q_lora_rank, 1,
          kDTypeBF16, 0.0f, scratch.q);
    } else {
      err = launch_deepseek_fp8_f32_scale_encoded_matvec(
          session->stream,
          deepseek_fp8_ptr(session->device_arena, layout.deepseek_q_b),
          deepseek_scale_ptr(session->device_arena, layout.deepseek_q_b_scale),
          session->device_projection_input, session->dtype, q_rows,
          q_lora_rank, block_rows, block_cols, scratch.q);
    }
    if (err != cudaSuccess) return err;
    err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                      ? nullptr
                                                      : profile->qkv_projection_ns);
    if (err != cudaSuccess) return err;
  }

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  hf_decode_rms_norm_f32_to_encoded_kernel<<<1, kDecodeNormThreads, 0,
                                             session->stream>>>(
      session->device_arena, layout.k_norm, scratch.k,
      deepseek_norm_weight_dtype(layout), session->dtype, kv_lora_rank,
      session->device_step, max_steps, session->rms_eps,
      session->device_projection_input);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->norm_ns);
  if (err != cudaSuccess) return err;

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  hf_deepseek_v3_mla_cache_encode_kernel<<<1, kDecodeThreads, 0,
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

  const bool has_v32_sparse_attention =
      layout.attention_kind == kAttentionKindDeepSeekMla &&
      layout.deepseek_mode == kDeepSeekModeV32MlaIndexer &&
      (layout.deepseek_flags & kDeepSeekFlagSparseIndexer) != 0 &&
      layout.deepseek_index_topk != 0 &&
      session->device_deepseek_indexer_state != nullptr &&
      session->device_deepseek_indexer_kv != nullptr &&
      layout.deepseek_q_lora_rank != 0 &&
      layout.deepseek_index_n_heads != 0 &&
      layout.deepseek_index_head_dim != 0 &&
      layout.deepseek_indexer_q != kMissingOffset &&
      layout.deepseek_indexer_q_scale != kMissingOffset &&
      layout.deepseek_indexer_weights != kMissingOffset &&
      deepseek_v32_indexer_kv_block_count(session, layout) != 0;
  const bool has_precomputed_v32_sparse_attention =
      has_v32_sparse_attention &&
      session->device_deepseek_sparse_topk_slots != nullptr &&
      session->device_deepseek_sparse_topk_count != nullptr;
  if (has_precomputed_v32_sparse_attention) {
    hf_deepseek_v32_sparse_topk_select_kernel<<<1, kDecodeThreads, 0,
                                                 session->stream>>>(
        layout, session->device_step, max_steps,
        reinterpret_cast<const uint8_t *>(session->device_deepseek_indexer_state),
        deepseek_v32_indexer_query_state_layer_offset_bytes(session,
                                                            layer_index),
        session->device_deepseek_indexer_kv,
        deepseek_v32_indexer_kv_layer_offset_bytes(session, layer_index),
        deepseek_v32_indexer_kv_block_count(session, layout),
        session->kv_block_count, session->device_kv_block_table,
        session->device_deepseek_sparse_topk_slots,
        session->device_deepseek_sparse_topk_count,
        session->device_deepseek_sparse_topk_scores,
        static_cast<uint32_t>(session->deepseek_sparse_topk_scores_bytes /
                              sizeof(float)),
        session->device_deepseek_runtime_counters);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }
  const size_t mla_attention_serial_shared_bytes =
      (static_cast<size_t>(kv_lora_rank) * 2u + qk_rope) * sizeof(float) +
      (has_v32_sparse_attention && !has_precomputed_v32_sparse_attention
           ? static_cast<size_t>(kDeepSeekSparseTopKSlotCapacity) *
                 (sizeof(int32_t) + sizeof(float))
           : 0u);
  const size_t mla_attention_chunk_shared_bytes =
      (static_cast<size_t>(kv_lora_rank) + qk_rope) * sizeof(float);
  const bool use_chunked_mla_attention =
      attention_chunks != 0 &&
      session->device_decode_attention_values != nullptr &&
      session->device_decode_attention_m != nullptr &&
      session->device_decode_attention_l != nullptr &&
      (!has_v32_sparse_attention || has_precomputed_v32_sparse_attention);
  if (use_chunked_mla_attention) {
    hf_deepseek_v3_mla_query_latent_kernel<<<
        session->heads, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, layout, session->dtype, session->heads, scratch.q,
        scratch.attn);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    const dim3 chunk_grid(session->heads, attention_chunks);
    hf_deepseek_v3_mla_attention_chunk_kernel<<<
        chunk_grid, kDecodeThreads, mla_attention_chunk_shared_bytes,
        session->stream>>>(
        session->device_arena, layout, layer_index, session->dtype,
        session->heads, session->device_step, max_steps, session->rope_theta,
        scratch.q, scratch.attn, session->device_kv_keys, session->kv_block_count,
        session->device_kv_block_table, attention_chunks,
        session->device_decode_attention_values,
        session->device_decode_attention_m, session->device_decode_attention_l,
        has_precomputed_v32_sparse_attention
            ? session->device_deepseek_sparse_topk_slots
            : nullptr,
        has_precomputed_v32_sparse_attention
            ? session->device_deepseek_sparse_topk_count
            : nullptr,
        session->device_deepseek_runtime_counters);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    const size_t reduce_shared_bytes =
        (static_cast<size_t>(attention_chunks) + kv_lora_rank) * sizeof(float);
    hf_deepseek_v3_mla_attention_reduce_kernel<<<
        session->heads, kDecodeThreads, reduce_shared_bytes,
        session->stream>>>(
        session->device_arena, layout, session->dtype, session->heads,
        session->device_step, max_steps, attention_chunks,
        session->device_decode_attention_values,
        session->device_decode_attention_m, session->device_decode_attention_l,
        session->device_projection_input,
        session->device_deepseek_runtime_counters,
        has_precomputed_v32_sparse_attention ? 1u : 0u);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  } else {
    hf_deepseek_v3_mla_attention_encode_kernel<<<
        session->heads, kDecodeThreads, mla_attention_serial_shared_bytes,
        session->stream>>>(
        session->device_arena, layout, layer_index, session->dtype,
        session->heads, session->device_step, max_steps, session->rope_theta,
        scratch.q, session->device_kv_keys, session->kv_block_count,
        session->device_kv_block_table, session->device_projection_input,
        reinterpret_cast<const uint8_t *>(session->device_deepseek_indexer_state),
        deepseek_v32_indexer_query_state_layer_offset_bytes(session,
                                                            layer_index),
        session->device_deepseek_indexer_kv,
        deepseek_v32_indexer_kv_layer_offset_bytes(session, layer_index),
        deepseek_v32_indexer_kv_block_count(session, layout),
        has_precomputed_v32_sparse_attention
            ? session->device_deepseek_sparse_topk_slots
            : nullptr,
        has_precomputed_v32_sparse_attention
            ? session->device_deepseek_sparse_topk_count
            : nullptr,
        session->device_deepseek_runtime_counters);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->attention_ns);
  if (err != cudaSuccess) return err;

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  if (bf16_storage) {
    err = project_encoded_rows(
        session, nullptr, session->device_arena + layout.w_o,
        session->device_projection_input, session->hidden, value_rows, 1,
        kDTypeBF16, 0.0f, scratch.residual);
  } else {
    err = launch_deepseek_fp8_f32_scale_encoded_matvec(
        session->stream, deepseek_fp8_ptr(session->device_arena, layout.w_o),
        deepseek_scale_ptr(session->device_arena, layout.deepseek_o_a_scale),
        session->device_projection_input, session->dtype, session->hidden,
        value_rows, block_rows, block_cols, scratch.residual);
  }
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(
      session, profile,
      profile == nullptr ? nullptr : profile->attention_output_projection_ns);
  if (err != cudaSuccess) return err;

  err = deepseek_profile_begin_if(session, profile);
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
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->norm_ns);
  if (err != cudaSuccess) return err;

  if (layout.mlp_kind == kMlpKindSparseMoe) {
    err = deepseek_profile_begin_if(session, profile);
    if (err != cudaSuccess) return err;
    const uint32_t router_blocks =
        layout.num_experts == 0 ? 1u : layout.num_experts;
    hf_deepseek_v3_sparse_moe_router_logits_kernel<<<
        router_blocks, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, layout, session->dtype, session->hidden,
        attention_rows, kv_cache_width, session->intermediate,
        session->device_step, max_steps, session->device_scratch,
        session->device_projection_input);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    hf_deepseek_v3_sparse_moe_route_kernel<<<1, kDecodeThreads, 0,
                                             session->stream>>>(
        session->device_arena, layout, session->dtype, session->hidden,
        attention_rows, kv_cache_width, session->intermediate,
        session->device_step, max_steps, session->device_scratch,
        session->device_projection_input,
        session->device_deepseek_runtime_counters);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                      ? nullptr
                                                      : profile->mlp_ns);
    if (err != cudaSuccess) return err;
    const uint32_t routed_top_k = layout.experts_per_token;
    const bool parallel_sparse_experts =
        routed_top_k > 1u &&
        deepseek_v4_aux_ready(session, kDeepSeekV4AttentionAuxStreamCount,
                              kDeepSeekV4AttentionEventCount);
    if (parallel_sparse_experts) {
      err = deepseek_profile_begin_if(session, profile);
      if (err != cudaSuccess) return err;
      err = deepseek_v4_aux_fanout(session, kDeepSeekV4AttentionAuxStreamCount);
      if (err != cudaSuccess) return err;
      for (uint32_t rank = 0; rank < routed_top_k; ++rank) {
        cudaStream_t expert_stream =
            session->deepseek_v4_attention_aux_streams
                [rank % kDeepSeekV4AttentionAuxStreamCount];
        hf_deepseek_v3_sparse_moe_expert_gate_up_kernel<<<
            layout.moe_intermediate, kDecodeThreads, 0, expert_stream>>>(
            session->device_arena, layout, session->dtype, session->hidden,
            attention_rows, kv_cache_width, session->intermediate, rank,
            session->device_step, max_steps, session->device_scratch);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        hf_deepseek_v3_sparse_moe_expert_down_kernel<<<
            session->hidden, kDecodeThreads, 0, expert_stream>>>(
            session->device_arena, layout, session->dtype, session->hidden,
            attention_rows, kv_cache_width, session->intermediate, rank,
            session->device_step, max_steps, session->device_scratch);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
      }
      err = deepseek_v4_aux_join(session, kDeepSeekV4AttentionAuxStreamCount);
      if (err != cudaSuccess) return err;
      const uint32_t reduce_blocks =
          (session->hidden + kDecodeThreads - 1u) / kDecodeThreads;
      hf_deepseek_sparse_moe_reduce_down_kernel<<<
          reduce_blocks, kDecodeThreads, 0, session->stream>>>(
          layout, session->hidden, attention_rows, kv_cache_width,
          session->intermediate, session->device_step, max_steps,
          session->device_scratch);
      err = cudaGetLastError();
      if (err != cudaSuccess) return err;
      err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                        ? nullptr
                                                        : profile->gate_up_projection_ns);
    } else {
      for (uint32_t rank = 0;
           err == cudaSuccess && rank < routed_top_k; ++rank) {
        err = deepseek_profile_begin_if(session, profile);
        if (err != cudaSuccess) return err;
        hf_deepseek_v3_sparse_moe_expert_gate_up_kernel<<<
            layout.moe_intermediate, kDecodeThreads, 0, session->stream>>>(
            session->device_arena, layout, session->dtype, session->hidden,
            attention_rows, kv_cache_width, session->intermediate, rank,
            session->device_step, max_steps, session->device_scratch);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                          ? nullptr
                                                          : profile->gate_up_projection_ns);
        if (err != cudaSuccess) return err;
        err = deepseek_profile_begin_if(session, profile);
        if (err != cudaSuccess) return err;
        hf_deepseek_v3_sparse_moe_expert_down_kernel<<<
            session->hidden, kDecodeThreads, 0, session->stream>>>(
            session->device_arena, layout, session->dtype, session->hidden,
            attention_rows, kv_cache_width, session->intermediate, rank,
            session->device_step, max_steps, session->device_scratch);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                          ? nullptr
                                                          : profile->down_projection_ns);
      }
      if (err == cudaSuccess && routed_top_k > 1u) {
        const uint32_t reduce_blocks =
            (session->hidden + kDecodeThreads - 1u) / kDecodeThreads;
        hf_deepseek_sparse_moe_reduce_down_kernel<<<
            reduce_blocks, kDecodeThreads, 0, session->stream>>>(
            layout, session->hidden, attention_rows, kv_cache_width,
            session->intermediate, session->device_step, max_steps,
            session->device_scratch);
        err = cudaGetLastError();
      }
    }
    if (err != cudaSuccess) return err;
    if (err == cudaSuccess && layout.shared_expert_intermediate != 0) {
      err = deepseek_profile_begin_if(session, profile);
      if (err != cudaSuccess) return err;
      hf_deepseek_v3_sparse_moe_shared_gate_up_kernel<<<
          layout.shared_expert_intermediate, kDecodeThreads, 0,
          session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          attention_rows, kv_cache_width, session->intermediate,
          session->device_step, max_steps, session->device_scratch);
      err = cudaGetLastError();
      if (err != cudaSuccess) return err;
      err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                        ? nullptr
                                                        : profile->gate_up_projection_ns);
    }
    if (err == cudaSuccess && layout.shared_expert_intermediate != 0) {
      err = deepseek_profile_begin_if(session, profile);
      if (err != cudaSuccess) return err;
      hf_deepseek_v3_sparse_moe_shared_down_kernel<<<
          session->hidden, kDecodeThreads, 0, session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          attention_rows, kv_cache_width, session->intermediate,
          session->device_step, max_steps, session->device_scratch);
      err = cudaGetLastError();
      if (err != cudaSuccess) return err;
      err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                        ? nullptr
                                                        : profile->down_projection_ns);
    }
    return err;
  }
  if (layout.w_gate == kMissingOffset || layout.w_up == kMissingOffset ||
      layout.w_down == kMissingOffset) {
    return cudaErrorInvalidValue;
  }

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  if (bf16_storage) {
    err = project_encoded_rows(
        session, nullptr, session->device_arena + layout.w_gate,
        session->device_projection_input, session->intermediate,
        session->hidden, 1, kDTypeBF16, 0.0f, scratch.gate);
    if (err == cudaSuccess) {
      err = project_encoded_rows(
          session, nullptr, session->device_arena + layout.w_up,
          session->device_projection_input, session->intermediate,
          session->hidden, 1, kDTypeBF16, 0.0f, scratch.up);
    }
  } else {
    const uint64_t gate_scale =
        deepseek_f32_scale_offset(layout.w_gate, session->intermediate,
                                  session->hidden);
    const uint64_t up_scale =
        deepseek_f32_scale_offset(layout.w_up, session->intermediate,
                                  session->hidden);
    err = launch_deepseek_fp8_f32_scale_dual_encoded_matvec_from_arena(
        session->stream, session->device_arena, layout.w_gate, gate_scale,
        layout.w_up, up_scale, session->device_projection_input,
        session->dtype, session->intermediate, session->hidden, block_rows,
        block_cols, scratch.gate, scratch.up);
  }
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->gate_up_projection_ns);
  if (err != cudaSuccess) return err;
  {
    err = deepseek_profile_begin_if(session, profile);
    if (err != cudaSuccess) return err;
    const uint32_t ff_blocks =
        (session->intermediate + kDecodeThreads - 1) / kDecodeThreads;
    hf_layer_ff_encode_kernel<<<ff_blocks, kDecodeThreads, 0,
                                session->stream>>>(
        session->dtype, session->hidden, attention_rows, kv_cache_width,
        session->intermediate, session->device_step, max_steps,
        session->device_scratch, session->device_projection_input);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                      ? nullptr
                                                      : profile->mlp_ns);
    if (err != cudaSuccess) return err;
  }
  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  if (bf16_storage) {
    err = project_encoded_rows(
        session, nullptr, session->device_arena + layout.w_down,
        session->device_projection_input, session->hidden,
        session->intermediate, 1, kDTypeBF16, 0.0f, scratch.down);
  } else {
    const uint64_t down_scale =
        deepseek_f32_scale_offset(layout.w_down, session->hidden,
                                  session->intermediate);
    err = launch_deepseek_fp8_f32_scale_encoded_matvec_from_arena(
        session->stream, session->device_arena, layout.w_down, down_scale,
        session->device_projection_input, session->dtype, session->hidden,
        session->intermediate, block_rows, block_cols, scratch.down);
  }
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->down_projection_ns);
  if (err != cudaSuccess) return err;

  return cudaSuccess;
}
