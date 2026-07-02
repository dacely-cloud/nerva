cudaError_t launch_deepseek_v4_swa_dense_projection_step(
    NervaCudaHfDecodeSequenceSession *session, const SequenceLayerLayout &layout,
    uint32_t layer_index, uint32_t max_steps, uint32_t prompt_token_count,
    const DeepseekDecodeProfileBuckets *profile = nullptr) {
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
      layout.k_norm == kMissingOffset ||
      layout.w_o == kMissingOffset ||
      layout.deepseek_o_b == kMissingOffset ||
      layout.deepseek_o_b_scale == kMissingOffset) {
    return cudaErrorInvalidValue;
  }
  const uint32_t q_lora_rank = layout.deepseek_q_lora_rank;
  const uint32_t qk_nope = layout.deepseek_qk_nope_head_dim;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  const uint32_t o_lora_rank = layout.deepseek_o_lora_rank;
  const uint32_t o_groups = layout.deepseek_o_groups;
  if (q_lora_rank == 0 || qk_rope == 0 ||
      qk_nope + qk_rope != session->head_dim || session->heads == 0) {
    return cudaErrorInvalidValue;
  }
  const uint32_t attention_hidden = session->heads * session->head_dim;
  if (q_lora_rank > attention_hidden || o_lora_rank == 0 ||
      o_groups == 0 || session->heads % o_groups != 0 ||
      o_lora_rank > attention_hidden / o_groups) {
    return cudaErrorInvalidValue;
  }
  const uint32_t heads_per_group = session->heads / o_groups;
  const uint32_t wo_a_cols = heads_per_group * session->head_dim;
  const uint32_t wo_a_rows = o_groups * o_lora_rank;
  LayerScratch scratch =
      layer_scratch_ptrs(session->device_scratch, session->hidden,
                         attention_hidden, session->head_dim,
                         session->intermediate);
  const bool dense_mlp = layout.mlp_kind == kMlpKindDense;
  const bool sparse_moe_mlp = layout.mlp_kind == kMlpKindSparseMoe;
  if (!dense_mlp && !sparse_moe_mlp) {
    return cudaErrorInvalidValue;
  }
  if (dense_mlp &&
      (layout.w_gate == kMissingOffset || layout.w_up == kMissingOffset ||
       layout.w_down == kMissingOffset || session->intermediate == 0)) {
    return cudaErrorInvalidValue;
  }
  if (sparse_moe_mlp &&
      (layout.w_router == kMissingOffset ||
       layout.w_expert_gate_up == kMissingOffset ||
       layout.w_expert_down == kMissingOffset || layout.num_experts == 0 ||
       layout.num_experts > kSparseMoeExpertsMax ||
       layout.experts_per_token == 0 ||
       layout.experts_per_token > kSparseMoeTopKMax ||
       layout.experts_per_token > layout.num_experts ||
       layout.moe_intermediate == 0 ||
       layout.moe_intermediate > session->intermediate ||
       (session->hidden & 1u) != 0 || (layout.moe_intermediate & 1u) != 0)) {
    return cudaErrorInvalidValue;
  }
  const uint32_t shared_intermediate = layout.shared_expert_intermediate;
  const bool external_shared_expert =
      sparse_moe_mlp && shared_intermediate != 0;
  if (external_shared_expert &&
      (layout.w_shared_expert_gate == kMissingOffset ||
       layout.w_shared_expert_up == kMissingOffset ||
       layout.w_shared_expert_down == kMissingOffset ||
       shared_intermediate > session->intermediate)) {
    return cudaErrorInvalidValue;
  }
  constexpr uint32_t block_rows = 128;
  constexpr uint32_t block_cols = 128;
  const bool parallel_swa_attention =
      layout.deepseek_mode == kDeepSeekModeV4Swa;
  const bool parallel_compressed_attention =
      layout.deepseek_mode == kDeepSeekModeV4Compressed ||
      layout.deepseek_mode == kDeepSeekModeV4CompressedIndexer;
  const bool parallel_v4_attention =
      parallel_swa_attention || parallel_compressed_attention;
  const float layer_rope_theta =
      deepseek_v4_layer_rope_theta(session->rope_theta, layout);

  cudaError_t err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  hf_deepseek_v4_attn_mhc_pre_kernel<<<1, 1, 0, session->stream>>>(
      session->device_arena, layout, session->dtype, session->hidden,
      session->heads, session->head_dim, session->intermediate, layer_index,
      session->device_step, max_steps, session->rms_eps,
      session->device_scratch, session->device_projection_input,
      session->device_deepseek_mhc_residual,
      session->device_deepseek_mhc_post_mix,
      session->device_deepseek_mhc_comb_mix);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->norm_ns);
  if (err != cudaSuccess) return err;

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  if (deepseek_v4_aux_ready(session, 2u, 3u)) {
    err = deepseek_v4_aux_fanout(session, 2u);
    if (err != cudaSuccess) return err;
    err = launch_deepseek_fp8_e8m0_scale_encoded_matvec(
        session->deepseek_v4_attention_aux_streams[0],
        deepseek_fp8_ptr(session->device_arena, layout.w_q),
        deepseek_fp8_ptr(session->device_arena, layout.deepseek_q_a_scale),
        session->device_projection_input, session->dtype, q_lora_rank,
        session->hidden, block_rows, block_cols, scratch.q);
    if (err != cudaSuccess) return err;
    err = launch_deepseek_fp8_e8m0_scale_encoded_matvec(
        session->deepseek_v4_attention_aux_streams[1],
        deepseek_fp8_ptr(session->device_arena, layout.w_k),
        deepseek_fp8_ptr(session->device_arena, layout.deepseek_kv_a_scale),
        session->device_projection_input, session->dtype, session->head_dim,
        session->hidden, block_rows, block_cols, scratch.k);
    if (err != cudaSuccess) return err;
    err = deepseek_v4_aux_join(session, 2u);
    if (err != cudaSuccess) return err;
  } else {
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
  }
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->qkv_projection_ns);
  if (err != cudaSuccess) return err;

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  hf_deepseek_v4_q_a_norm_kernel<<<1, kDecodeNormThreads, 0,
                                   session->stream>>>(
      session->device_arena, layout, session->hidden, session->heads,
      session->head_dim, session->intermediate, session->device_step,
      max_steps, session->rms_eps, session->device_scratch);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->norm_ns);
  if (err != cudaSuccess) return err;

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  err = launch_deepseek_fp8_e8m0_scale_matvec(
      session->stream,
      deepseek_fp8_ptr(session->device_arena, layout.deepseek_q_b),
      deepseek_fp8_ptr(session->device_arena, layout.deepseek_q_b_scale),
      scratch.q_gate, attention_hidden, q_lora_rank, block_rows, block_cols,
      scratch.q);
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->qkv_projection_ns);
  if (err != cudaSuccess) return err;

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  hf_deepseek_v4_finalize_preprojected_qk_kernel<<<session->heads + 1u,
                                                    kDecodeThreads, 0,
                                                    session->stream>>>(
      session->device_arena, layout, session->dtype, session->hidden,
      session->heads, session->head_dim, session->intermediate,
      session->device_step, max_steps, session->rms_eps, layer_rope_theta,
      session->device_scratch);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->attention_ns);
  if (err != cudaSuccess) return err;

  uint32_t precomputed_compressor_state = 0;
  uint32_t precomputed_indexer_state = 0;
  const uint32_t compressor_coff =
      layout.deepseek_compress_ratio == 4 ? 2u : 1u;
  const bool can_precompute_compressor =
      (layout.deepseek_mode == kDeepSeekModeV4Compressed ||
       layout.deepseek_mode == kDeepSeekModeV4CompressedIndexer) &&
      layout.deepseek_compress_ratio > 1 &&
      compressor_coff * session->head_dim != 0 &&
      session->device_deepseek_compressor_state != nullptr &&
      session->device_deepseek_compressed_kv != nullptr &&
      layout.deepseek_compressor_wkv != kMissingOffset &&
      layout.deepseek_compressor_wgate != kMissingOffset &&
      layout.deepseek_compressor_ape != kMissingOffset &&
      layout.deepseek_compressor_norm != kMissingOffset;
  const bool can_precompute_indexer =
      layout.deepseek_mode == kDeepSeekModeV4CompressedIndexer &&
      layout.deepseek_compress_ratio > 1 &&
      compressor_coff * layout.deepseek_index_head_dim != 0 &&
      session->device_deepseek_indexer_state != nullptr &&
      session->device_deepseek_indexer_kv != nullptr &&
      layout.deepseek_indexer_compressor_wkv != kMissingOffset &&
      layout.deepseek_indexer_compressor_wgate != kMissingOffset &&
      layout.deepseek_indexer_compressor_ape != kMissingOffset &&
      layout.deepseek_indexer_compressor_norm != kMissingOffset;
  if (can_precompute_compressor && can_precompute_indexer &&
      deepseek_v4_aux_ready(session, 2u, 3u)) {
    err = deepseek_profile_begin_if(session, profile);
    if (err != cudaSuccess) return err;
    err = deepseek_v4_aux_fanout(session, 2u);
    if (err != cudaSuccess) return err;
    err = launch_deepseek_v4_compressor_state_and_kv(
        session, layout, layer_index, max_steps, layer_rope_theta,
        session->deepseek_v4_attention_aux_streams[0]);
    if (err != cudaSuccess) return err;
    err = launch_deepseek_v4_indexer_state_and_kv(
        session, layout, layer_index, max_steps, layer_rope_theta,
        session->deepseek_v4_attention_aux_streams[1]);
    if (err != cudaSuccess) return err;
    err = deepseek_v4_aux_join(session, 2u);
    if (err != cudaSuccess) return err;
    err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                      ? nullptr
                                                      : profile->attention_ns);
    if (err != cudaSuccess) return err;
    precomputed_compressor_state = 1;
    precomputed_indexer_state = 1;
  } else {
    if (can_precompute_compressor) {
      err = deepseek_profile_begin_if(session, profile);
      if (err != cudaSuccess) return err;
      err = launch_deepseek_v4_compressor_state_and_kv(
          session, layout, layer_index, max_steps, layer_rope_theta,
          session->stream);
      if (err != cudaSuccess) return err;
      err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                        ? nullptr
                                                        : profile->attention_ns);
      if (err != cudaSuccess) return err;
      precomputed_compressor_state = 1;
    }
    if (can_precompute_indexer) {
      err = deepseek_profile_begin_if(session, profile);
      if (err != cudaSuccess) return err;
      err = launch_deepseek_v4_indexer_state_and_kv(
          session, layout, layer_index, max_steps, layer_rope_theta,
          session->stream);
      if (err != cudaSuccess) return err;
      err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                        ? nullptr
                                                        : profile->attention_ns);
      if (err != cudaSuccess) return err;
      precomputed_indexer_state = 1;
    }
  }

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  hf_deepseek_v4_swa_dense_layer_kernel<<<1, 1, 0, session->stream>>>(
      session->device_arena, layout, layer_index, session->dtype,
      session->hidden, session->heads, session->head_dim,
      session->intermediate, session->device_step, max_steps,
      session->rms_eps, layer_rope_theta, session->device_scratch,
      session->device_kv_keys, session->device_kv_values,
      session->kv_block_count, session->device_kv_block_table,
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
      precomputed_compressor_state, precomputed_indexer_state,
      parallel_v4_attention ? 1u : 0u);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                     ? nullptr
                                                     : profile->attention_ns);
  if (err != cudaSuccess) return err;

  if (parallel_swa_attention) {
    err = deepseek_profile_begin_if(session, profile);
    if (err != cudaSuccess) return err;
    hf_deepseek_v4_swa_attention_kernel<<<session->heads, kDecodeThreads, 0,
                                          session->stream>>>(
        session->device_arena, layout, layer_index, session->dtype,
        session->hidden, session->heads, session->head_dim,
        session->intermediate, session->device_step, max_steps,
        layer_rope_theta, session->device_scratch, session->device_kv_keys,
        session->device_kv_values, session->kv_block_count,
        session->device_kv_block_table, session->device_deepseek_swa_kv,
        deepseek_v4_swa_kv_layer_offset_bytes(session, layer_index),
        deepseek_v4_swa_kv_block_count(session, layout),
        session->device_deepseek_runtime_counters,
        session->experimental_rt_local_window_tokens);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                       ? nullptr
                                                       : profile->attention_ns);
    if (err != cudaSuccess) return err;
  }
  if (parallel_compressed_attention) {
    if (layout.deepseek_mode == kDeepSeekModeV4CompressedIndexer) {
      err = deepseek_profile_begin_if(session, profile);
      if (err != cudaSuccess) return err;
      hf_deepseek_v4_compressed_indexer_sparse_topk_select_kernel<<<
          1, kDecodeThreads, 0, session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          session->heads, session->head_dim, session->intermediate,
          session->device_step, max_steps, layer_rope_theta,
          session->device_scratch, session->device_projection_input,
          session->device_deepseek_compressed_kv,
          deepseek_v4_main_compressed_kv_layer_offset_bytes(session,
                                                            layer_index),
          deepseek_v4_compressed_kv_block_count(session, layout),
          session->device_deepseek_indexer_kv,
          deepseek_v4_indexer_kv_layer_offset_bytes(session, layer_index),
          deepseek_v4_compressed_kv_block_count(session, layout),
          session->kv_block_count, session->device_kv_block_table,
          session->device_deepseek_sparse_topk_slots,
          session->device_deepseek_sparse_topk_count,
          session->device_deepseek_sparse_topk_scores,
          static_cast<uint32_t>(session->deepseek_sparse_topk_scores_bytes /
                                sizeof(float)),
          session->device_deepseek_runtime_counters);
      err = cudaGetLastError();
      if (err != cudaSuccess) return err;
      err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                         ? nullptr
                                                         : profile->attention_ns);
      if (err != cudaSuccess) return err;
    }
    err = deepseek_profile_begin_if(session, profile);
    if (err != cudaSuccess) return err;
    hf_deepseek_v4_compressed_attention_kernel<<<session->heads,
                                                 kDecodeThreads,
                                                 0,
                                                 session->stream>>>(
        session->device_arena, layout, layer_index, session->dtype,
        session->hidden, session->heads, session->head_dim,
        session->intermediate, session->device_step, max_steps,
        layer_rope_theta, session->device_scratch, session->device_kv_keys,
        session->device_kv_values, session->kv_block_count,
        session->device_kv_block_table, session->device_deepseek_swa_kv,
        deepseek_v4_swa_kv_layer_offset_bytes(session, layer_index),
        deepseek_v4_swa_kv_block_count(session, layout),
        session->device_deepseek_compressed_kv,
        deepseek_v4_main_compressed_kv_layer_offset_bytes(session, layer_index),
        deepseek_v4_compressed_kv_block_count(session, layout),
        layout.deepseek_mode == kDeepSeekModeV4CompressedIndexer
            ? session->device_deepseek_sparse_topk_slots
            : nullptr,
        layout.deepseek_mode == kDeepSeekModeV4CompressedIndexer
            ? session->device_deepseek_sparse_topk_count
            : nullptr,
        session->device_deepseek_runtime_counters,
        session->experimental_rt_local_window_tokens);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                       ? nullptr
                                                       : profile->attention_ns);
    if (err != cudaSuccess) return err;
  }

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  if (layout.deepseek_o_a_scale == kMissingOffset) {
    err = launch_deepseek_bf16_grouped_matvec(
        session->stream, session->device_arena + layout.w_o, scratch.attn,
        o_groups, o_lora_rank, wo_a_cols, scratch.q_gate);
  } else {
    err = launch_deepseek_fp8_e8m0_scale_grouped_matvec(
        session->stream, deepseek_fp8_ptr(session->device_arena, layout.w_o),
        deepseek_fp8_ptr(session->device_arena, layout.deepseek_o_a_scale),
        scratch.attn, o_groups, o_lora_rank, wo_a_cols, block_rows, block_cols,
        scratch.q_gate);
  }
  if (err != cudaSuccess) return err;
  err = launch_deepseek_fp8_e8m0_scale_matvec(
      session->stream,
      deepseek_fp8_ptr(session->device_arena, layout.deepseek_o_b),
      deepseek_fp8_ptr(session->device_arena, layout.deepseek_o_b_scale),
      scratch.q_gate, session->hidden, wo_a_rows, block_rows, block_cols,
      scratch.residual);
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->attention_output_projection_ns);
  if (err != cudaSuccess) return err;

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  hf_deepseek_v4_ffn_mhc_pre_kernel<<<1, kDecodeNormThreads, 0,
                                      session->stream>>>(
      session->device_arena, layout, session->dtype, session->hidden,
      session->heads, session->head_dim, session->intermediate,
      session->device_step, max_steps, session->rms_eps,
      session->device_scratch, session->device_projection_input,
      session->device_deepseek_mhc_residual,
      session->device_deepseek_mhc_post_mix,
      session->device_deepseek_mhc_comb_mix);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->norm_ns);
  if (err != cudaSuccess) return err;

  if (sparse_moe_mlp) {
    err = deepseek_profile_begin_if(session, profile);
    if (err != cudaSuccess) return err;
    const uint32_t router_blocks =
        layout.num_experts == 0 ? 1u : layout.num_experts;
    hf_deepseek_v4_sparse_moe_router_logits_kernel<<<
        router_blocks, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, layout, session->hidden, attention_hidden,
        session->head_dim, session->intermediate, session->device_step,
        max_steps, session->device_scratch);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    hf_deepseek_v4_sparse_moe_route_kernel<<<1, kDecodeThreads, 0,
                                             session->stream>>>(
        session->device_arena, layout, session->hidden, attention_hidden,
        session->head_dim, session->intermediate, session->device_step,
        max_steps, session->vocab_size, session->device_prompt_tokens,
        prompt_token_count, session->device_slots, session->device_scratch,
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
        hf_deepseek_v4_sparse_moe_expert_gate_up_kernel<<<
            layout.moe_intermediate, kDecodeThreads, 0, expert_stream>>>(
            session->device_arena, layout, session->hidden, attention_hidden,
            session->head_dim, session->intermediate, rank,
            session->device_step, max_steps, session->device_scratch);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        hf_deepseek_v4_sparse_moe_expert_down_kernel<<<
            session->hidden, kDecodeThreads, 0, expert_stream>>>(
            session->device_arena, layout, session->hidden, attention_hidden,
            session->head_dim, session->intermediate, rank,
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
          layout, session->hidden, attention_hidden, session->head_dim,
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
        hf_deepseek_v4_sparse_moe_expert_gate_up_kernel<<<
            layout.moe_intermediate, kDecodeThreads, 0, session->stream>>>(
            session->device_arena, layout, session->hidden, attention_hidden,
            session->head_dim, session->intermediate, rank,
            session->device_step, max_steps, session->device_scratch);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                          ? nullptr
                                                          : profile->gate_up_projection_ns);
        if (err != cudaSuccess) return err;

        err = deepseek_profile_begin_if(session, profile);
        if (err != cudaSuccess) return err;
        hf_deepseek_v4_sparse_moe_expert_down_kernel<<<
            session->hidden, kDecodeThreads, 0, session->stream>>>(
            session->device_arena, layout, session->hidden, attention_hidden,
            session->head_dim, session->intermediate, rank,
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
            layout, session->hidden, attention_hidden, session->head_dim,
            session->intermediate, session->device_step, max_steps,
            session->device_scratch);
        err = cudaGetLastError();
      }
    }
    if (err != cudaSuccess) return err;
    if (!external_shared_expert) return err;
  }

  if (external_shared_expert) {
    const uint64_t shared_gate_scale =
        layout.w_shared_expert_gate +
        deepseek_fp8_slots_u64(shared_intermediate, session->hidden);
    const uint64_t shared_up_scale =
        layout.w_shared_expert_up +
        deepseek_fp8_slots_u64(shared_intermediate, session->hidden);
    const uint64_t shared_down_scale =
        layout.w_shared_expert_down +
        deepseek_fp8_slots_u64(session->hidden, shared_intermediate);

    err = deepseek_profile_begin_if(session, profile);
    if (err != cudaSuccess) return err;
    err = launch_deepseek_fp8_e8m0_scale_encoded_matvec(
        session->stream,
        deepseek_fp8_ptr(session->device_arena, layout.w_shared_expert_gate),
        deepseek_fp8_ptr(session->device_arena, shared_gate_scale),
        session->device_projection_input, session->dtype, shared_intermediate,
        session->hidden, block_rows, block_cols, scratch.gate);
    if (err != cudaSuccess) return err;
    err = launch_deepseek_fp8_e8m0_scale_encoded_matvec(
        session->stream,
        deepseek_fp8_ptr(session->device_arena, layout.w_shared_expert_up),
        deepseek_fp8_ptr(session->device_arena, shared_up_scale),
        session->device_projection_input, session->dtype, shared_intermediate,
        session->hidden, block_rows, block_cols, scratch.up);
    if (err != cudaSuccess) return err;
    err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                      ? nullptr
                                                      : profile->gate_up_projection_ns);
    if (err != cudaSuccess) return err;

    err = deepseek_profile_begin_if(session, profile);
    if (err != cudaSuccess) return err;
    const uint32_t shared_ff_blocks =
        (shared_intermediate + kDecodeThreads - 1) / kDecodeThreads;
    hf_deepseek_ff_encode_kernel<<<shared_ff_blocks, kDecodeThreads, 0,
                                   session->stream>>>(
        layout, session->dtype, session->hidden, attention_hidden,
        session->head_dim, session->intermediate, shared_intermediate,
        session->device_step, max_steps, session->device_scratch,
        session->device_projection_input);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                      ? nullptr
                                                      : profile->mlp_ns);
    if (err != cudaSuccess) return err;

    err = deepseek_profile_begin_if(session, profile);
    if (err != cudaSuccess) return err;
    err = launch_deepseek_fp8_e8m0_scale_encoded_matvec(
        session->stream,
        deepseek_fp8_ptr(session->device_arena, layout.w_shared_expert_down),
        deepseek_fp8_ptr(session->device_arena, shared_down_scale),
        session->device_projection_input, session->dtype, session->hidden,
        shared_intermediate, block_rows, block_cols, scratch.residual);
    if (err != cudaSuccess) return err;
    hf_deepseek_accumulate_residual_down_kernel<<<1, kDecodeNormThreads, 0,
                                                  session->stream>>>(
        session->hidden, attention_hidden, session->head_dim,
        session->intermediate, session->device_step, max_steps,
        session->device_scratch);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                      ? nullptr
                                                      : profile->down_projection_ns);
    return err;
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

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  err = launch_deepseek_fp8_f32_scale_dual_encoded_matvec_from_arena(
      session->stream, session->device_arena, layout.w_gate, gate_scale,
      layout.w_up, up_scale, session->device_projection_input, session->dtype,
      session->intermediate, session->hidden, block_rows, block_cols,
      scratch.gate, scratch.up);
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->gate_up_projection_ns);
  if (err != cudaSuccess) return err;

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  const uint32_t ff_blocks =
      (session->intermediate + kDecodeThreads - 1) / kDecodeThreads;
  hf_deepseek_ff_encode_kernel<<<ff_blocks, kDecodeThreads, 0,
                                 session->stream>>>(
      layout, session->dtype, session->hidden, attention_hidden,
      session->head_dim, session->intermediate, session->intermediate,
      session->device_step, max_steps, session->device_scratch,
      session->device_projection_input);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  err = deepseek_profile_end_if(session, profile, profile == nullptr
                                                    ? nullptr
                                                    : profile->mlp_ns);
  if (err != cudaSuccess) return err;

  err = deepseek_profile_begin_if(session, profile);
  if (err != cudaSuccess) return err;
  err = launch_deepseek_fp8_f32_scale_encoded_matvec_from_arena(
      session->stream, session->device_arena, layout.w_down, down_scale,
      session->device_projection_input, session->dtype, session->hidden,
      session->intermediate, block_rows, block_cols, scratch.down);
  if (err != cudaSuccess) return err;
  return deepseek_profile_end_if(session, profile, profile == nullptr
                                                     ? nullptr
                                                     : profile->down_projection_ns);
}
