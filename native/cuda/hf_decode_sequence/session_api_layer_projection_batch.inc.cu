extern "C" int nerva_cuda_hf_decode_sequence_layer_projection_batch_execute(
    const NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteRequest *request,
    NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->reason = kProjectionBatchPlanInvalidRequest;
  if (request == nullptr) {
    return -1;
  }

  out->requested_session_count = request->session_count;
  out->target_block_tokens =
      request->target_block_tokens == 0 ? 1u : request->target_block_tokens;
  out->min_block_tokens =
      request->min_block_tokens == 0 ? 1u : request->min_block_tokens;
  out->layer_index = request->layer_index;
  if (request->session_count == 0) {
    out->reason = kProjectionBatchPlanNoSessions;
    out->status = 0;
    return 0;
  }
  if (request->sessions == nullptr) {
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  auto ready_session = [](const NervaCudaHfDecodeSequenceSession *session) {
    return session != nullptr && session->active_started &&
           !session->active_finished && session->active_prompt_token_count != 0 &&
           session->active_cursor < session->max_context_tokens &&
           projection_batch_session_ready(session);
  };
  auto same_projection_model =
      [](const NervaCudaHfDecodeSequenceSession *lhs,
         const NervaCudaHfDecodeSequenceSession *rhs) {
        return lhs->planned_weight_descriptor_hash != 0 &&
               rhs->planned_weight_descriptor_hash != 0 &&
               lhs->planned_weight_descriptor_hash ==
                   rhs->planned_weight_descriptor_hash &&
               lhs->dtype == rhs->dtype && lhs->hidden == rhs->hidden &&
               lhs->heads == rhs->heads && lhs->kv_heads == rhs->kv_heads &&
               lhs->head_dim == rhs->head_dim &&
               lhs->intermediate == rhs->intermediate &&
               lhs->vocab_size == rhs->vocab_size &&
               lhs->layer_count == rhs->layer_count &&
               lhs->resident_weight_bytes == rhs->resident_weight_bytes;
      };

  uint32_t ready_count = 0;
  bool any_hash = false;
  for (uint32_t index = 0; index < request->session_count; ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (!ready_session(session)) {
      continue;
    }
    ready_count += 1;
    any_hash = any_hash || session->planned_weight_descriptor_hash != 0;
  }
  if (ready_count == 0) {
    out->reason = kProjectionBatchPlanNoReadySessions;
    out->status = 0;
    return 0;
  }
  if (!any_hash) {
    out->reason = kProjectionBatchPlanSharedWeightsUnproven;
    out->status = 0;
    return 0;
  }

  NervaCudaHfDecodeSequenceSession *best = nullptr;
  uint32_t best_count = 0;
  for (uint32_t candidate_index = 0; candidate_index < request->session_count;
       ++candidate_index) {
    NervaCudaHfDecodeSequenceSession *candidate =
        request->sessions[candidate_index];
    if (!ready_session(candidate) ||
        candidate->planned_weight_descriptor_hash == 0 ||
        request->layer_index >= candidate->layer_count) {
      continue;
    }
    uint32_t compatible = 0;
    for (uint32_t other_index = 0; other_index < request->session_count;
         ++other_index) {
      NervaCudaHfDecodeSequenceSession *other = request->sessions[other_index];
      if (ready_session(other) && same_projection_model(candidate, other)) {
        compatible += 1;
      }
    }
    if (compatible > best_count) {
      best = candidate;
      best_count = compatible;
    }
  }
  out->eligible_session_count = best_count;
  if (best == nullptr || best_count < out->min_block_tokens) {
    out->reason = kProjectionBatchPlanInsufficientCompatibleReady;
    out->status = 0;
    return 0;
  }
  if (request->layer_index >= best->layer_count) {
    out->reason = kProjectionBatchPlanInvalidLayer;
    out->status = 0;
    return 0;
  }
  err = ensure_session_cublas_resources(best);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  const uint32_t max_block_tokens =
      std::min(out->target_block_tokens, kProjectionBatchWorkspaceTokens);
  std::vector<NervaCudaHfDecodeSequenceSession *> selected;
  selected.reserve(max_block_tokens);
  for (uint32_t index = 0; index < request->session_count &&
                           selected.size() < max_block_tokens;
       ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (ready_session(session) && same_projection_model(best, session)) {
      selected.push_back(session);
    }
  }
  if (selected.size() < out->min_block_tokens) {
    out->reason = kProjectionBatchPlanInsufficientCompatibleReady;
    out->status = 0;
    return 0;
  }
  out->block_tokens = static_cast<uint32_t>(selected.size());
  out->dtype = best->dtype;

  if (best->projection_batch_peer_streams_synchronized == 0) {
    for (NervaCudaHfDecodeSequenceSession *session : selected) {
      if (session == best) {
        continue;
      }
      if (session->projection_batch_own_stream_synchronized == 0) {
        err = cudaStreamSynchronize(session->stream);
        out->sync_calls += 1;
        if (err != cudaSuccess) {
          out->cuda_error = static_cast<int32_t>(err);
          return -1;
        }
        session->projection_batch_own_stream_synchronized = 1;
      }
    }
  }
  ScopedProjectionBatchFlags layer_scope(best, true, false);

  auto run_stage =
      [&](uint32_t projection_kind,
          NervaCudaHfDecodeSequenceProjectionBatchExecuteResult *stage_out)
          -> int {
    NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest stage_request{};
    stage_request.sessions = request->sessions;
    stage_request.session_count = request->session_count;
    stage_request.target_block_tokens = request->target_block_tokens;
    stage_request.min_block_tokens = request->min_block_tokens;
    stage_request.projection_kind = projection_kind;
    stage_request.layer_index = request->layer_index;
    const int rc = nerva_cuda_hf_decode_sequence_projection_batch_execute(
        &stage_request, stage_out);
    out->cuda_error = stage_out->cuda_error;
    out->device_count = stage_out->device_count;
    out->reason = stage_out->reason;
    out->eligible_session_count = stage_out->eligible_session_count;
    out->block_tokens = stage_out->block_tokens;
    out->target_block_tokens = stage_out->target_block_tokens;
    out->min_block_tokens = stage_out->min_block_tokens;
    out->dtype = stage_out->dtype;
    return rc;
  };

  auto launch_attention_encode =
      [&](NervaCudaHfDecodeSequenceSession *session) -> cudaError_t {
    const uint32_t attention_hidden = session->heads * session->head_dim;
    const uint32_t kv_hidden = session->kv_heads * session->head_dim;
    const uint32_t decode_head_threads = decode_head_threads_for_session(session);
    const uint32_t attention_chunks =
        decode_attention_chunks_for_cursor(session, session->active_cursor);
    const SequenceLayerLayout layout =
        session->host_layouts[request->layer_index];
    const uint32_t max_steps = session->max_context_tokens;
    if (attention_chunks == 0) {
      hf_layer_qkv_attention_encode_kernel<<<session->heads,
                                             decode_head_threads, 0,
                                             best->stream>>>(
          session->device_arena, layout, request->layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          session->device_projection_input);
      out->dependency_kernel_launches += 1;
      return cudaGetLastError();
    }

    hf_layer_qkv_prepare_kernel<<<session->heads, decode_head_threads, 0,
                                  best->stream>>>(
        session->device_arena, layout, request->layer_index, session->dtype,
        session->hidden, session->heads, session->kv_heads, session->head_dim,
        session->intermediate, session->device_step, max_steps,
        session->rms_eps, session->rope_theta, session->device_scratch,
        session->device_kv_keys, session->device_kv_values,
        session->kv_block_count, session->device_kv_block_table, nullptr,
        nullptr, nullptr);
    out->dependency_kernel_launches += 1;
    cudaError_t local_err = cudaGetLastError();
    if (local_err != cudaSuccess) {
      return local_err;
    }
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
    if (!use_fused_qk_selector) {
      local_err = launch_experimental_rt_qk_page_selector(
          session, request->layer_index, attention_chunks, max_steps, best->stream);
      if (local_err != cudaSuccess) {
        return local_err;
      }
      if (experimental_rt_qk_selector_active(session, attention_chunks)) {
        out->dependency_kernel_launches += 1;
      }
    }
    const dim3 grid((use_shared_warp_gqa || use_grouped_gqa) ? session->kv_heads
                                                             : session->heads,
                    attention_chunks);
    launch_hf_layer_attention_chunk_kernel(
        best->stream, grid, session->dtype, use_shared_warp_gqa,
        use_grouped_gqa, decode_head_threads, request->layer_index, session->hidden,
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

    out->dependency_kernel_launches += 1;
    local_err = cudaGetLastError();
    if (local_err != cudaSuccess) {
      return local_err;
    }
    const size_t reduce_shared_bytes =
        static_cast<size_t>(attention_chunks) * sizeof(float);
    hf_layer_attention_reduce_kernel<<<session->heads, decode_head_threads,
                                       reduce_shared_bytes, best->stream>>>(
        session->dtype, session->hidden, session->heads, session->kv_heads,
        session->head_dim, session->intermediate, session->device_step,
        max_steps, attention_chunks, session->device_scratch,
        session->device_decode_attention_values, session->device_decode_attention_m,
        session->device_decode_attention_l, session->device_projection_input);
    out->dependency_kernel_launches += 1;
    return cudaGetLastError();
  };

  if (request->layer_index == 0) {
    for (NervaCudaHfDecodeSequenceSession *session : selected) {
      hf_decode_set_step_kernel<<<1, 1, 0, best->stream>>>(
          session->device_step, session->active_cursor);
      out->dependency_kernel_launches += 1;
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        out->cuda_error = static_cast<int32_t>(err);
        return -1;
      }
      const uint32_t attention_hidden = session->heads * session->head_dim;
      const uint32_t kv_hidden = session->kv_heads * session->head_dim;
      const SequenceLayerLayout first_layout = session->host_layouts[0];
      hf_decode_prepare_first_attn_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, best->stream>>>(
          session->device_arena, session->arena_layout, first_layout,
          session->dtype, session->hidden,
          layer_norm_weight_dtype(first_layout, session->dtype),
          attention_hidden, kv_hidden, session->intermediate,
          session->device_step, session->max_context_tokens,
          session->device_prompt_tokens, session->active_prompt_token_count,
          session->device_slots, session->rms_eps, session->device_scratch,
          session->device_projection_input);
      out->dependency_kernel_launches += 1;
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        out->cuda_error = static_cast<int32_t>(err);
        return -1;
      }
    }
  }

  constexpr uint32_t kLayerProjectionKinds[] = {
      kProjectionBatchKindQkv,
      kProjectionBatchKindAttentionOutput,
      kProjectionBatchKindGateUp,
      kProjectionBatchKindDown,
  };
  NervaCudaHfDecodeSequenceProjectionBatchExecuteResult stages[4];

  int rc = run_stage(kLayerProjectionKinds[0], &stages[0]);
  if (rc != 0 || stages[0].status != 0 || stages[0].exact == 0) {
    out->exact = 0;
    out->status = stages[0].status;
    return rc;
  }

  for (NervaCudaHfDecodeSequenceSession *session : selected) {
    err = launch_attention_encode(session);
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  rc = run_stage(kLayerProjectionKinds[1], &stages[1]);
  if (rc != 0 || stages[1].status != 0 || stages[1].exact == 0) {
    out->exact = 0;
    out->status = stages[1].status;
    return rc;
  }

  for (NervaCudaHfDecodeSequenceSession *session : selected) {
    const uint32_t attention_hidden = session->heads * session->head_dim;
    const uint32_t kv_hidden = session->kv_heads * session->head_dim;
    const SequenceLayerLayout layout =
        session->host_layouts[request->layer_index];
    hf_layer_mlp_norm_encode_kernel<<<1, kDecodeNormThreads, 0, best->stream>>>(
        session->device_arena, layout, session->dtype, session->hidden,
        attention_hidden, kv_hidden, session->intermediate, session->device_step,
        session->max_context_tokens, session->rms_eps, session->device_scratch,
        session->device_projection_input);
    out->dependency_kernel_launches += 1;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  rc = run_stage(kLayerProjectionKinds[2], &stages[2]);
  if (rc != 0 || stages[2].status != 0 || stages[2].exact == 0) {
    out->exact = 0;
    out->status = stages[2].status;
    return rc;
  }

  for (NervaCudaHfDecodeSequenceSession *session : selected) {
    const uint32_t attention_hidden = session->heads * session->head_dim;
    const uint32_t kv_hidden = session->kv_heads * session->head_dim;
    const uint32_t ff_blocks =
        (session->intermediate + kDecodeThreads - 1) / kDecodeThreads;
    hf_layer_ff_encode_kernel<<<ff_blocks, kDecodeThreads, 0, best->stream>>>(
        session->dtype, session->hidden, attention_hidden, kv_hidden,
        session->intermediate, session->device_step, session->max_context_tokens,
        session->device_scratch, session->device_projection_input);
    out->dependency_kernel_launches += 1;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  rc = run_stage(kLayerProjectionKinds[3], &stages[3]);
  if (rc != 0 || stages[3].status != 0 || stages[3].exact == 0) {
    out->exact = 0;
    out->status = stages[3].status;
    return rc;
  }

  for (NervaCudaHfDecodeSequenceSession *session : selected) {
    const uint32_t attention_hidden = session->heads * session->head_dim;
    const uint32_t kv_hidden = session->kv_heads * session->head_dim;
    if (request->layer_index + 1 < session->layer_count) {
      const SequenceLayerLayout next_layout =
          session->host_layouts[request->layer_index + 1];
      const uint64_t output_offset =
          (request->layer_index % 2 == 0) ? session->arena_layout.scratch
                                          : session->arena_layout.input;
      hf_layer_finish_next_attn_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, best->stream>>>(
          session->device_arena, output_offset, next_layout, session->dtype,
          session->dtype, session->hidden, attention_hidden, kv_hidden,
          session->intermediate, session->device_step,
          session->max_context_tokens, session->rms_eps, session->device_scratch,
          session->device_projection_input);
    } else {
      hf_layer_finish_final_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, best->stream>>>(
          session->device_arena, session->arena_layout, session->dtype,
          session->dtype, session->hidden, attention_hidden, kv_hidden,
          session->intermediate, session->device_step,
          session->max_context_tokens, session->rms_eps, session->device_scratch,
          session->device_projection_input);
    }
    out->dependency_kernel_launches += 1;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  if (best->projection_batch_defer_layer_sync == 0) {
    err = cudaStreamSynchronize(best->stream);
    out->sync_calls += 1;
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  const auto &qkv = stages[0];
  const auto &attention_output = stages[1];
  const auto &gate_up = stages[2];
  const auto &down = stages[3];
  out->reason = kProjectionBatchPlanReady;
  out->exact = 1;
  out->status = 0;
  out->qkv_rows = qkv.rows;
  out->attention_output_rows = attention_output.rows;
  out->gate_up_rows = gate_up.rows;
  out->down_rows = down.rows;
  out->hidden_cols = qkv.cols;
  out->attention_output_cols = attention_output.cols;
  out->down_cols = down.cols;
  out->input_bytes = qkv.input_bytes + attention_output.input_bytes +
                     gate_up.input_bytes + down.input_bytes;
  out->output_bytes = qkv.output_bytes + attention_output.output_bytes +
                      gate_up.output_bytes + down.output_bytes;
  out->qkv_elapsed_ns = qkv.elapsed_ns;
  out->attention_output_elapsed_ns = attention_output.elapsed_ns;
  out->gate_up_elapsed_ns = gate_up.elapsed_ns;
  out->down_elapsed_ns = down.elapsed_ns;
  out->elapsed_ns = qkv.elapsed_ns + attention_output.elapsed_ns +
                    gate_up.elapsed_ns + down.elapsed_ns;
  out->pack_kernel_launches = qkv.pack_kernel_launches +
                              attention_output.pack_kernel_launches +
                              gate_up.pack_kernel_launches +
                              down.pack_kernel_launches;
  out->projection_kernel_launches =
      qkv.projection_kernel_launches +
      attention_output.projection_kernel_launches +
      gate_up.projection_kernel_launches + down.projection_kernel_launches;
  out->scatter_kernel_launches = qkv.scatter_kernel_launches +
                                 attention_output.scatter_kernel_launches +
                                 gate_up.scatter_kernel_launches +
                                 down.scatter_kernel_launches;
  out->sync_calls += qkv.sync_calls + attention_output.sync_calls +
                     gate_up.sync_calls + down.sync_calls;
  out->hot_path_allocations = qkv.hot_path_allocations +
                              attention_output.hot_path_allocations +
                              gate_up.hot_path_allocations +
                              down.hot_path_allocations;
  return 0;
}
