extern "C" int nerva_cuda_hf_decode_sequence_batch_advance_one(
    const NervaCudaHfDecodeSequenceBatchAdvanceRequest *request,
    NervaCudaHfDecodeSequenceBatchAdvanceResult *out) {
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
  if (request->session_count == 0) {
    out->reason = kProjectionBatchPlanNoSessions;
    out->status = 0;
    return 0;
  }
  if (request->sessions == nullptr || request->output_tokens == nullptr ||
      request->output_token_capacity < request->session_count) {
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  auto ready_session = [](const NervaCudaHfDecodeSequenceSession *session) {
    if (session == nullptr || !session->active_started ||
        session->active_finished || session->active_prompt_token_count == 0 ||
        session->active_cursor >= session->max_context_tokens ||
        !projection_batch_session_ready(session)) {
      return false;
    }
    const uint32_t prompt_count = session->active_prompt_token_count;
    const uint32_t target_cursor =
        prompt_count + session->active_observed_tokens;
    return target_cursor == session->active_cursor + 1u;
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
        candidate->layer_count == 0) {
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
  err = ensure_session_cublas_resources(best);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  const uint32_t max_block_tokens =
      std::min(out->target_block_tokens, kProjectionBatchWorkspaceTokens);
  std::vector<uint32_t> selected_indices;
  selected_indices.reserve(max_block_tokens);
  for (uint32_t index = 0; index < request->session_count &&
                           selected_indices.size() < max_block_tokens;
       ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (ready_session(session) && same_projection_model(best, session)) {
      selected_indices.push_back(index);
    }
  }
  if (selected_indices.size() < out->min_block_tokens) {
    out->reason = kProjectionBatchPlanInsufficientCompatibleReady;
    out->status = 0;
    return 0;
  }
  std::vector<NervaCudaHfDecodeSequenceSession *> selected_sessions;
  selected_sessions.reserve(selected_indices.size());
  for (uint32_t request_index : selected_indices) {
    selected_sessions.push_back(request->sessions[request_index]);
  }

  for (NervaCudaHfDecodeSequenceSession *session : selected_sessions) {
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
  ScopedProjectionBatchFlags batch_scope(best, true, true);

  out->reason = kProjectionBatchPlanReady;
  out->exact = 1;
  out->block_tokens = static_cast<uint32_t>(selected_indices.size());
  out->dtype = best->dtype;
  out->layer_count = best->layer_count;

  for (uint32_t layer_index = 0;
       layer_index < best->layer_count; ++layer_index) {
    NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteRequest layer_request{};
    layer_request.sessions = selected_sessions.data();
    layer_request.session_count =
        static_cast<uint32_t>(selected_sessions.size());
    layer_request.target_block_tokens = request->target_block_tokens;
    layer_request.min_block_tokens = request->min_block_tokens;
    layer_request.layer_index = layer_index;
    NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult layer_out{};
    const int rc = nerva_cuda_hf_decode_sequence_layer_projection_batch_execute(
        &layer_request, &layer_out);
    out->cuda_error = layer_out.cuda_error;
    out->device_count = layer_out.device_count;
    out->reason = layer_out.reason;
    out->eligible_session_count = layer_out.eligible_session_count;
    out->block_tokens = layer_out.block_tokens;
    out->target_block_tokens = layer_out.target_block_tokens;
    out->min_block_tokens = layer_out.min_block_tokens;
    out->dtype = layer_out.dtype;
    if (rc != 0 || layer_out.status != 0 || layer_out.exact == 0) {
      out->exact = 0;
      out->status = layer_out.status;
      return rc;
    }
    out->projection_elapsed_ns += layer_out.elapsed_ns;
    out->qkv_elapsed_ns += layer_out.qkv_elapsed_ns;
    out->attention_output_elapsed_ns += layer_out.attention_output_elapsed_ns;
    out->gate_up_elapsed_ns += layer_out.gate_up_elapsed_ns;
    out->down_elapsed_ns += layer_out.down_elapsed_ns;
    out->pack_kernel_launches += layer_out.pack_kernel_launches;
    out->projection_kernel_launches += layer_out.projection_kernel_launches;
    out->scatter_kernel_launches += layer_out.scatter_kernel_launches;
    out->dependency_kernel_launches += layer_out.dependency_kernel_launches;
    out->experimental_rt_selector_launches +=
        layer_out.experimental_rt_selector_launches;
    out->sync_calls += layer_out.sync_calls;
    out->hot_path_allocations += layer_out.hot_path_allocations;
  }

  NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest lm_head_request{};
  lm_head_request.sessions = selected_sessions.data();
  lm_head_request.session_count =
      static_cast<uint32_t>(selected_sessions.size());
  lm_head_request.target_block_tokens = request->target_block_tokens;
  lm_head_request.min_block_tokens = request->min_block_tokens;
  lm_head_request.projection_kind = kProjectionBatchKindLmHead;
  lm_head_request.layer_index = 0;
  NervaCudaHfDecodeSequenceProjectionBatchExecuteResult lm_head_out{};
  const int lm_rc = nerva_cuda_hf_decode_sequence_projection_batch_execute(
      &lm_head_request, &lm_head_out);
  out->cuda_error = lm_head_out.cuda_error;
  out->device_count = lm_head_out.device_count;
  out->reason = lm_head_out.reason;
  out->eligible_session_count = lm_head_out.eligible_session_count;
  out->block_tokens = lm_head_out.block_tokens;
  out->target_block_tokens = lm_head_out.target_block_tokens;
  out->min_block_tokens = lm_head_out.min_block_tokens;
  out->dtype = lm_head_out.dtype;
  if (lm_rc != 0 || lm_head_out.status != 0 || lm_head_out.exact == 0) {
    out->exact = 0;
    out->status = lm_head_out.status;
    return lm_rc;
  }
  out->projection_elapsed_ns += lm_head_out.elapsed_ns;
  out->lm_head_elapsed_ns = lm_head_out.elapsed_ns;
  out->pack_kernel_launches += lm_head_out.pack_kernel_launches;
  out->projection_kernel_launches += lm_head_out.projection_kernel_launches;
  out->scatter_kernel_launches += lm_head_out.scatter_kernel_launches;
  out->sync_calls += lm_head_out.sync_calls;
  out->hot_path_allocations += lm_head_out.hot_path_allocations;

  for (uint32_t selected = 0; selected < selected_indices.size(); ++selected) {
    const uint32_t request_index = selected_indices[selected];
    NervaCudaHfDecodeSequenceSession *session = request->sessions[request_index];
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        best->stream, session->device_step, session->max_context_tokens,
        session->active_has_eos_token, session->active_eos_token,
        device_logits, session->vocab_size, session->device_slots,
        session->active_sampler);
    out->sampling_kernel_launches += 1;
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  for (uint32_t selected = 0; selected < selected_indices.size(); ++selected) {
    const uint32_t request_index = selected_indices[selected];
    NervaCudaHfDecodeSequenceSession *session = request->sessions[request_index];
    const uint32_t slot_start =
        session->active_prompt_token_count - 1u + session->active_observed_tokens;
    err = cudaMemcpyAsync(session->host_slots + slot_start,
                          session->device_slots + slot_start,
                          sizeof(NervaCudaSyntheticTokenSlot),
                          cudaMemcpyDeviceToHost, best->stream);
    out->d2h_bytes += sizeof(NervaCudaSyntheticTokenSlot);
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }
  err = cudaStreamSynchronize(best->stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }
  for (NervaCudaHfDecodeSequenceSession *session : selected_sessions) {
    session->projection_batch_own_stream_synchronized = 1;
  }

  std::vector<uint32_t> observed;
  observed.reserve(selected_indices.size());
  for (uint32_t selected = 0; selected < selected_indices.size(); ++selected) {
    const uint32_t request_index = selected_indices[selected];
    NervaCudaHfDecodeSequenceSession *session = request->sessions[request_index];
    const uint32_t slot_start =
        session->active_prompt_token_count - 1u + session->active_observed_tokens;
    const NervaCudaSyntheticTokenSlot &slot = session->host_slots[slot_start];
    if (slot.request_id != kRequestId || slot.sequence_id != kSequenceId ||
        slot.completion != kCompletionDeviceComplete ||
        slot.token_index != slot_start) {
      out->status = -1;
      return -1;
    }
    request->output_tokens[request_index] = slot.token;
    observed.push_back(slot.token);
    out->last_token = slot.token;
    session->active_observed_tokens += 1;
    session->active_cursor += 1;
    const uint32_t kv_tokens = slot_start + 1u;
    session->active_finished =
        (session->active_has_eos_token != 0 &&
         slot.token == session->active_eos_token) ||
        kv_tokens >= session->max_context_tokens;
  }
  out->observed_tokens = static_cast<uint32_t>(observed.size());
  out->observed_token_hash =
      hash_tokens(observed.data(), static_cast<uint32_t>(observed.size()));
  out->reason = kProjectionBatchPlanReady;
  out->exact = 1;
  out->status = out->observed_tokens == out->block_tokens ? 0 : -1;
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_hf_decode_sequence_session_fork_shared_weights(
    const NervaCudaHfDecodeSequenceSessionForkSharedWeightsRequest *request,
    NervaCudaHfDecodeSequenceSessionCreateResult *out,
    NervaCudaHfDecodeSequenceSession **session_out) {
  if (out == nullptr || session_out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  *session_out = nullptr;
  if (request == nullptr || request->parent == nullptr) {
    out->failure_stage = kCreateStageInvalidRequest;
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *parent = request->parent;
  if (parent->shared_weights == nullptr || !use_cublas_layer_path(parent)) {
    out->failure_stage = kCreateStageInvalidRequest;
    return -1;
  }
  const bool clone_active_state =
      parent->active_started && !parent->active_finished &&
      parent->active_prompt_token_count != 0;

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err, kCreateStageGetDeviceCount);
  }
  if (out->device_count <= 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorNoDevice);
    out->failure_stage = kCreateStageGetDeviceCount;
    return -1;
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err, kCreateStageSetDevice);
  }

  auto *session = new (std::nothrow) NervaCudaHfDecodeSequenceSession();
  if (session == nullptr) {
    out->cuda_error = static_cast<int32_t>(cudaErrorMemoryAllocation);
    out->failure_stage = kCreateStageSessionAlloc;
    return -1;
  }

  session->dtype = parent->dtype;
  session->hidden = parent->hidden;
  session->heads = parent->heads;
  session->kv_heads = parent->kv_heads;
  session->head_dim = parent->head_dim;
  session->head_threads = parent->head_threads;
  session->intermediate = parent->intermediate;
  session->vocab_size = parent->vocab_size;
  session->layer_count = parent->layer_count;
  session->max_context_tokens = parent->max_context_tokens;
  session->kv_block_count = parent->kv_block_count;
  session->kv_token_capacity = parent->kv_token_capacity;
  session->prefill_chunk_tokens = parent->prefill_chunk_tokens;
  session->detailed_profile = request->detailed_profile == 0 ? 0u : 1u;
  session->experimental_rt_decode_requested =
      parent->experimental_rt_decode_requested;
  session->experimental_rt_decode_enabled = 0;
  session->experimental_rt_mode = parent->experimental_rt_mode;
  session->experimental_rt_page_tokens = parent->experimental_rt_page_tokens;
  session->experimental_rt_pages = parent->experimental_rt_pages;
  session->experimental_rt_local_window_tokens =
      parent->experimental_rt_local_window_tokens;
  session->experimental_rt_sink_tokens = parent->experimental_rt_sink_tokens;
  session->experimental_rt_query_descriptor_selector =
      parent->experimental_rt_query_descriptor_selector;
  session->experimental_rt_kv_descriptor_selector =
      parent->experimental_rt_kv_descriptor_selector;
  session->experimental_rt_query_key_selector =
      parent->experimental_rt_query_key_selector;
  session->experimental_rt_query_key_fused_selector =
      parent->experimental_rt_query_key_fused_selector;
  session->experimental_prefill_local_window_tokens =
      parent->experimental_prefill_local_window_tokens;
  session->rms_eps = parent->rms_eps;
  session->rope_theta = parent->rope_theta;
  session->arena_layout = parent->arena_layout;
  session->arena_bytes = parent->arena_bytes;
  session->resident_weight_bytes = parent->resident_weight_bytes;
  session->layout_bytes = parent->layout_bytes;
  session->scratch_bytes = parent->scratch_bytes;
  session->projection_input_bytes = parent->projection_input_bytes;
  session->projection_batch_input_bytes = parent->projection_batch_input_bytes;
  session->projection_batch_output_bytes = parent->projection_batch_output_bytes;
  session->prefill_hidden_bytes =
      clone_active_state ? 0 : parent->prefill_hidden_bytes;
  session->prefill_norm_bytes =
      clone_active_state ? 0 : parent->prefill_norm_bytes;
  session->prefill_qkv_bytes =
      clone_active_state ? 0 : parent->prefill_qkv_bytes;
  session->prefill_qkv_encoded_bytes =
      clone_active_state ? 0 : parent->prefill_qkv_encoded_bytes;
  session->prefill_attn_bytes =
      clone_active_state ? 0 : parent->prefill_attn_bytes;
  session->prefill_o_bytes =
      clone_active_state ? 0 : parent->prefill_o_bytes;
  session->prefill_q_gate_bytes =
      clone_active_state ? 0 : parent->prefill_q_gate_bytes;
  session->prefill_gate_up_bytes =
      clone_active_state ? 0 : parent->prefill_gate_up_bytes;
  session->prefill_ff_bytes =
      clone_active_state ? 0 : parent->prefill_ff_bytes;
  session->prefill_down_bytes =
      clone_active_state ? 0 : parent->prefill_down_bytes;
  session->decode_attention_values_bytes =
      parent->decode_attention_values_bytes;
  session->decode_attention_stats_bytes = parent->decode_attention_stats_bytes;
  session->decode_attention_max_chunks = parent->decode_attention_max_chunks;
  session->decode_q_bytes = parent->decode_q_bytes;
  session->decode_seq_len_bytes = parent->decode_seq_len_bytes;
  session->packed_qkv_bytes = parent->packed_qkv_bytes;
  session->packed_gate_up_bytes = parent->packed_gate_up_bytes;
  session->kv_bytes = parent->kv_bytes;
  copy_deepseek_session_byte_fields(session, parent);
  session->kv_block_table_bytes = parent->kv_block_table_bytes;
  session->slots_bytes = parent->slots_bytes;
  session->prompt_bytes = parent->prompt_bytes;
  session->planned_weight_blocks = parent->planned_weight_blocks;
  session->planned_gpu_resident_blocks = parent->planned_gpu_resident_blocks;
  session->planned_gpu_staged_blocks = parent->planned_gpu_staged_blocks;
  session->planned_weight_bytes = parent->planned_weight_bytes;
  session->planned_gpu_resident_weight_bytes =
      parent->planned_gpu_resident_weight_bytes;
  session->planned_gpu_staged_weight_bytes =
      parent->planned_gpu_staged_weight_bytes;
  session->planned_weight_descriptor_count =
      parent->planned_weight_descriptor_count;
  session->planned_weight_descriptor_hash =
      parent->planned_weight_descriptor_hash;
  session->host_layouts = parent->host_layouts;
  session->shared_weights = parent->shared_weights;
  session->device_arena = session->shared_weights->device_arena;
  session->device_layouts = session->shared_weights->device_layouts;
  session->device_qkv_packed = session->shared_weights->device_qkv_packed;
  session->device_gate_up_packed =
      session->shared_weights->device_gate_up_packed;

  int32_t failure_stage = kCreateStageHostSlotsAlloc;
  err = cudaHostAlloc(reinterpret_cast<void **>(&session->host_slots),
                      session->slots_bytes, cudaHostAllocDefault);
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDeviceScratchAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_scratch),
                     session->scratch_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageProjectionInputAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_projection_input),
                     session->projection_input_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_projection_batch_input),
        session->projection_batch_input_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_projection_batch_output),
        session->projection_batch_output_bytes);
  }
  if (err == cudaSuccess && session->prefill_hidden_bytes != 0) {
    failure_stage = kCreateStagePrefillHiddenAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_hidden_a),
                     session->prefill_hidden_bytes);
  }
  if (err == cudaSuccess && session->prefill_hidden_bytes != 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_hidden_b),
                     session->prefill_hidden_bytes);
  }
  if (err == cudaSuccess && session->prefill_norm_bytes != 0) {
    failure_stage = kCreateStagePrefillChunkAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_norm),
                     session->prefill_norm_bytes);
  }
  if (err == cudaSuccess && session->prefill_qkv_bytes != 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_qkv),
                     session->prefill_qkv_bytes);
  }
  if (err == cudaSuccess && session->prefill_qkv_encoded_bytes != 0) {
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_prefill_qkv_encoded),
        session->prefill_qkv_encoded_bytes);
  }
  if (err == cudaSuccess && session->prefill_attn_bytes != 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_attn),
                     session->prefill_attn_bytes);
  }
  if (err == cudaSuccess && session->prefill_o_bytes != 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_o),
                     session->prefill_o_bytes);
  }
  if (err == cudaSuccess && session->prefill_q_gate_bytes != 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_q_gate),
                     session->prefill_q_gate_bytes);
  }
  if (err == cudaSuccess && session->prefill_gate_up_bytes != 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_gate_up),
                     session->prefill_gate_up_bytes);
  }
  if (err == cudaSuccess && session->prefill_ff_bytes != 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_ff),
                     session->prefill_ff_bytes);
  }
  if (err == cudaSuccess && session->prefill_down_bytes != 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_down),
                     session->prefill_down_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDecodeAttentionAlloc;
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_decode_attention_values),
        session->decode_attention_values_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_decode_attention_m),
                     session->decode_attention_stats_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_decode_attention_l),
                     session->decode_attention_stats_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDecodeSdpaAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_decode_q),
                     session->decode_q_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_decode_seq_len_q),
        sizeof(int32_t));
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_decode_seq_len_kv),
        sizeof(int32_t));
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageKvKeysAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_kv_keys),
                     session->kv_bytes / 2);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageKvValuesAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_kv_values),
                     session->kv_bytes / 2);
  }
  if (err == cudaSuccess) {
    err = allocate_deepseek_session_device_state(session, &failure_stage);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_kv_block_table),
                     session->kv_block_table_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStagePromptTokensAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prompt_tokens),
                     session->prompt_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDeviceSlotsAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_slots),
                     session->slots_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDeviceStepAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_step),
                     sizeof(uint32_t));
  }
  if (err == cudaSuccess && !clone_active_state) {
    failure_stage = kCreateStageCublasWorkspaceAlloc;
    err = cudaMalloc(&session->cublas_workspace, kCublasWorkspaceBytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageStreamCreate;
    err = cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking);
  }
  if (err == cudaSuccess && !clone_active_state) {
    failure_stage = kCreateStageCublasCreate;
    err = cublas_to_cuda(cublasCreate(&session->cublas));
  }
  if (err == cudaSuccess && !clone_active_state) {
    failure_stage = kCreateStageCublasLtCreate;
    err = cublas_to_cuda(cublasLtCreate(&session->cublas_lt));
  }
#if NERVA_HAVE_CUDNN_FRONTEND
  if (err == cudaSuccess && !clone_active_state) {
    failure_stage = kCreateStageCublasConfigure;
    err = cudnn_to_cuda(cudnnCreate(&session->cudnn));
  }
#endif
  if (err == cudaSuccess && !clone_active_state) {
    failure_stage = kCreateStageCublasConfigure;
    err = configure_cublas(session->cublas, session->stream,
                           session->cublas_workspace,
                           kCublasWorkspaceBytes);
  }
#if NERVA_HAVE_CUDNN_FRONTEND
  if (err == cudaSuccess && !clone_active_state) {
    err = cudnn_to_cuda(cudnnSetStream(session->cudnn, session->stream));
  }
#endif
  if (err == cudaSuccess) {
    failure_stage = kCreateStageStartEventCreate;
    err = cudaEventCreate(&session->device_start);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageStopEventCreate;
    err = cudaEventCreate(&session->device_stop);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageStartEventCreate;
    err = cudaEventCreate(&session->profile_start);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageStopEventCreate;
    err = cudaEventCreate(&session->profile_stop);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDeepSeekV4AttentionAuxInit;
    err = initialize_deepseek_v4_attention_aux_resources(session);
  }
  if (err == cudaSuccess && session->experimental_rt_decode_requested != 0) {
    failure_stage = kCreateStageExperimentalRtDecodeInit;
    err = initialize_experimental_rt_selector(session);
  }
  if (err == cudaSuccess) {
    const uint32_t blocks =
        (session->kv_block_count + kDecodeThreads - 1u) / kDecodeThreads;
    hf_init_identity_kv_block_table_kernel<<<blocks, kDecodeThreads, 0,
                                             session->stream>>>(
        session->device_kv_block_table, session->kv_block_count);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess && !clone_active_state) {
    err = reset_deepseek_session_device_state(session);
  }
  if (err == cudaSuccess && !clone_active_state) {
    failure_stage = kCreateStageProjectionPlanAutotune;
    err = autotune_session_lt_gemv_plans(session);
  }
  if (err == cudaSuccess && clone_active_state) {
    failure_stage = kCreateStageSetupSynchronize;
    err = cudaStreamSynchronize(parent->stream);
  }
  if (err == cudaSuccess && clone_active_state) {
    memcpy(session->host_slots, parent->host_slots, session->slots_bytes);
    err = cudaMemcpyAsync(session->device_prompt_tokens,
                          parent->device_prompt_tokens, session->prompt_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  if (err == cudaSuccess && clone_active_state) {
    err = cudaMemcpyAsync(session->device_slots, parent->device_slots,
                          session->slots_bytes, cudaMemcpyDeviceToDevice,
                          session->stream);
  }
  if (err == cudaSuccess && clone_active_state) {
    err = cudaMemcpyAsync(session->device_step, parent->device_step,
                          sizeof(uint32_t), cudaMemcpyDeviceToDevice,
                          session->stream);
  }
  if (err == cudaSuccess && clone_active_state) {
    err = cudaMemcpyAsync(session->device_kv_keys, parent->device_kv_keys,
                          session->kv_bytes / 2, cudaMemcpyDeviceToDevice,
                          session->stream);
  }
  if (err == cudaSuccess && clone_active_state) {
    err = cudaMemcpyAsync(session->device_kv_values, parent->device_kv_values,
                          session->kv_bytes / 2, cudaMemcpyDeviceToDevice,
                          session->stream);
  }
  if (err == cudaSuccess && clone_active_state) {
    err = clone_deepseek_session_device_state(session, parent);
  }
  if (err == cudaSuccess && clone_active_state) {
    session->active_prompt_token_count = parent->active_prompt_token_count;
    session->active_has_eos_token = parent->active_has_eos_token;
    session->active_eos_token = parent->active_eos_token;
    session->active_seed_token = parent->active_seed_token;
    session->active_sampler = parent->active_sampler;
    session->active_observed_tokens = parent->active_observed_tokens;
    session->active_cursor = parent->active_cursor;
    session->active_started = parent->active_started;
    session->active_finished = parent->active_finished;
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageSetupSynchronize;
    err = cudaStreamSynchronize(session->stream);
  }
  if (err != cudaSuccess) {
    fail(out, err, failure_stage);
    free_session_fields(session);
    delete session;
    return -1;
  }

  session->setup_sync_calls = clone_active_state ? 2 : 1;
  session->projection_batch_own_stream_synchronized = 1;
  fill_create_result(session, out);
  *session_out = session;
  return 0;
}

extern "C" int nerva_cuda_hf_decode_sequence_session_destroy(
    NervaCudaHfDecodeSequenceSession *session,
    NervaCudaHfDecodeSequenceSessionCreateResult *out) {
  if (out != nullptr) {
    memset(out, 0, sizeof(*out));
    out->status = -1;
  }
  if (session == nullptr) {
    return -1;
  }
  if (out != nullptr) {
    fill_create_result(session, out);
  }
  free_session_fields(session);
  delete session;
  return 0;
}
