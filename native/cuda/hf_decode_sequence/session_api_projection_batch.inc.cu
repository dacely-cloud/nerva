extern "C" int nerva_cuda_hf_decode_sequence_projection_batch_plan(
    const NervaCudaHfDecodeSequenceProjectionBatchPlanRequest *request,
    NervaCudaHfDecodeSequenceProjectionBatchPlanResult *out) {
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
  if (request->sessions == nullptr) {
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  std::vector<NervaCudaHfDecodeSequenceSession *> ready;
  ready.reserve(request->session_count);
  for (uint32_t index = 0; index < request->session_count; ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (session == nullptr || !session->active_started ||
        session->active_finished || session->active_prompt_token_count == 0 ||
        session->active_cursor >= session->max_context_tokens) {
      continue;
    }
    ready.push_back(session);
  }

  if (ready.empty()) {
    out->reason = kProjectionBatchPlanNoReadySessions;
    out->status = 0;
    return 0;
  }

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

  bool any_hash = false;
  for (const NervaCudaHfDecodeSequenceSession *session : ready) {
    any_hash = any_hash || session->planned_weight_descriptor_hash != 0;
  }
  if (!any_hash) {
    out->reason = kProjectionBatchPlanSharedWeightsUnproven;
    out->status = 0;
    return 0;
  }

  NervaCudaHfDecodeSequenceSession *best = nullptr;
  uint32_t best_count = 0;
  for (NervaCudaHfDecodeSequenceSession *candidate : ready) {
    if (candidate->planned_weight_descriptor_hash == 0) {
      continue;
    }
    uint32_t compatible = 0;
    for (NervaCudaHfDecodeSequenceSession *other : ready) {
      if (same_projection_model(candidate, other)) {
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

  const uint32_t block_tokens =
      std::min(std::min(best_count, out->target_block_tokens),
               kProjectionBatchWorkspaceTokens);
  const uint64_t attention_hidden =
      static_cast<uint64_t>(best->heads) * best->head_dim;
  const uint64_t kv_hidden =
      static_cast<uint64_t>(best->kv_heads) * best->head_dim;
  const PackedProjectionShape packed_shape = packed_projection_shape(
      best->hidden, attention_hidden, kv_hidden, best->intermediate);
  const uint64_t hidden = best->hidden;
  const uint64_t intermediate = best->intermediate;
  const uint64_t vocab_size = best->vocab_size;
  const uint64_t token_u16 = static_cast<uint64_t>(block_tokens) * sizeof(uint16_t);
  const uint64_t token_f32 = static_cast<uint64_t>(block_tokens) * sizeof(float);
  const uint64_t max_input_cols =
      std::max<uint64_t>(hidden, std::max<uint64_t>(attention_hidden, intermediate));
  const uint64_t max_output_rows =
      std::max<uint64_t>(
          vocab_size,
          std::max<uint64_t>(packed_shape.qkv_rows,
                             std::max<uint64_t>(packed_shape.gate_up_rows,
                                                hidden)));

  out->reason = kProjectionBatchPlanReady;
  out->exact = 1;
  out->block_tokens = block_tokens;
  out->dtype = best->dtype;
  out->hidden = best->hidden;
  out->heads = best->heads;
  out->kv_heads = best->kv_heads;
  out->head_dim = best->head_dim;
  out->intermediate = best->intermediate;
  out->vocab_size = best->vocab_size;
  out->layer_count = best->layer_count;
  out->max_context_tokens = best->max_context_tokens;
  out->planned_weight_descriptor_hash = best->planned_weight_descriptor_hash;
  out->resident_weight_bytes = best->resident_weight_bytes;
  out->qkv_rows = packed_shape.qkv_rows;
  out->gate_up_rows = packed_shape.gate_up_rows;
  out->qkv_input_bytes = hidden * token_u16;
  out->qkv_output_bytes = packed_shape.qkv_rows * token_f32;
  out->attention_output_input_bytes = attention_hidden * token_u16;
  out->attention_output_output_bytes = hidden * token_f32;
  out->gate_up_input_bytes = hidden * token_u16;
  out->gate_up_output_bytes = packed_shape.gate_up_rows * token_f32;
  out->down_input_bytes = intermediate * token_u16;
  out->down_output_bytes = hidden * token_f32;
  out->lm_head_input_bytes = hidden * token_u16;
  out->lm_head_output_bytes = vocab_size * token_f32;
  out->pack_input_bytes = max_input_cols * token_u16;
  out->max_projection_output_bytes = max_output_rows * token_f32;
  out->hot_path_allocations = 0;
  out->status = 0;
  return 0;
}

extern "C" int nerva_cuda_hf_decode_sequence_projection_batch_execute(
    const NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest *request,
    NervaCudaHfDecodeSequenceProjectionBatchExecuteResult *out) {
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
  out->projection_kind = request->projection_kind;
  out->layer_index = request->layer_index;
  if (request->session_count == 0) {
    out->reason = kProjectionBatchPlanNoSessions;
    out->status = 0;
    return 0;
  }
  if (request->sessions == nullptr) {
    return -1;
  }
  const bool layer_projection =
      request->projection_kind == kProjectionBatchKindQkv ||
      request->projection_kind == kProjectionBatchKindAttentionOutput ||
      request->projection_kind == kProjectionBatchKindGateUp ||
      request->projection_kind == kProjectionBatchKindDown;
  const bool lm_head_projection =
      request->projection_kind == kProjectionBatchKindLmHead;
  if (!layer_projection && !lm_head_projection) {
    out->reason = kProjectionBatchPlanUnsupportedProjection;
    out->status = 0;
    return 0;
  }
  if (lm_head_projection && request->layer_index != 0) {
    out->reason = kProjectionBatchPlanInvalidLayer;
    out->status = 0;
    return 0;
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
        (layer_projection && request->layer_index >= candidate->layer_count)) {
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
  if (layer_projection && request->layer_index >= best->layer_count) {
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
  uint32_t block_tokens = 0;
  for (uint32_t index = 0; index < request->session_count; ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (ready_session(session) && same_projection_model(best, session)) {
      block_tokens += 1;
      if (block_tokens >= max_block_tokens) {
        break;
      }
    }
  }
  if (block_tokens < out->min_block_tokens) {
    out->reason = kProjectionBatchPlanInsufficientCompatibleReady;
    out->status = 0;
    return 0;
  }

  const uint32_t attention_hidden = best->heads * best->head_dim;
  const uint32_t kv_hidden = best->kv_heads * best->head_dim;
  const PackedProjectionShape packed_shape = packed_projection_shape(
      best->hidden, attention_hidden, kv_hidden, best->intermediate);
  uint32_t rows = 0;
  uint32_t cols = 0;
  const uint16_t *matrix = nullptr;
  const SequenceLayerLayout layout =
      layer_projection ? best->host_layouts[request->layer_index]
                       : SequenceLayerLayout{};
  switch (request->projection_kind) {
    case kProjectionBatchKindQkv:
      rows = static_cast<uint32_t>(packed_shape.qkv_rows);
      cols = best->hidden;
      matrix = best->device_qkv_packed +
               packed_shape.qkv_elements_per_layer * request->layer_index;
      break;
    case kProjectionBatchKindAttentionOutput:
      rows = best->hidden;
      cols = attention_hidden;
      matrix = best->device_arena + layout.w_o;
      break;
    case kProjectionBatchKindGateUp:
      rows = static_cast<uint32_t>(packed_shape.gate_up_rows);
      cols = best->hidden;
      matrix = best->device_gate_up_packed +
               packed_shape.gate_up_elements_per_layer * request->layer_index;
      break;
    case kProjectionBatchKindDown:
      rows = best->hidden;
      cols = best->intermediate;
      matrix = best->device_arena + layout.w_down;
      break;
    case kProjectionBatchKindLmHead:
      rows = best->vocab_size;
      cols = best->hidden;
      matrix = best->device_arena + best->arena_layout.lm_head;
      break;
    default:
      break;
  }
  uint16_t *batch_input = best->device_projection_batch_input;
  float *batch_output = best->device_projection_batch_output;
  const uint64_t input_bytes =
      static_cast<uint64_t>(cols) * block_tokens * sizeof(uint16_t);
  const uint64_t output_bytes =
      static_cast<uint64_t>(rows) * block_tokens * sizeof(float);
  if (rows == 0 || cols == 0 || matrix == nullptr || batch_output == nullptr ||
      batch_input == nullptr || best->projection_batch_input_bytes < input_bytes ||
      best->projection_batch_output_bytes < output_bytes) {
    out->reason = kProjectionBatchPlanInsufficientScratch;
    out->status = 0;
    return 0;
  }

  auto scatter_destination =
      [&](NervaCudaHfDecodeSequenceSession *session) -> float * {
    LayerScratch scratch = layer_scratch_ptrs(
        session->device_scratch, session->hidden, attention_hidden, kv_hidden,
        session->intermediate);
    switch (request->projection_kind) {
      case kProjectionBatchKindQkv:
        return scratch.q;
      case kProjectionBatchKindAttentionOutput:
        return scratch.residual;
      case kProjectionBatchKindGateUp:
        return scratch.gate;
      case kProjectionBatchKindDown:
        return scratch.down;
      case kProjectionBatchKindLmHead:
        return session->device_scratch + session->hidden * 2;
      default:
        return nullptr;
    }
  };
  uint32_t selected_index = 0;
  if (best->projection_batch_peer_streams_synchronized == 0) {
    for (uint32_t index = 0; index < request->session_count &&
                             selected_index < block_tokens;
         ++index) {
      NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
      if (!ready_session(session) || !same_projection_model(best, session)) {
        continue;
      }
      selected_index += 1;
      if (session != best) {
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
  }

  const bool use_small_fused_batch = block_tokens >= 2 && block_tokens <= 32;
  const uint16_t *pack_src[32] = {
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  float *scatter_dst[32] = {
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  if (use_small_fused_batch) {
    selected_index = 0;
    for (uint32_t index = 0; index < request->session_count &&
                             selected_index < block_tokens;
         ++index) {
      NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
      if (!ready_session(session) || !same_projection_model(best, session)) {
        continue;
      }
      float *dst = scatter_destination(session);
      if (dst == nullptr) {
        err = cudaErrorInvalidValue;
        break;
      }
      pack_src[selected_index] = session->device_projection_input;
      scatter_dst[selected_index] = dst;
      selected_index += 1;
    }
    if (err == cudaSuccess) {
      for (uint32_t index = 0; index < block_tokens; ++index) {
        if (pack_src[index] == nullptr || scatter_dst[index] == nullptr) {
          err = cudaErrorInvalidValue;
          break;
        }
      }
    }
  }

  const bool collect_profile = best->detailed_profile != 0;
  if (collect_profile) {
    err = cudaEventRecord(best->device_start, best->stream);
  }
  if (err == cudaSuccess && use_small_fused_batch) {
    const uint32_t pack_blocks = ceil_div_u64_to_u32(
        static_cast<uint64_t>(cols) * block_tokens, kDecodeThreads);
    hf_projection_batch_pack_small_u16_kernel<<<pack_blocks, kDecodeThreads, 0,
                                                best->stream>>>(
        pack_src[0], pack_src[1], pack_src[2], pack_src[3], pack_src[4],
        pack_src[5], pack_src[6], pack_src[7], pack_src[8], pack_src[9],
        pack_src[10], pack_src[11], pack_src[12], pack_src[13], pack_src[14],
        pack_src[15], pack_src[16], pack_src[17], pack_src[18], pack_src[19],
        pack_src[20], pack_src[21], pack_src[22], pack_src[23], pack_src[24],
        pack_src[25], pack_src[26], pack_src[27], pack_src[28], pack_src[29],
        pack_src[30], pack_src[31], batch_input, cols, block_tokens);
    err = cudaGetLastError();
    out->pack_kernel_launches += 1;
  } else {
    const uint32_t pack_blocks = ceil_div_u32(cols, kDecodeThreads);
    selected_index = 0;
    for (uint32_t index = 0; err == cudaSuccess &&
                             index < request->session_count &&
                             selected_index < block_tokens;
         ++index) {
      NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
      if (!ready_session(session) || !same_projection_model(best, session)) {
        continue;
      }
      hf_projection_batch_pack_u16_kernel<<<pack_blocks, kDecodeThreads, 0,
                                            best->stream>>>(
          session->device_projection_input, batch_input, cols, selected_index);
      err = cudaGetLastError();
      out->pack_kernel_launches += 1;
      selected_index += 1;
    }
  }

  if (err == cudaSuccess) {
    const bool exact_projection =
        block_tokens > 16 && request->projection_kind == kProjectionBatchKindQkv;
    if (exact_projection) {
      err = encoded_row_major_gemv_strided_batched(
          best->cublas, matrix, batch_input, rows, cols, block_tokens,
          best->dtype, 0.0f, batch_output);
    } else {
      err = project_encoded_rows(best, nullptr, matrix, batch_input, rows, cols,
                                 block_tokens, best->dtype, 0.0f, batch_output);
    }
    out->projection_kernel_launches += 1;
  }

  if (err == cudaSuccess && use_small_fused_batch) {
    const uint32_t scatter_blocks = ceil_div_u64_to_u32(
        static_cast<uint64_t>(rows) * block_tokens, kDecodeThreads);
    hf_projection_batch_scatter_small_f32_kernel<<<
        scatter_blocks, kDecodeThreads, 0, best->stream>>>(
        batch_output, scatter_dst[0], scatter_dst[1], scatter_dst[2],
        scatter_dst[3], scatter_dst[4], scatter_dst[5], scatter_dst[6],
        scatter_dst[7], scatter_dst[8], scatter_dst[9], scatter_dst[10],
        scatter_dst[11], scatter_dst[12], scatter_dst[13], scatter_dst[14],
        scatter_dst[15], scatter_dst[16], scatter_dst[17], scatter_dst[18],
        scatter_dst[19], scatter_dst[20], scatter_dst[21], scatter_dst[22],
        scatter_dst[23], scatter_dst[24], scatter_dst[25], scatter_dst[26],
        scatter_dst[27], scatter_dst[28], scatter_dst[29], scatter_dst[30],
        scatter_dst[31], rows, block_tokens);
    err = cudaGetLastError();
    out->scatter_kernel_launches += 1;
  } else {
    const uint32_t scatter_blocks = ceil_div_u32(rows, kDecodeThreads);
    selected_index = 0;
    for (uint32_t index = 0; err == cudaSuccess &&
                             index < request->session_count &&
                             selected_index < block_tokens;
         ++index) {
      NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
      if (!ready_session(session) || !same_projection_model(best, session)) {
        continue;
      }
      float *scatter_dst = scatter_destination(session);
      if (scatter_dst == nullptr) {
        err = cudaErrorInvalidValue;
        break;
      }
      hf_projection_batch_scatter_f32_kernel<<<scatter_blocks, kDecodeThreads,
                                               0, best->stream>>>(
          batch_output, scatter_dst, rows, selected_index);
      err = cudaGetLastError();
      out->scatter_kernel_launches += 1;
      selected_index += 1;
    }
  }
  if (err == cudaSuccess && collect_profile) {
    err = cudaEventRecord(best->device_stop, best->stream);
  }
  if (err == cudaSuccess && collect_profile) {
    err = cudaEventSynchronize(best->device_stop);
    out->sync_calls += 1;
  }
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }

  uint64_t elapsed_ns = 0;
  if (collect_profile) {
    float elapsed_ms = 0.0f;
    err = cudaEventElapsedTime(&elapsed_ms, best->device_start,
                               best->device_stop);
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
    elapsed_ns = elapsed_ms <= 0.0f
                     ? 1
                     : static_cast<uint64_t>(elapsed_ms * 1000000.0f);
  }
  out->reason = kProjectionBatchPlanReady;
  out->exact = 1;
  out->block_tokens = block_tokens;
  out->dtype = best->dtype;
  out->rows = rows;
  out->cols = cols;
  out->input_bytes = input_bytes;
  out->output_bytes = output_bytes;
  out->elapsed_ns = elapsed_ns;
  out->hot_path_allocations = 0;
  out->status = 0;
  return 0;
}
