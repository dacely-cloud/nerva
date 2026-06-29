void fill_session_result_header(const NervaCudaHfDecodeSequenceSession *session,
                                NervaCudaHfDecodeSequenceResult *out,
                                uint32_t steps, uint32_t seed_token) {
  out->device_count = 1;
  out->dtype = session->dtype;
  out->hidden = session->hidden;
  out->heads = session->heads;
  out->kv_heads = session->kv_heads;
  out->head_dim = session->head_dim;
  out->intermediate = session->intermediate;
  out->vocab_size = session->vocab_size;
  out->layer_count = session->layer_count;
  out->steps = steps;
  out->seed_token = seed_token;
  out->resident_weight_bytes = session->resident_weight_bytes;
  out->planned_weight_blocks = session->planned_weight_blocks;
  out->planned_gpu_resident_blocks = session->planned_gpu_resident_blocks;
  out->planned_gpu_staged_blocks = session->planned_gpu_staged_blocks;
  out->planned_weight_bytes = session->planned_weight_bytes;
  out->planned_gpu_resident_weight_bytes =
      session->planned_gpu_resident_weight_bytes;
  out->planned_gpu_staged_weight_bytes =
      session->planned_gpu_staged_weight_bytes;
  out->planned_weight_descriptor_count =
      session->planned_weight_descriptor_count;
  out->planned_weight_descriptor_hash =
      session->planned_weight_descriptor_hash;
}

uint32_t observed_from_slot_range(uint32_t steps, uint32_t has_eos_token,
                                  uint32_t eos_token,
                                  const NervaCudaSyntheticTokenSlot *slots) {
  uint32_t count = steps;
  for (uint32_t index = 0; index < steps; ++index) {
    if (slots[index].completion != kCompletionDeviceComplete) {
      count = index;
      break;
    }
    if (has_eos_token != 0 && slots[index].token == eos_token) {
      count = index + 1;
      break;
    }
  }
  return count;
}

cudaError_t ensure_session_graph(NervaCudaHfDecodeSequenceSession *session,
                                 uint32_t max_steps,
                                 uint32_t prompt_token_count,
                                 uint32_t has_eos_token,
                                 uint32_t eos_token,
                                 uint32_t attention_chunks,
                                 uint32_t profile_cursor,
                                 NervaCudaHfDecodeSequenceResult *out) {
  uint32_t cache_attention_chunks = attention_chunks;
#if NERVA_HAVE_CUDNN_FRONTEND
  if (session->cudnn_decode_sdpa != nullptr &&
      can_use_cudnn_decode_sdpa(session, attention_chunks)) {
    cache_attention_chunks = 1;
  }
#endif
  if (session_graph_matches(session, max_steps, prompt_token_count,
                            has_eos_token, eos_token, cache_attention_chunks,
                            session->active_sampler)) {
    out->graph_nodes = session->cached_graph_nodes;
    out->graph_cache_hits = 1;
    copy_cached_profile(session, out);
    return cudaSuccess;
  }
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  cudaError_t err = cudaSuccess;
  for (uint32_t attempt = 0; attempt < 2; ++attempt) {
    reset_session_graph(session);
    bool tried_cudnn_decode_sdpa = false;
    bool captured_cudnn_decode_sdpa = false;
#if NERVA_HAVE_CUDNN_FRONTEND
    if (can_use_cudnn_decode_sdpa(session, attention_chunks)) {
      tried_cudnn_decode_sdpa = true;
      err = ensure_cudnn_decode_sdpa_plan(session);
      if (err != cudaSuccess) {
        session->cudnn_decode_sdpa_disabled = 1;
        err = cudaSuccess;
        tried_cudnn_decode_sdpa = false;
      } else {
        captured_cudnn_decode_sdpa = true;
      }
    }
#endif
    bool capture_started = false;
    err = cudaStreamBeginCapture(session->stream, cudaStreamCaptureModeGlobal);
    capture_started = err == cudaSuccess;
    if (err == cudaSuccess) {
      err = use_cublas_layer_path(session)
                ? launch_cublas_layer_session_step(
                      session, max_steps, prompt_token_count, has_eos_token,
                      eos_token, attention_chunks)
                : launch_monolithic_session_step(
                      session, max_steps, prompt_token_count, has_eos_token,
                      eos_token);
    }
    if (capture_started) {
      cudaError_t end_err = cudaStreamEndCapture(session->stream, &graph);
      if (err == cudaSuccess) {
        err = end_err;
      } else if (graph != nullptr) {
        cudaGraphDestroy(graph);
        graph = nullptr;
      }
    }
    if (err == cudaSuccess) {
      size_t graph_nodes = 0;
      err = cudaGraphGetNodes(graph, nullptr, &graph_nodes);
      out->graph_nodes = static_cast<uint64_t>(graph_nodes);
    }
    if (err == cudaSuccess) {
      err = cudaGraphInstantiate(&graph_exec, graph, 0);
    }
    if (err == cudaSuccess) {
      session->cached_graph = graph;
      session->cached_graph_exec = graph_exec;
      session->cached_context_steps = max_steps;
      session->cached_prompt_token_count = prompt_token_count;
      session->cached_has_eos_token = has_eos_token;
      session->cached_eos_token = eos_token;
      session->cached_attention_chunks = captured_cudnn_decode_sdpa
                                             ? 1
                                             : attention_chunks;
      session->cached_experimental_rt_sparse_attention_active =
          session->experimental_rt_sparse_attention_active;
      session->cached_sampler = session->active_sampler;
      session->cached_graph_nodes = out->graph_nodes;
      out->graph_captures = 1;
      graph = nullptr;
      graph_exec = nullptr;
      break;
    }
#if NERVA_HAVE_CUDNN_FRONTEND
    if (tried_cudnn_decode_sdpa) {
      log_cudnn_decode_cuda_error("graph capture", err);
      session->cudnn_decode_sdpa_disabled = 1;
      if (graph_exec != nullptr) {
        cudaGraphExecDestroy(graph_exec);
        graph_exec = nullptr;
      }
      if (graph != nullptr) {
        cudaGraphDestroy(graph);
        graph = nullptr;
      }
      continue;
    }
#endif
    break;
  }
  if (err == cudaSuccess && use_cublas_layer_path(session) &&
      session->detailed_profile != 0) {
    err = profile_cublas_layer_session_step(
        session, max_steps, prompt_token_count, has_eos_token, eos_token,
        attention_chunks, profile_cursor);
    if (err == cudaSuccess) {
      copy_cached_profile(session, out);
    }
  }
  if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  return err;
}

void fill_create_result(const NervaCudaHfDecodeSequenceSession *session,
                        NervaCudaHfDecodeSequenceSessionCreateResult *out) {
  out->status = 0;
  out->dtype = session->dtype;
  out->hidden = session->hidden;
  out->heads = session->heads;
  out->kv_heads = session->kv_heads;
  out->head_dim = session->head_dim;
  out->intermediate = session->intermediate;
  out->vocab_size = session->vocab_size;
  out->layer_count = session->layer_count;
  out->max_context_tokens = session->max_context_tokens;
  out->prefill_chunk_tokens = session->prefill_chunk_tokens;
  out->head_threads = session->head_threads;
  out->resident_weight_bytes = session->resident_weight_bytes;
  out->planned_weight_blocks = session->planned_weight_blocks;
  out->planned_gpu_resident_blocks = session->planned_gpu_resident_blocks;
  out->planned_gpu_staged_blocks = session->planned_gpu_staged_blocks;
  out->planned_weight_bytes = session->planned_weight_bytes;
  out->planned_gpu_resident_weight_bytes = session->planned_gpu_resident_weight_bytes;
  out->planned_gpu_staged_weight_bytes = session->planned_gpu_staged_weight_bytes;
  out->planned_weight_descriptor_count = session->planned_weight_descriptor_count;
  out->planned_weight_descriptor_hash = session->planned_weight_descriptor_hash;
  out->experimental_rt_decode_requested =
      session->experimental_rt_decode_requested;
  out->experimental_rt_decode_enabled = session->experimental_rt_decode_enabled;
  out->experimental_rt_mode = session->experimental_rt_mode;
  out->experimental_rt_page_tokens = session->experimental_rt_page_tokens;
  out->experimental_rt_pages = session->experimental_rt_pages;
  out->experimental_rt_local_window_tokens =
      session->experimental_rt_local_window_tokens;
  out->experimental_rt_sink_tokens = session->experimental_rt_sink_tokens;
  out->descriptor_gpu_resident_h2d_bytes = session->descriptor_gpu_resident_h2d_bytes;
  out->descriptor_gpu_staged_h2d_bytes = session->descriptor_gpu_staged_h2d_bytes;
  out->resident_kv_bytes = session->kv_bytes;
  out->device_arena_bytes = session_device_footprint(session);
  out->pinned_host_bytes = session->slots_bytes + session->load_staging_bytes;
  out->h2d_bytes = session->h2d_bytes;
  out->sync_calls = session->setup_sync_calls + 1;
}

int fail(NervaCudaHfDecodeSequenceSessionCreateResult *out, cudaError_t err,
         int32_t failure_stage) {
  out->cuda_error = static_cast<int32_t>(err);
  out->failure_stage = failure_stage;
  return -1;
}
