extern "C" int nerva_cuda_hf_decode_sequence_u16(
    const NervaCudaHfDecodeSequenceRequest *request,
    NervaCudaHfDecodeSequenceResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(request, out);
  if (!valid_request(request)) {
    return -1;
  }
  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorNoDevice);
    return -1;
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  const uint64_t hidden = request->hidden;
  const NervaCudaHfDecodeSamplerConfig request_sampler =
      normalize_hf_decode_sampler_config(request->sampler);
  const uint64_t attention_hidden = request->heads * request->head_dim;
  const uint64_t kv_hidden = request->kv_heads * request->head_dim;
  const uint64_t intermediate = request->intermediate;
  const uint64_t vocab_size = request->vocab_size;
  const uint32_t context_steps = request->prompt_token_count + request->steps - 1u;
  SequenceArenaLayout arena_layout{};
  std::vector<SequenceLayerLayout> layouts(request->layer_count);
  uint64_t elements = 0;
  arena_layout.embeddings = push(elements, vocab_size * hidden);
  arena_layout.input = push(elements, hidden);
  arena_layout.scratch = push(elements, hidden);
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    pack_layer(layouts[index], elements, request->layers[index], hidden,
               attention_hidden, kv_hidden, request->head_dim, intermediate);
  }
  uint64_t linear_gdn_conv_state_elements = 0;
  uint64_t linear_gdn_recurrent_state_elements = 0;
  assign_linear_gdn_state_offsets(layouts, &linear_gdn_conv_state_elements,
                                  &linear_gdn_recurrent_state_elements);
  arena_layout.final_norm = push(elements, hidden);
  arena_layout.lm_head = push(elements, vocab_size * hidden);
  const uint64_t arena_bytes = elements * sizeof(uint16_t);
  const uint64_t resident_weight_bytes = arena_bytes - (hidden * 2 * sizeof(uint16_t));
  if (request->planned_weight_blocks != 0 &&
      request->planned_weight_bytes != resident_weight_bytes) {
    out->status = -1;
    return -1;
  }
  if (!validate_weight_descriptors(request, resident_weight_bytes, out)) {
    out->status = -1;
    return -1;
  }
  const uint64_t layout_bytes = layouts.size() * sizeof(SequenceLayerLayout);
  const uint64_t block_scratch = max_layer_scratch_elements(
      layouts, hidden, attention_hidden, kv_hidden, intermediate);
  const uint64_t final_scratch = hidden * 2 + vocab_size;
  const uint64_t scratch_elements =
      block_scratch > final_scratch ? block_scratch : final_scratch;
  const uint64_t scratch_bytes = scratch_elements * sizeof(float);
  const uint32_t kv_block_count =
      ceil_div_u32(context_steps, kKvCacheBlockTokens);
  const uint32_t kv_token_capacity = kv_block_count * kKvCacheBlockTokens;
  const uint64_t kv_bytes =
      request->layer_count * static_cast<uint64_t>(kv_token_capacity) * kv_hidden *
      sizeof(uint16_t) * 2;
  const uint64_t kv_block_table_bytes =
      static_cast<uint64_t>(kv_block_count) * sizeof(uint32_t);
  const uint64_t linear_gdn_conv_state_bytes =
      linear_gdn_conv_state_elements * sizeof(float);
  const uint64_t linear_gdn_recurrent_state_bytes =
      linear_gdn_recurrent_state_elements * sizeof(float);
  const uint64_t slots_bytes =
      static_cast<uint64_t>(context_steps) * sizeof(NervaCudaSyntheticTokenSlot);
  const uint64_t prompt_bytes =
      static_cast<uint64_t>(request->prompt_token_count) * sizeof(uint32_t);
  const bool descriptor_mode = request->planned_weight_blocks != 0;
  const uint64_t host_weight_bytes =
      descriptor_mode ? pinned_weight_staging_bytes(request, resident_weight_bytes)
                      : arena_bytes;
  uint64_t setup_sync_calls = 0;

  uint16_t *host_arena = nullptr;
  uint16_t *device_arena = nullptr;
  SequenceLayerLayout *device_layouts = nullptr;
  float *device_scratch = nullptr;
  uint16_t *device_kv_keys = nullptr;
  uint16_t *device_kv_values = nullptr;
  uint32_t *device_kv_block_table = nullptr;
  float *device_linear_gdn_conv_state = nullptr;
  float *device_linear_gdn_recurrent_state = nullptr;
  uint32_t *device_prompt_tokens = nullptr;
  NervaCudaSyntheticTokenSlot *host_slots = nullptr;
  NervaCudaSyntheticTokenSlot *device_slots = nullptr;
  uint32_t *device_step = nullptr;
  void *cublas_workspace = nullptr;
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas = nullptr;
  cudaEvent_t device_start = nullptr;
  cudaEvent_t device_stop = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  err = cudaHostAlloc(reinterpret_cast<void **>(&host_arena), host_weight_bytes,
                      cudaHostAllocDefault);
  if (err == cudaSuccess)
    err = cudaHostAlloc(reinterpret_cast<void **>(&host_slots), slots_bytes,
                        cudaHostAllocDefault);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_arena), arena_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_layouts), layout_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_scratch), scratch_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_kv_keys), kv_bytes / 2);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_kv_values), kv_bytes / 2);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_kv_block_table), kv_block_table_bytes);
  if (err == cudaSuccess && linear_gdn_conv_state_bytes != 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&device_linear_gdn_conv_state),
                     linear_gdn_conv_state_bytes);
  }
  if (err == cudaSuccess && linear_gdn_recurrent_state_bytes != 0) {
    err = cudaMalloc(
        reinterpret_cast<void **>(&device_linear_gdn_recurrent_state),
        linear_gdn_recurrent_state_bytes);
  }
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_prompt_tokens), prompt_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_slots), slots_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_step), sizeof(uint32_t));
  if (err == cudaSuccess) err = cudaMalloc(&cublas_workspace, kCublasWorkspaceBytes);
  if (err == cudaSuccess) err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err == cudaSuccess) err = cublas_to_cuda(cublasCreate(&cublas));
  if (err == cudaSuccess) {
    err = configure_cublas(cublas, stream, cublas_workspace,
                           kCublasWorkspaceBytes);
  }
  if (err == cudaSuccess) err = cudaEventCreate(&device_start);
  if (err == cudaSuccess) err = cudaEventCreate(&device_stop);
  if (err != cudaSuccess) {
    fail(out, err);
    if (device_stop != nullptr) cudaEventDestroy(device_stop);
    if (device_start != nullptr) cudaEventDestroy(device_start);
    if (cublas != nullptr) cublasDestroy(cublas);
    if (stream != nullptr) cudaStreamDestroy(stream);
    cudaFree(cublas_workspace);
    cudaFree(device_step);
    cudaFree(device_slots);
    cudaFree(device_prompt_tokens);
    cudaFree(device_linear_gdn_recurrent_state);
    cudaFree(device_linear_gdn_conv_state);
    cudaFree(device_kv_block_table);
    cudaFree(device_kv_values);
    cudaFree(device_kv_keys);
    cudaFree(device_scratch);
    cudaFree(device_layouts);
    cudaFree(device_arena);
    cudaFreeHost(host_slots);
    cudaFreeHost(host_arena);
    return -1;
  }

  memset(host_arena, 0, host_weight_bytes);
  memset(host_slots, 0, slots_bytes);
  const uint64_t embedding_bytes = vocab_size * hidden * sizeof(uint16_t);
  const uint64_t scratch_gap_bytes = hidden * 2 * sizeof(uint16_t);
  if (!descriptor_mode) {
    memcpy(host_arena + arena_layout.embeddings, request->embeddings,
           vocab_size * hidden * sizeof(uint16_t));
    for (uint32_t index = 0; index < request->layer_count; ++index) {
      copy_layer(host_arena, layouts[index], request->layers[index], hidden,
                 attention_hidden, kv_hidden, request->head_dim, intermediate);
    }
    memcpy(host_arena + arena_layout.final_norm, request->final_norm_weight,
           hidden * sizeof(uint16_t));
    memcpy(host_arena + arena_layout.lm_head, request->lm_head,
           vocab_size * hidden * sizeof(uint16_t));
  }

  if (err == cudaSuccess && descriptor_mode) {
    err = copy_weight_descriptors_to_device(
        device_arena, host_arena, host_weight_bytes, request, arena_bytes,
        embedding_bytes, scratch_gap_bytes, stream, out, &setup_sync_calls);
    if (err == cudaSuccess) {
      err = deinterleave_descriptor_query_gate_weights(
          device_arena, layouts, request->hidden, request->heads,
          request->head_dim, stream, &setup_sync_calls);
    }
  } else if (err == cudaSuccess) {
    err = cudaMemcpyAsync(device_arena, host_arena, arena_bytes,
                          cudaMemcpyHostToDevice, stream);
    out->h2d_bytes = arena_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(device_layouts, layouts.data(), layout_bytes,
                          cudaMemcpyHostToDevice, stream);
    out->h2d_bytes += layout_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(device_prompt_tokens, request->prompt_tokens, prompt_bytes,
                          cudaMemcpyHostToDevice, stream);
    out->h2d_bytes += prompt_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(device_slots, 0, slots_bytes, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(device_kv_keys, 0, kv_bytes / 2, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(device_kv_values, 0, kv_bytes / 2, stream);
  }
  if (err == cudaSuccess && linear_gdn_conv_state_bytes != 0) {
    err = cudaMemsetAsync(device_linear_gdn_conv_state, 0,
                          linear_gdn_conv_state_bytes, stream);
  }
  if (err == cudaSuccess && linear_gdn_recurrent_state_bytes != 0) {
    err = cudaMemsetAsync(device_linear_gdn_recurrent_state, 0,
                          linear_gdn_recurrent_state_bytes, stream);
  }
  if (err == cudaSuccess) {
    const uint32_t blocks = (kv_block_count + kDecodeThreads - 1u) / kDecodeThreads;
    hf_init_identity_kv_block_table_kernel<<<blocks, kDecodeThreads, 0, stream>>>(
        device_kv_block_table, kv_block_count);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(device_step, 0, sizeof(uint32_t), stream);
  }
  if (err == cudaSuccess) {
    err = warm_cublas_gemv(cublas, device_arena, arena_layout, request->dtype,
                           device_scratch, stream);
  }
  bool capture_started = false;
  if (err == cudaSuccess) {
    err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    capture_started = err == cudaSuccess;
  }
  if (err == cudaSuccess) {
    hf_decode_sequence_kernel<<<1, kDecodeThreads, 0, stream>>>(
        device_arena, arena_layout, device_layouts, request->layer_count, request->dtype,
        request->hidden, request->heads, request->kv_heads, request->head_dim,
        request->intermediate, 0, device_step, context_steps, device_prompt_tokens,
        request->prompt_token_count, request->rms_eps, request->rope_theta,
        device_scratch, device_kv_keys, device_kv_values, kv_block_count,
        device_kv_block_table, device_slots, device_linear_gdn_conv_state,
        device_linear_gdn_recurrent_state);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    float *device_logits = device_scratch + hidden * 2;
    err = final_head_gemv(cublas, device_arena, arena_layout, request->dtype,
                          request->hidden, request->vocab_size, device_logits);
  }
  if (err == cudaSuccess) {
    float *device_logits = device_scratch + hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        stream, device_step, context_steps, request->has_eos_token,
        request->eos_token, device_logits, request->vocab_size, device_slots,
        request_sampler);
  }
  if (capture_started) {
    cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
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
    out->graph_captures = 1;
  }
  if (err == cudaSuccess) {
    err = cudaGraphInstantiate(&graph_exec, graph, 0);
  }
  if (err == cudaSuccess) {
    err = cudaEventRecord(device_start, stream);
  }
  for (uint32_t step = 0; err == cudaSuccess && step < context_steps; ++step) {
    err = cudaGraphLaunch(graph_exec, stream);
    if (err == cudaSuccess) {
      out->graph_replays += 1;
      out->graph_launches += 1;
      out->kernel_launches += out->graph_nodes == 0 ? 1 : out->graph_nodes;
    }
  }
  if (err == cudaSuccess) {
    err = cudaEventRecord(device_stop, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_slots, device_slots, slots_bytes,
                          cudaMemcpyDeviceToHost, stream);
    out->d2h_bytes = slots_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
    out->sync_calls = setup_sync_calls + 1;
  }
  if (err == cudaSuccess) {
    float device_ms = 0.0f;
    err = cudaEventElapsedTime(&device_ms, device_start, device_stop);
    if (err == cudaSuccess && device_ms > 0.0f) {
      out->device_elapsed_ns = static_cast<uint64_t>(device_ms * 1000000.0f);
      if (out->device_elapsed_ns == 0) {
        out->device_elapsed_ns = 1;
      }
    }
  }

  if (err == cudaSuccess) {
    out->observed_tokens = observed_count(request, host_slots);
    const uint32_t output_start = request->prompt_token_count - 1u;
    for (uint32_t index = 0; index < out->observed_tokens; ++index) {
      request->output_tokens[index] = host_slots[output_start + index].token;
    }
    out->last_token = out->observed_tokens == 0
                          ? 0
                          : request->output_tokens[out->observed_tokens - 1];
    out->observed_token_hash = hash_tokens(request->output_tokens, out->observed_tokens);
    out->resident_weight_bytes = resident_weight_bytes;
    out->resident_kv_bytes = kv_bytes;
    out->kv_tokens = output_start + out->observed_tokens;
    out->device_arena_bytes =
        arena_bytes + layout_bytes + scratch_bytes + kv_bytes +
        kv_block_table_bytes + linear_gdn_conv_state_bytes +
        linear_gdn_recurrent_state_bytes + prompt_bytes + slots_bytes + sizeof(uint32_t) +
        kCublasWorkspaceBytes;
    out->pinned_host_bytes = host_weight_bytes + slots_bytes;
    out->host_causality_edges = 0;
    out->status = out->observed_tokens > 0 ? 0 : -1;
    for (uint32_t index = 0; out->status == 0 && index < out->observed_tokens; ++index) {
      const NervaCudaSyntheticTokenSlot &slot = host_slots[output_start + index];
      if (slot.request_id != kRequestId || slot.sequence_id != kSequenceId ||
          slot.completion != kCompletionDeviceComplete ||
          slot.token_index != output_start + index) {
        out->status = -1;
      }
    }
  } else {
    fail(out, err);
  }

  if (graph_exec != nullptr) {
    cudaGraphExecDestroy(graph_exec);
  }
  if (graph != nullptr) {
    cudaGraphDestroy(graph);
  }
  if (device_stop != nullptr) {
    cudaEventDestroy(device_stop);
  }
  if (device_start != nullptr) {
    cudaEventDestroy(device_start);
  }
  if (cublas != nullptr) {
    cublasDestroy(cublas);
  }
  cudaStreamDestroy(stream);
  cudaFree(cublas_workspace);
  cudaFree(device_step);
  cudaFree(device_slots);
  cudaFree(device_prompt_tokens);
  cudaFree(device_linear_gdn_recurrent_state);
  cudaFree(device_linear_gdn_conv_state);
  cudaFree(device_kv_block_table);
  cudaFree(device_kv_values);
  cudaFree(device_kv_keys);
  cudaFree(device_scratch);
  cudaFree(device_layouts);
  cudaFree(device_arena);
  cudaFreeHost(host_slots);
  cudaFreeHost(host_arena);
  return out->status == 0 ? 0 : -1;
}
