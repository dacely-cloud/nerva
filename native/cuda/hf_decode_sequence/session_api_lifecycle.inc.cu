extern "C" int nerva_cuda_hf_decode_sequence_session_create(
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request,
    NervaCudaHfDecodeSequenceSessionCreateResult *out,
    NervaCudaHfDecodeSequenceSession **session_out) {
  if (out == nullptr || session_out == nullptr) {
    return -1;
  }
  *session_out = nullptr;
  clear_session_create_result(request, out);
  if (request == nullptr) {
    out->failure_stage = kCreateStageInvalidRequest;
    return -1;
  }
  const bool descriptor_mode = has_declared_weight_plan(request);
  if (request->layers == nullptr ||
      (!descriptor_mode &&
       (request->embeddings == nullptr || request->final_norm_weight == nullptr ||
        request->lm_head == nullptr)) ||
      request->layer_count == 0 || request->max_context_tokens == 0 ||
      request->hidden == 0 || request->heads == 0 || request->kv_heads == 0 ||
      request->head_dim == 0 || request->intermediate == 0 ||
      request->vocab_size == 0 || request->dtype > kDTypeBF16 ||
      request->kv_heads > request->heads || request->heads % request->kv_heads != 0 ||
      (request->rope_theta > 0.0f && request->head_dim % 2 != 0)) {
    out->failure_stage = kCreateStageInvalidRequest;
    return -1;
  }
  bool has_dense_mlp_layers = false;
  bool has_query_gate_layers = false;
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    if (!valid_layer(request->layers[index], !descriptor_mode)) {
      out->failure_stage = kCreateStageInvalidRequest;
      return -1;
    }
    const auto &layer = request->layers[index];
    if (layer.w_q_gate != nullptr) {
      has_query_gate_layers = true;
    }
    if (layer.mlp_kind == kMlpKindSparseMoe) {
      if (layer.moe_intermediate > request->intermediate) {
        out->failure_stage = kCreateStageInvalidRequest;
        return -1;
      }
    } else if (layer.mlp_kind == kMlpKindDense) {
      has_dense_mlp_layers = true;
    }
  }
  if (descriptor_mode &&
      (request->planned_weight_blocks == 0 || request->planned_weight_bytes == 0 ||
       request->planned_weight_descriptors == nullptr ||
       request->planned_weight_descriptor_count != request->planned_weight_blocks ||
       request->planned_weight_descriptor_hash == 0)) {
    out->failure_stage = kCreateStageInvalidRequest;
    return -1;
  }
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
  cudaDeviceProp device_props{};
  cudaError_t props_err = cudaGetDeviceProperties(&device_props, 0);
  if (props_err != cudaSuccess) {
    device_props.warpSize = 32;
    device_props.major = 0;
    cudaGetLastError();
  }
  size_t device_free_before_alloc = 0;
  size_t device_total_before_alloc = 0;
  cudaError_t mem_info_err =
      cudaMemGetInfo(&device_free_before_alloc, &device_total_before_alloc);
  if (mem_info_err != cudaSuccess) {
    device_free_before_alloc = 0;
    device_total_before_alloc = 0;
  }

  auto *session = new (std::nothrow) NervaCudaHfDecodeSequenceSession();
  if (session == nullptr) {
    out->cuda_error = static_cast<int32_t>(cudaErrorMemoryAllocation);
    out->failure_stage = kCreateStageSessionAlloc;
    return -1;
  }
  const uint64_t hidden = request->hidden;
  const uint64_t attention_hidden = request->heads * request->head_dim;
  const uint64_t kv_hidden = request->kv_heads * request->head_dim;
  const uint64_t intermediate = request->intermediate;
  const uint64_t vocab_size = request->vocab_size;
  const bool request_has_deepseek_layers =
      has_deepseek_layers(request->layers, request->layer_count);
  std::vector<SequenceLayerLayout> layouts(request->layer_count);
  uint64_t elements = 0;
  session->arena_layout.embeddings = push(elements, vocab_size * hidden);
  session->arena_layout.input = push(elements, hidden);
  session->arena_layout.scratch = push(elements, hidden);
  session->arena_layout.deepseek_hc_head_base = kMissingOffset;
  session->arena_layout.deepseek_hc_head_fn = kMissingOffset;
  session->arena_layout.deepseek_hc_head_scale = kMissingOffset;
  pack_deepseek_static(session->arena_layout, elements, request->layers,
                       request->layer_count, hidden);
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    pack_layer(layouts[index], elements, request->layers[index], hidden,
               attention_hidden, kv_hidden, request->head_dim, intermediate,
               vocab_size);
  }
  uint64_t linear_gdn_conv_state_elements = 0;
  uint64_t linear_gdn_recurrent_state_elements = 0;
  assign_linear_gdn_state_offsets(layouts, &linear_gdn_conv_state_elements,
                                  &linear_gdn_recurrent_state_elements);
  session->arena_layout.final_norm = push(elements, hidden);
  session->arena_layout.lm_head = push(elements, vocab_size * hidden);
  session->arena_bytes = elements * sizeof(uint16_t);
  session->resident_weight_bytes = session->arena_bytes - hidden * 2 * sizeof(uint16_t);
  if (request->planned_weight_blocks != 0 &&
      request->planned_weight_bytes != session->resident_weight_bytes) {
    out->failure_stage = kCreateStageInvalidRequest;
    delete session;
    return -1;
  }
  if (!validate_weight_descriptors(request, session->resident_weight_bytes, out)) {
    out->failure_stage = kCreateStageInvalidRequest;
    delete session;
    return -1;
  }
  if (has_unsupported_deepseek_layers(request->layers, request->layer_count)) {
    out->cuda_error = static_cast<int32_t>(cudaErrorNotSupported);
    out->failure_stage = kCreateStageInvalidRequest;
    delete session;
    return -1;
  }

  const uint64_t attention_workspace_rows =
      max_attention_workspace_rows(layouts, attention_hidden);
  const uint64_t kv_cache_width = max_kv_cache_width(layouts, kv_hidden);
  const uint64_t block_scratch = max_layer_scratch_elements(
      layouts, hidden, attention_hidden, kv_hidden, intermediate);
  const uint64_t final_scratch = hidden * 2 + vocab_size;
  const uint64_t scratch_elements =
      block_scratch > final_scratch ? block_scratch : final_scratch;
  const uint64_t projection_input_elements =
      std::max<uint64_t>(hidden,
                         std::max<uint64_t>(intermediate,
                                            attention_workspace_rows));
  const uint64_t prefill_qkv_rows = attention_hidden + kv_hidden * 2;
  const uint64_t prefill_gate_up_rows = intermediate * 2;
  const bool pack_cublas =
      !request_has_deepseek_layers &&
      should_pack_cublas_weights(request->hidden, attention_hidden);
  const uint64_t prefill_q_gate_rows =
      (has_query_gate_layers && pack_cublas) ? attention_hidden : 0;
  const PackedProjectionShape packed_shape = packed_projection_shape(
      hidden, attention_hidden, kv_hidden, intermediate);
  const uint64_t projection_batch_output_rows =
      std::max<uint64_t>(vocab_size,
                         std::max<uint64_t>(
                             static_cast<uint64_t>(packed_shape.qkv_rows),
                             std::max<uint64_t>(
                                 static_cast<uint64_t>(packed_shape.gate_up_rows),
                                 hidden)));
  session->dtype = request->dtype;
  session->hidden = request->hidden;
  session->heads = request->heads;
  session->kv_heads = request->kv_heads;
  session->head_dim = request->head_dim;
  session->head_threads = tuned_head_threads(request->head_dim, device_props);
  session->intermediate = request->intermediate;
  session->vocab_size = request->vocab_size;
  session->layer_count = request->layer_count;
  session->max_context_tokens = request->max_context_tokens;
  session->kv_block_count =
      ceil_div_u32(request->max_context_tokens, kKvCacheBlockTokens);
  session->kv_token_capacity = session->kv_block_count * kKvCacheBlockTokens;
  session->detailed_profile = request->detailed_profile == 0 ? 0u : 1u;
  session->experimental_rt_decode_requested =
      request->experimental_rt_decode == 0 ? 0u : 1u;
  session->experimental_rt_decode_enabled = 0;
  session->experimental_rt_mode =
      normalize_experimental_rt_mode(request->experimental_rt_mode);
  session->experimental_rt_page_tokens = request->experimental_rt_page_tokens;
  session->experimental_rt_pages = request->experimental_rt_pages;
  session->experimental_rt_local_window_tokens =
      request->experimental_rt_local_window_tokens;
  session->experimental_rt_sink_tokens = request->experimental_rt_sink_tokens;
  session->experimental_rt_kv_descriptor_selector =
      experimental_rt_kv_descriptor_selector_enabled();
  session->experimental_rt_query_descriptor_selector =
      session->experimental_rt_kv_descriptor_selector != 0
          ? 1u
          : experimental_rt_query_descriptor_selector_enabled();
  session->experimental_rt_query_key_selector =
      session->experimental_rt_query_descriptor_selector == 0
          ? experimental_rt_query_key_selector_enabled()
          : 0u;
  session->experimental_rt_query_key_fused_selector =
      session->experimental_rt_query_key_selector == 0
          ? 0u
          : experimental_rt_query_key_fused_selector_enabled();
  session->experimental_prefill_local_window_tokens =
      experimental_prefill_local_window_tokens();
  session->rms_eps = request->rms_eps;
  session->rope_theta = request->rope_theta;
  session->layout_bytes = layouts.size() * sizeof(SequenceLayerLayout);
  session->scratch_bytes = scratch_elements * sizeof(float);
  session->projection_input_bytes = projection_input_elements * sizeof(uint16_t);
  session->projection_batch_input_bytes =
      projection_input_elements *
      static_cast<uint64_t>(kProjectionBatchWorkspaceTokens) * sizeof(uint16_t);
  session->projection_batch_output_bytes =
      projection_batch_output_rows *
      static_cast<uint64_t>(kProjectionBatchWorkspaceTokens) * sizeof(float);
  session->prefill_hidden_bytes =
      static_cast<uint64_t>(request->max_context_tokens) * hidden *
      sizeof(uint16_t);
  session->decode_attention_max_chunks =
      ceil_div_u32(request->max_context_tokens, kDecodeAttentionChunkTokens);
  session->decode_attention_values_bytes =
      attention_workspace_rows * session->decode_attention_max_chunks *
      sizeof(float);
  session->decode_attention_stats_bytes =
      static_cast<uint64_t>(request->heads) *
      session->decode_attention_max_chunks * sizeof(float);
  session->decode_q_bytes = attention_workspace_rows * sizeof(uint16_t);
  session->decode_seq_len_bytes = sizeof(int32_t) * 2u;
  session->linear_gdn_conv_state_bytes =
      linear_gdn_conv_state_elements * sizeof(float);
  session->linear_gdn_recurrent_state_bytes =
      linear_gdn_recurrent_state_elements * sizeof(float);
  if (pack_cublas) {
    session->packed_qkv_bytes =
        packed_shape.qkv_elements_per_layer * request->layer_count *
        sizeof(uint16_t);
    if (has_dense_mlp_layers) {
      session->packed_gate_up_bytes =
          packed_shape.gate_up_elements_per_layer * request->layer_count *
          sizeof(uint16_t);
    }
  }
  session->kv_bytes =
      request->layer_count * static_cast<uint64_t>(session->kv_token_capacity) *
      kv_cache_width * sizeof(uint16_t) * 2;
  session->deepseek_v32_mla_kv_bytes =
      accumulate_deepseek_v32_mla_kv_bytes(layouts,
                                           request->max_context_tokens);
  accumulate_deepseek_v4_compressed_runtime_bytes(
      layouts, request->max_context_tokens, session->kv_token_capacity,
      &session->deepseek_swa_kv_bytes,
      &session->deepseek_compressor_state_bytes,
      &session->deepseek_compressed_kv_bytes,
      &session->deepseek_indexer_state_bytes,
      &session->deepseek_indexer_kv_bytes);
  accumulate_deepseek_v4_mhc_runtime_bytes(
      layouts, request->max_context_tokens, request->hidden,
      &session->deepseek_mhc_residual_bytes,
      &session->deepseek_mhc_post_mix_bytes,
      &session->deepseek_mhc_comb_mix_bytes);
  bool has_deepseek_layout = false;
  for (const SequenceLayerLayout &layout : layouts) {
    if (layout.deepseek_mode != 0) {
      has_deepseek_layout = true;
      break;
    }
  }
  if (has_deepseek_layout || session->deepseek_compressor_state_bytes != 0 ||
      session->deepseek_v32_mla_kv_bytes != 0 ||
      session->deepseek_swa_kv_bytes != 0 ||
      session->deepseek_compressed_kv_bytes != 0 ||
      session->deepseek_indexer_state_bytes != 0 ||
      session->deepseek_indexer_kv_bytes != 0 ||
      session->deepseek_mhc_residual_bytes != 0 ||
      session->deepseek_mhc_post_mix_bytes != 0 ||
      session->deepseek_mhc_comb_mix_bytes != 0) {
    session->deepseek_runtime_counters_bytes =
        kDeepSeekRuntimeCounterCount * sizeof(uint64_t);
  }
  session->kv_block_table_bytes =
      static_cast<uint64_t>(session->kv_block_count) * sizeof(uint32_t);
  session->slots_bytes =
      static_cast<uint64_t>(request->max_context_tokens) *
      sizeof(NervaCudaSyntheticTokenSlot);
  session->prompt_bytes =
      static_cast<uint64_t>(request->max_context_tokens) * sizeof(uint32_t);
  const uint64_t fixed_device_bytes =
      session_fixed_footprint_without_prefill_chunk(session);
  const uint32_t prefill_chunk = tune_prefill_chunk_tokens(
      request->max_context_tokens, fixed_device_bytes, projection_input_elements,
      prefill_qkv_rows, attention_hidden, hidden, prefill_q_gate_rows,
      prefill_gate_up_rows, intermediate,
      static_cast<uint64_t>(device_free_before_alloc));
  session->prefill_chunk_tokens = prefill_chunk;
  session->prefill_norm_bytes =
      projection_input_elements * static_cast<uint64_t>(prefill_chunk) *
      sizeof(uint16_t);
  session->prefill_qkv_bytes =
      prefill_qkv_rows * static_cast<uint64_t>(prefill_chunk) * sizeof(float);
  session->prefill_qkv_encoded_bytes =
      prefill_qkv_rows * static_cast<uint64_t>(prefill_chunk) *
      sizeof(uint16_t);
  session->prefill_attn_bytes =
      attention_hidden * static_cast<uint64_t>(prefill_chunk) * sizeof(uint16_t);
  session->prefill_o_bytes =
      hidden * static_cast<uint64_t>(prefill_chunk) * sizeof(float);
  session->prefill_q_gate_bytes =
      prefill_q_gate_rows * static_cast<uint64_t>(prefill_chunk) *
      sizeof(float);
  session->prefill_gate_up_bytes =
      prefill_gate_up_rows * static_cast<uint64_t>(prefill_chunk) * sizeof(float);
  session->prefill_ff_bytes =
      intermediate * static_cast<uint64_t>(prefill_chunk) * sizeof(uint16_t);
  session->prefill_down_bytes =
      hidden * static_cast<uint64_t>(prefill_chunk) * sizeof(float);
  session->planned_weight_blocks = request->planned_weight_blocks;
  session->planned_gpu_resident_blocks = request->planned_gpu_resident_blocks;
  session->planned_gpu_staged_blocks = request->planned_gpu_staged_blocks;
  session->planned_weight_bytes = request->planned_weight_bytes;
  session->planned_gpu_resident_weight_bytes =
      request->planned_gpu_resident_weight_bytes;
  session->planned_gpu_staged_weight_bytes =
      request->planned_gpu_staged_weight_bytes;
  session->planned_weight_descriptor_count =
      request->planned_weight_descriptor_count;
  session->planned_weight_descriptor_hash = request->planned_weight_descriptor_hash;
  session->host_layouts = layouts;

  uint16_t *host_arena = nullptr;
  const uint64_t host_weight_bytes =
      descriptor_mode
          ? pinned_weight_staging_bytes(request, session->resident_weight_bytes)
          : session->arena_bytes;
  out->prefill_chunk_tokens = session->prefill_chunk_tokens;
  out->head_threads = session->head_threads;
  out->resident_weight_bytes = session->resident_weight_bytes;
  out->deepseek_v4_swa_kv_bytes = session->deepseek_swa_kv_bytes;
  out->deepseek_mhc_residual_bytes = session->deepseek_mhc_residual_bytes;
  out->deepseek_mhc_post_mix_bytes = session->deepseek_mhc_post_mix_bytes;
  out->deepseek_mhc_comb_mix_bytes = session->deepseek_mhc_comb_mix_bytes;
  out->resident_kv_bytes = session_resident_kv_bytes(session);
  out->device_arena_bytes = session_device_footprint(session);
  out->pinned_host_bytes = session->slots_bytes + host_weight_bytes;
  uint64_t setup_sync_calls = 0;
  int32_t failure_stage = kCreateStageHostWeightAlloc;
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_arena), host_weight_bytes,
                      cudaHostAllocDefault);
  if (err == cudaSuccess) {
    failure_stage = kCreateStageHostSlotsAlloc;
    err = cudaHostAlloc(reinterpret_cast<void **>(&session->host_slots),
                        session->slots_bytes, cudaHostAllocDefault);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDeviceArenaAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_arena),
                     session->arena_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageDeviceLayoutsAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_layouts),
                     session->layout_bytes);
  }
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
  if (err == cudaSuccess) {
    failure_stage = kCreateStagePrefillHiddenAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_hidden_a),
                     session->prefill_hidden_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_hidden_b),
                     session->prefill_hidden_bytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStagePrefillChunkAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_norm),
                     session->prefill_norm_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_qkv),
                     session->prefill_qkv_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_prefill_qkv_encoded),
        session->prefill_qkv_encoded_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_attn),
                     session->prefill_attn_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_o),
                     session->prefill_o_bytes);
  }
  if (err == cudaSuccess && session->prefill_q_gate_bytes != 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_q_gate),
                     session->prefill_q_gate_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_gate_up),
                     session->prefill_gate_up_bytes);
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_prefill_ff),
                     session->prefill_ff_bytes);
  }
  if (err == cudaSuccess) {
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
  if (err == cudaSuccess && session->linear_gdn_conv_state_bytes != 0) {
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_linear_gdn_conv_state),
        session->linear_gdn_conv_state_bytes);
  }
  if (err == cudaSuccess && session->linear_gdn_recurrent_state_bytes != 0) {
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_linear_gdn_recurrent_state),
        session->linear_gdn_recurrent_state_bytes);
  }
  if (err == cudaSuccess && session->packed_qkv_bytes != 0) {
    failure_stage = kCreateStagePackedQkvAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_qkv_packed),
                     session->packed_qkv_bytes);
  }
  if (err == cudaSuccess && session->packed_gate_up_bytes != 0) {
    failure_stage = kCreateStagePackedGateUpAlloc;
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_gate_up_packed),
                     session->packed_gate_up_bytes);
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
  if (err == cudaSuccess) {
    failure_stage = kCreateStageCublasWorkspaceAlloc;
    err = cudaMalloc(&session->cublas_workspace, kCublasWorkspaceBytes);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageStreamCreate;
    err = cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageCublasCreate;
    err = cublas_to_cuda(cublasCreate(&session->cublas));
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageCublasLtCreate;
    err = cublas_to_cuda(cublasLtCreate(&session->cublas_lt));
  }
#if NERVA_HAVE_CUDNN_FRONTEND
  if (err == cudaSuccess) {
    failure_stage = kCreateStageCublasConfigure;
    err = cudnn_to_cuda(cudnnCreate(&session->cudnn));
  }
#endif
  if (err == cudaSuccess) {
    failure_stage = kCreateStageCublasConfigure;
    err = configure_cublas(session->cublas, session->stream,
                           session->cublas_workspace,
                           kCublasWorkspaceBytes);
  }
#if NERVA_HAVE_CUDNN_FRONTEND
  if (err == cudaSuccess) {
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
  if (err != cudaSuccess) {
    fail(out, err, failure_stage);
    cudaFreeHost(host_arena);
    free_session_fields(session);
    delete session;
    return -1;
  }

  memset(host_arena, 0, host_weight_bytes);
  const uint64_t embedding_bytes = vocab_size * hidden * sizeof(uint16_t);
  const uint64_t scratch_gap_bytes = hidden * 2 * sizeof(uint16_t);
  if (!descriptor_mode) {
    memcpy(host_arena + session->arena_layout.embeddings, request->embeddings,
           vocab_size * hidden * sizeof(uint16_t));
    for (uint32_t index = 0; index < request->layer_count; ++index) {
      copy_layer(host_arena, layouts[index], request->layers[index], hidden,
                 attention_hidden, kv_hidden, request->head_dim, intermediate);
    }
    memcpy(host_arena + session->arena_layout.final_norm,
           request->final_norm_weight, hidden * sizeof(uint16_t));
    memcpy(host_arena + session->arena_layout.lm_head, request->lm_head,
           vocab_size * hidden * sizeof(uint16_t));
  }
  if (err == cudaSuccess && descriptor_mode) {
    failure_stage = kCreateStageDescriptorCopy;
    err = copy_weight_descriptors_to_device(
        session->device_arena, host_arena, host_weight_bytes, request,
        session->arena_bytes, embedding_bytes, scratch_gap_bytes,
        session->stream, out, &setup_sync_calls);
    if (err == cudaSuccess) {
      err = deinterleave_descriptor_query_gate_weights(
          session->device_arena, session->host_layouts, session->hidden,
          session->heads, session->head_dim, session->stream,
          &setup_sync_calls);
    }
  } else if (err == cudaSuccess) {
    failure_stage = kCreateStageDescriptorCopy;
    err = cudaMemcpyAsync(session->device_arena, host_arena, session->arena_bytes,
                          cudaMemcpyHostToDevice, session->stream);
    out->h2d_bytes = session->arena_bytes;
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageLayoutCopy;
    err = cudaMemcpyAsync(session->device_layouts, layouts.data(),
                          session->layout_bytes, cudaMemcpyHostToDevice,
                          session->stream);
    out->h2d_bytes += session->layout_bytes;
  }
  if (err == cudaSuccess && session->linear_gdn_conv_state_bytes != 0) {
    err = cudaMemsetAsync(session->device_linear_gdn_conv_state, 0,
                          session->linear_gdn_conv_state_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->linear_gdn_recurrent_state_bytes != 0) {
    err = cudaMemsetAsync(session->device_linear_gdn_recurrent_state, 0,
                          session->linear_gdn_recurrent_state_bytes,
                          session->stream);
  }
  if (err == cudaSuccess) {
    err = reset_deepseek_session_device_state(session);
  }
  if (err == cudaSuccess) {
    const uint32_t blocks =
        (session->kv_block_count + kDecodeThreads - 1u) / kDecodeThreads;
    hf_init_identity_kv_block_table_kernel<<<blocks, kDecodeThreads, 0,
                                             session->stream>>>(
        session->device_kv_block_table, session->kv_block_count);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStagePackReplicas;
    err = pack_session_weight_replicas(session);
  }
  if (err == cudaSuccess && use_cublas_layer_path(session)) {
    failure_stage = kCreateStageProjectionPlanAutotune;
    err = autotune_session_lt_gemv_plans(session);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageWarmCublas;
    err = warm_cublas_gemv(session->cublas, session->device_arena,
                           session->arena_layout, session->dtype,
                           session->device_scratch, session->stream);
  }
  if (err == cudaSuccess) {
    failure_stage = kCreateStageSetupSynchronize;
    err = cudaStreamSynchronize(session->stream);
  }
  cudaFreeHost(host_arena);
  if (err != cudaSuccess) {
    fail(out, err, failure_stage);
    free_session_fields(session);
    delete session;
    return -1;
  }
  auto shared_weights = std::make_shared<SessionSharedWeights>();
  shared_weights->device_arena = session->device_arena;
  shared_weights->device_layouts = session->device_layouts;
  shared_weights->device_qkv_packed = session->device_qkv_packed;
  shared_weights->device_gate_up_packed = session->device_gate_up_packed;
  session->shared_weights = shared_weights;
  session->h2d_bytes = out->h2d_bytes;
  session->load_staging_bytes = host_weight_bytes;
  session->setup_sync_calls = setup_sync_calls;
  session->projection_batch_own_stream_synchronized = 1;
  session->descriptor_gpu_resident_h2d_bytes =
      out->descriptor_gpu_resident_h2d_bytes;
  session->descriptor_gpu_staged_h2d_bytes =
      out->descriptor_gpu_staged_h2d_bytes;
  fill_create_result(session, out);
  *session_out = session;
  return 0;
}

extern "C" int nerva_cuda_hf_decode_sequence_session_run(
    const NervaCudaHfDecodeSequenceSessionRunRequest *request,
    NervaCudaHfDecodeSequenceResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->prompt_tokens == nullptr || request->output_tokens == nullptr ||
      request->steps == 0 || request->prompt_token_count == 0 ||
      request->output_token_capacity < request->steps ||
      request->prompt_tokens[request->prompt_token_count - 1u] !=
          request->seed_token ||
      !std::isfinite(request->sampler.temperature) ||
      request->sampler.temperature < 0.0f ||
      !std::isfinite(request->sampler.top_p) ||
      request->sampler.top_p <= 0.0f || request->sampler.top_p > 1.0f ||
      request->prompt_token_count > UINT32_MAX - request->steps + 1u) {
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  session->active_sampler = normalize_hf_decode_sampler_config(request->sampler);
  const uint32_t context_steps =
      request->prompt_token_count + request->steps - 1u;
  if (context_steps > session->max_context_tokens) {
    return -1;
  }
  if (validate_deepseek_v4_compressed_context(session, context_steps) !=
      cudaSuccess) {
    out->device_count = 1;
    out->cuda_error = static_cast<int32_t>(cudaErrorNotSupported);
    return -1;
  }
  for (uint32_t index = 0; index < request->prompt_token_count; ++index) {
    if (request->prompt_tokens[index] >= session->vocab_size) {
      return -1;
    }
  }
  out->device_count = 1;
  out->dtype = session->dtype;
  out->hidden = session->hidden;
  out->heads = session->heads;
  out->kv_heads = session->kv_heads;
  out->head_dim = session->head_dim;
  out->intermediate = session->intermediate;
  out->vocab_size = session->vocab_size;
  out->layer_count = session->layer_count;
  out->steps = request->steps;
  out->seed_token = request->seed_token;
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

  cudaError_t err = cudaMemsetAsync(session->device_slots, 0,
                                    session->slots_bytes, session->stream);
  if (err == cudaSuccess)
    err = cudaMemsetAsync(session->device_step, 0, sizeof(uint32_t),
                          session->stream);
  if (err == cudaSuccess && session->linear_gdn_conv_state_bytes != 0) {
    err = cudaMemsetAsync(session->device_linear_gdn_conv_state, 0,
                          session->linear_gdn_conv_state_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->linear_gdn_recurrent_state_bytes != 0) {
    err = cudaMemsetAsync(session->device_linear_gdn_recurrent_state, 0,
                          session->linear_gdn_recurrent_state_bytes,
                          session->stream);
  }
  if (err == cudaSuccess) {
    err = reset_deepseek_session_device_state(session);
  }
  const uint64_t prompt_bytes =
      static_cast<uint64_t>(request->prompt_token_count) * sizeof(uint32_t);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(session->device_prompt_tokens, request->prompt_tokens,
                          prompt_bytes, cudaMemcpyHostToDevice, session->stream);
    out->h2d_bytes = prompt_bytes;
  }

  const bool graph_hit = err == cudaSuccess &&
                         session_graph_matches(session, context_steps,
                                               request->prompt_token_count,
                                               request->has_eos_token,
                                               request->eos_token, 0,
                                               normalize_hf_decode_sampler_config(request->sampler));
  if (graph_hit) {
    out->graph_nodes = session->cached_graph_nodes;
    out->graph_cache_hits = 1;
  }
  if (err == cudaSuccess && !graph_hit) {
    reset_session_graph(session);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool capture_started = false;
    err = cudaStreamBeginCapture(session->stream, cudaStreamCaptureModeGlobal);
    capture_started = err == cudaSuccess;
    if (err == cudaSuccess) {
      err = use_layer_decode_path(session)
                ? launch_cublas_layer_session_step(
                      session, context_steps, request->prompt_token_count,
                      request->has_eos_token, request->eos_token, 0)
                : launch_monolithic_session_step(
                      session, context_steps, request->prompt_token_count,
                      request->has_eos_token, request->eos_token);
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
    if (err == cudaSuccess) err = cudaGraphInstantiate(&graph_exec, graph, 0);
    if (err == cudaSuccess) {
      session->cached_graph = graph;
      session->cached_graph_exec = graph_exec;
      session->cached_context_steps = context_steps;
      session->cached_prompt_token_count = request->prompt_token_count;
      session->cached_has_eos_token = request->has_eos_token;
      session->cached_eos_token = request->eos_token;
      session->cached_attention_chunks = 0;
      session->cached_experimental_rt_sparse_attention_active = 0;
      session->cached_sampler = session->active_sampler;
      session->cached_graph_nodes = out->graph_nodes;
      out->graph_captures = 1;
      graph = nullptr;
      graph_exec = nullptr;
    }
    if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
    if (graph != nullptr) cudaGraphDestroy(graph);
  }
  if (err == cudaSuccess) err = cudaEventRecord(session->device_start, session->stream);
  for (uint32_t step = 0; err == cudaSuccess && step < context_steps; ++step) {
    uint32_t rt_launched = 0;
    err = launch_experimental_rt_selector(
        session, request->prompt_token_count + step, 0, &rt_launched);
    if (err == cudaSuccess && rt_launched != 0) {
      out->kernel_launches += 1;
    }
    if (err != cudaSuccess) {
      break;
    }
    err = cudaGraphLaunch(session->cached_graph_exec, session->stream);
    if (err == cudaSuccess) {
      out->graph_replays += 1;
      out->graph_launches += 1;
      out->kernel_launches += out->graph_nodes == 0 ? 1 : out->graph_nodes;
      if (session->cached_experimental_rt_sparse_attention_active != 0 &&
          session->experimental_rt_query_descriptor_selector != 0) {
        out->experimental_rt_selector_launches += session->layer_count;
      }
    }
  }
  if (err == cudaSuccess) err = cudaEventRecord(session->device_stop, session->stream);
  const uint64_t slots_bytes =
      static_cast<uint64_t>(context_steps) * sizeof(NervaCudaSyntheticTokenSlot);
  uint64_t deepseek_runtime_counters[kDeepSeekRuntimeCounterCount] = {};
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(session->host_slots, session->device_slots, slots_bytes,
                          cudaMemcpyDeviceToHost, session->stream);
    out->d2h_bytes = slots_bytes;
  }
  if (err == cudaSuccess && session->deepseek_runtime_counters_bytes != 0) {
    err = cudaMemcpyAsync(deepseek_runtime_counters,
                          session->device_deepseek_runtime_counters,
                          session->deepseek_runtime_counters_bytes,
                          cudaMemcpyDeviceToHost, session->stream);
    out->d2h_bytes += session->deepseek_runtime_counters_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(session->stream);
    out->sync_calls += 1;
  }
  if (err == cudaSuccess) {
    fill_deepseek_runtime_counter_result(out, deepseek_runtime_counters);
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
  if (err == cudaSuccess) {
    out->observed_tokens =
        observed_count_for(request->steps, request->prompt_token_count,
                           request->has_eos_token, request->eos_token,
                           session->host_slots);
    const uint32_t output_start = request->prompt_token_count - 1u;
    for (uint32_t index = 0; index < out->observed_tokens; ++index) {
      request->output_tokens[index] =
          session->host_slots[output_start + index].token;
    }
    out->last_token = out->observed_tokens == 0
                          ? 0
                          : request->output_tokens[out->observed_tokens - 1];
    out->observed_token_hash =
        hash_tokens(request->output_tokens, out->observed_tokens);
    out->resident_kv_bytes = session_resident_kv_bytes(session);
    out->kv_tokens = output_start + out->observed_tokens;
    out->device_arena_bytes = session_device_footprint(session);
    out->pinned_host_bytes = session->slots_bytes;
    out->host_causality_edges = 0;
    out->status = out->observed_tokens > 0 ? 0 : -1;
    for (uint32_t index = 0; out->status == 0 && index < out->observed_tokens;
         ++index) {
      const NervaCudaSyntheticTokenSlot &slot =
          session->host_slots[output_start + index];
      if (slot.request_id != kRequestId || slot.sequence_id != kSequenceId ||
          slot.completion != kCompletionDeviceComplete ||
          slot.token_index != output_start + index) {
        out->status = -1;
      }
    }
  } else {
    fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

#include "deepseek/session_api_snapshots.inc.cu"

extern "C" int nerva_cuda_hf_decode_sequence_session_start(
    const NervaCudaHfDecodeSequenceSessionStartRequest *request,
    NervaCudaHfDecodeSequenceResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->prompt_tokens == nullptr || request->prompt_token_count == 0 ||
      !std::isfinite(request->sampler.temperature) ||
      request->sampler.temperature < 0.0f ||
      !std::isfinite(request->sampler.top_p) ||
      request->sampler.top_p <= 0.0f || request->sampler.top_p > 1.0f) {
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  if (request->prompt_token_count > session->max_context_tokens) {
    return -1;
  }
  if (validate_deepseek_v4_compressed_context(
          session, request->prompt_token_count) != cudaSuccess) {
    out->device_count = 1;
    out->cuda_error = static_cast<int32_t>(cudaErrorNotSupported);
    return -1;
  }
  for (uint32_t index = 0; index < request->prompt_token_count; ++index) {
    if (request->prompt_tokens[index] >= session->vocab_size) {
      return -1;
    }
  }

  fill_session_result_header(
      session, out, 0, request->prompt_tokens[request->prompt_token_count - 1u]);
  session->active_sampler = normalize_hf_decode_sampler_config(request->sampler);
  session->pending_prefill_available = 0;
  session->projection_batch_own_stream_synchronized = 0;
  session->experimental_rt_selector_cache_valid = 0;
  cudaError_t err = cudaMemsetAsync(session->device_slots, 0,
                                    session->slots_bytes, session->stream);
  if (err == cudaSuccess)
    err = cudaMemsetAsync(session->device_step, 0, sizeof(uint32_t),
                          session->stream);
  if (err == cudaSuccess && session->linear_gdn_conv_state_bytes != 0) {
    err = cudaMemsetAsync(session->device_linear_gdn_conv_state, 0,
                          session->linear_gdn_conv_state_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->linear_gdn_recurrent_state_bytes != 0) {
    err = cudaMemsetAsync(session->device_linear_gdn_recurrent_state, 0,
                          session->linear_gdn_recurrent_state_bytes,
                          session->stream);
  }
  if (err == cudaSuccess) {
    err = reset_deepseek_session_device_state(session);
  }
  const uint64_t prompt_bytes =
      static_cast<uint64_t>(request->prompt_token_count) * sizeof(uint32_t);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(session->device_prompt_tokens, request->prompt_tokens,
                          prompt_bytes, cudaMemcpyHostToDevice, session->stream);
    out->h2d_bytes = prompt_bytes;
  }
  if (err == cudaSuccess) {
    err = use_cublas_prefill_path(session)
              ? launch_cublas_session_prefill(
                    session, request->prompt_token_count,
                    request->has_eos_token, request->eos_token, out)
              : launch_serial_session_prefill(
                    session, request->prompt_token_count,
                    request->has_eos_token, request->eos_token, out);
  }
  if (err == cudaSuccess) {
    err = initialize_experimental_rt_kv_descriptor_selector_after_prefill(
        session, request->prompt_token_count);
  }
  if (err == cudaSuccess) {
    stash_prefill_metrics(session, out);
    session->active_prompt_token_count = request->prompt_token_count;
    session->active_has_eos_token = request->has_eos_token;
    session->active_eos_token = request->eos_token;
    session->active_seed_token = request->prompt_tokens[request->prompt_token_count - 1u];
    session->active_observed_tokens = 0;
    session->active_cursor = request->prompt_token_count;
    session->active_started = true;
    session->active_finished = false;
    session->projection_batch_own_stream_synchronized = 1;
    out->resident_kv_bytes = session_resident_kv_bytes(session);
    out->kv_tokens = request->prompt_token_count;
    out->device_arena_bytes = session_device_footprint(session);
    out->pinned_host_bytes = session->slots_bytes;
    out->status = 0;
  } else {
    fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_hf_decode_sequence_session_advance(
    const NervaCudaHfDecodeSequenceSessionAdvanceRequest *request,
    NervaCudaHfDecodeSequenceResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->output_tokens == nullptr || request->steps == 0 ||
      request->output_token_capacity < request->steps) {
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  if (!session->active_started || session->active_finished ||
      session->active_prompt_token_count == 0) {
    return -1;
  }
  session->projection_batch_own_stream_synchronized = 0;
  const uint32_t prompt_count = session->active_prompt_token_count;
  const uint32_t slot_start = prompt_count - 1u + session->active_observed_tokens;
  const uint32_t target_cursor =
      prompt_count + session->active_observed_tokens + request->steps - 1u;
  if (target_cursor > session->max_context_tokens ||
      target_cursor < session->active_cursor) {
    return -1;
  }
  if (validate_deepseek_v4_compressed_context(session, target_cursor) !=
      cudaSuccess) {
    out->device_count = 1;
    out->cuda_error = static_cast<int32_t>(cudaErrorNotSupported);
    return -1;
  }
  const uint32_t run_count = target_cursor - session->active_cursor;
  const uint32_t seed_token =
      session->active_observed_tokens == 0
          ? session->active_seed_token
          : session->host_slots[slot_start - 1u].token;
  fill_session_result_header(session, out, request->steps, seed_token);

  cudaError_t err = reset_deepseek_runtime_counters(session);
  if (run_count != 0 && projection_batch_session_ready(session) &&
      !use_cublas_layer_path(session)) {
    err = ensure_session_cublas_resources(session);
  }
  if (run_count != 0) {
    const uint32_t dense_attention_chunks =
        decode_attention_chunks_for_cursor(session, target_cursor);
    session->experimental_rt_sparse_attention_active =
        experimental_rt_sparse_available(session, dense_attention_chunks) ? 1u : 0u;
    const uint32_t attention_chunks =
        experimental_rt_attention_chunks_for(session, dense_attention_chunks);
    out->experimental_rt_sparse_attention_active =
        session->experimental_rt_sparse_attention_active;
    out->experimental_rt_dense_attention_chunks = dense_attention_chunks;
    out->experimental_rt_attention_chunks = attention_chunks;
    if (err == cudaSuccess) {
      err = ensure_session_graph(session, session->max_context_tokens, prompt_count,
                                 session->active_has_eos_token,
                                 session->active_eos_token, attention_chunks,
                                 session->active_cursor, out);
    }
  }
  if (err == cudaSuccess && run_count != 0)
    err = cudaEventRecord(session->device_start, session->stream);
  for (uint32_t step = 0; err == cudaSuccess && step < run_count; ++step) {
    const uint32_t current_position = session->active_cursor + step;
    const uint32_t dense_attention_chunks =
        decode_attention_chunks_for_cursor(session, current_position);
    uint32_t rt_launched = 0;
    err = launch_experimental_rt_selector(session, current_position,
                                          dense_attention_chunks, &rt_launched);
    if (err == cudaSuccess && rt_launched != 0) {
      out->kernel_launches += 1;
      out->experimental_rt_selector_launches += 1;
    }
    if (err != cudaSuccess) {
      break;
    }
    err = cudaGraphLaunch(session->cached_graph_exec, session->stream);
    if (err == cudaSuccess) {
      out->graph_replays += 1;
      out->graph_launches += 1;
      out->kernel_launches += out->graph_nodes == 0 ? 1 : out->graph_nodes;
      if (session->cached_experimental_rt_sparse_attention_active != 0 &&
          session->experimental_rt_query_descriptor_selector != 0) {
        out->experimental_rt_selector_launches += session->layer_count;
      }
    }
  }
  if (err == cudaSuccess && run_count != 0)
    err = cudaEventRecord(session->device_stop, session->stream);
  const uint64_t slots_bytes =
      static_cast<uint64_t>(request->steps) * sizeof(NervaCudaSyntheticTokenSlot);
  uint64_t deepseek_runtime_counters[kDeepSeekRuntimeCounterCount] = {};
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(session->host_slots + slot_start,
                          session->device_slots + slot_start, slots_bytes,
                          cudaMemcpyDeviceToHost, session->stream);
    out->d2h_bytes = slots_bytes;
  }
  if (err == cudaSuccess && session->deepseek_runtime_counters_bytes != 0) {
    err = cudaMemcpyAsync(deepseek_runtime_counters,
                          session->device_deepseek_runtime_counters,
                          session->deepseek_runtime_counters_bytes,
                          cudaMemcpyDeviceToHost, session->stream);
    out->d2h_bytes += session->deepseek_runtime_counters_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(session->stream);
    out->sync_calls += 1;
  }
  if (err == cudaSuccess) {
    fill_deepseek_runtime_counter_result(out, deepseek_runtime_counters);
  }
  if (err == cudaSuccess && run_count != 0) {
    float device_ms = 0.0f;
    err = cudaEventElapsedTime(&device_ms, session->device_start,
                               session->device_stop);
    if (err == cudaSuccess && device_ms > 0.0f) {
      out->device_elapsed_ns += static_cast<uint64_t>(device_ms * 1000000.0f);
      if (out->device_elapsed_ns == 0) out->device_elapsed_ns = 1;
    }
  }
  if (err == cudaSuccess) {
    NervaCudaSyntheticTokenSlot *observed_slots = session->host_slots + slot_start;
    out->observed_tokens = observed_from_slot_range(
        request->steps, session->active_has_eos_token, session->active_eos_token,
        observed_slots);
    for (uint32_t index = 0; index < out->observed_tokens; ++index) {
      request->output_tokens[index] = observed_slots[index].token;
    }
    out->last_token = out->observed_tokens == 0
                          ? 0
                          : request->output_tokens[out->observed_tokens - 1];
    out->observed_token_hash =
        hash_tokens(request->output_tokens, out->observed_tokens);
    out->resident_kv_bytes = session_resident_kv_bytes(session);
    out->kv_tokens = slot_start + out->observed_tokens;
    out->device_arena_bytes = session_device_footprint(session);
    out->pinned_host_bytes = session->slots_bytes;
    out->host_causality_edges = 0;
    out->status = out->observed_tokens > 0 ? 0 : -1;
    for (uint32_t index = 0; out->status == 0 && index < out->observed_tokens;
         ++index) {
      const NervaCudaSyntheticTokenSlot &slot = observed_slots[index];
      if (slot.request_id != kRequestId || slot.sequence_id != kSequenceId ||
          slot.completion != kCompletionDeviceComplete ||
          slot.token_index != slot_start + index) {
        out->status = -1;
      }
    }
    if (out->status == 0) {
      scale_profile_counters(out, out->observed_tokens);
      session->active_observed_tokens += out->observed_tokens;
      session->active_cursor =
          out->observed_tokens < request->steps ? session->max_context_tokens
                                                : target_cursor;
      session->active_finished = out->observed_tokens < request->steps ||
                                 out->kv_tokens >= session->max_context_tokens;
      session->projection_batch_own_stream_synchronized = 1;
    }
  } else {
    fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}
