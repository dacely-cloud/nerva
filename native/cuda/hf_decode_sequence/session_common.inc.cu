void free_session_fields(NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr) {
    return;
  }
  if (session->experimental_rt_selector != nullptr) {
    nerva_cuda_rt_candidate_selector_destroy(session->experimental_rt_selector);
    session->experimental_rt_selector = nullptr;
  }
  cudaFree(session->device_experimental_rt_candidate_pages);
  session->device_experimental_rt_candidate_pages = nullptr;
  if (session->cached_graph_exec != nullptr) {
    cudaGraphExecDestroy(session->cached_graph_exec);
  }
  if (session->cached_graph != nullptr) {
    cudaGraphDestroy(session->cached_graph);
  }
#if NERVA_HAVE_CUDNN_FRONTEND
  delete session->cudnn_prefill_sdpa;
  session->cudnn_prefill_sdpa = nullptr;
  delete session->cudnn_decode_sdpa;
  session->cudnn_decode_sdpa = nullptr;
  if (session->cudnn != nullptr) cudnnDestroy(session->cudnn);
#endif
  if (session->profile_stop != nullptr) cudaEventDestroy(session->profile_stop);
  if (session->profile_start != nullptr) cudaEventDestroy(session->profile_start);
  if (session->device_stop != nullptr) cudaEventDestroy(session->device_stop);
  if (session->device_start != nullptr) cudaEventDestroy(session->device_start);
  for (LtGemmTokensPlan &plan : session->projection_block_plans) {
    destroy_lt_gemm_tokens_plan(&plan);
  }
  session->projection_block_plans.clear();
  destroy_lt_gemv_plan(&session->lm_head_plan);
  destroy_lt_gemv_plan(&session->down_plan);
  destroy_lt_gemv_plan(&session->gate_up_plan);
  destroy_lt_gemv_plan(&session->attention_output_plan);
  destroy_lt_gemv_plan(&session->qkv_plan);
  if (session->cublas_lt != nullptr) cublasLtDestroy(session->cublas_lt);
  if (session->cublas != nullptr) cublasDestroy(session->cublas);
  if (session->stream != nullptr) cudaStreamDestroy(session->stream);
  cudaFree(session->cublas_workspace);
  cudaFree(session->device_step);
  cudaFree(session->device_slots);
  cudaFree(session->device_prompt_tokens);
  cudaFree(session->device_kv_block_table);
  cudaFree(session->device_kv_values);
  cudaFree(session->device_kv_keys);
  if (session->shared_weights == nullptr) {
    cudaFree(session->device_gate_up_packed);
    cudaFree(session->device_qkv_packed);
  }
  cudaFree(session->device_prefill_down);
  cudaFree(session->device_prefill_ff);
  cudaFree(session->device_prefill_q_gate);
  cudaFree(session->device_prefill_gate_up);
  cudaFree(session->device_prefill_o);
  cudaFree(session->device_prefill_attn);
  cudaFree(session->device_prefill_qkv_encoded);
  cudaFree(session->device_prefill_qkv);
  cudaFree(session->device_prefill_norm);
  cudaFree(session->device_prefill_hidden_b);
  cudaFree(session->device_prefill_hidden_a);
  cudaFree(session->device_decode_attention_l);
  cudaFree(session->device_decode_attention_m);
  cudaFree(session->device_decode_attention_values);
  cudaFree(session->device_decode_seq_len_kv);
  cudaFree(session->device_decode_seq_len_q);
  cudaFree(session->device_decode_q);
  cudaFree(session->device_linear_gdn_recurrent_state);
  cudaFree(session->device_linear_gdn_conv_state);
  cudaFree(session->device_projection_batch_output);
  cudaFree(session->device_projection_batch_input);
  cudaFree(session->device_projection_input);
  cudaFree(session->device_scratch);
  if (session->shared_weights == nullptr) {
    cudaFree(session->device_layouts);
    cudaFree(session->device_arena);
  } else {
    session->device_gate_up_packed = nullptr;
    session->device_qkv_packed = nullptr;
    session->device_layouts = nullptr;
    session->device_arena = nullptr;
    session->shared_weights.reset();
  }
  cudaFreeHost(session->host_slots);
}

void reset_session_graph(NervaCudaHfDecodeSequenceSession *session) {
  if (session->cached_graph_exec != nullptr) {
    cudaGraphExecDestroy(session->cached_graph_exec);
    session->cached_graph_exec = nullptr;
  }
  if (session->cached_graph != nullptr) {
    cudaGraphDestroy(session->cached_graph);
    session->cached_graph = nullptr;
  }
  session->cached_context_steps = 0;
  session->cached_prompt_token_count = 0;
  session->cached_has_eos_token = 0;
  session->cached_eos_token = 0;
  session->cached_attention_chunks = 0;
  session->cached_experimental_rt_sparse_attention_active = 0;
  session->cached_sampler = default_hf_decode_sampler_config();
  session->cached_graph_nodes = 0;
  session->cached_projection_ns = 0;
  session->cached_qkv_projection_ns = 0;
  session->cached_attention_output_projection_ns = 0;
  session->cached_gate_up_projection_ns = 0;
  session->cached_down_projection_ns = 0;
  session->cached_lm_head_projection_ns = 0;
  session->cached_attention_ns = 0;
  session->cached_mlp_ns = 0;
  session->cached_norm_ns = 0;
  session->cached_sampling_ns = 0;
}

uint64_t session_device_footprint(
    const NervaCudaHfDecodeSequenceSession *session) {
  return session->arena_bytes + session->layout_bytes + session->scratch_bytes +
         session->projection_input_bytes + session->projection_batch_input_bytes +
         session->projection_batch_output_bytes + session->prefill_hidden_bytes * 2 +
         session->prefill_norm_bytes + session->prefill_qkv_bytes +
         session->prefill_qkv_encoded_bytes +
         session->prefill_attn_bytes + session->prefill_o_bytes +
         session->prefill_q_gate_bytes + session->prefill_gate_up_bytes +
         session->prefill_ff_bytes +
         session->prefill_down_bytes + session->decode_attention_values_bytes +
         session->decode_attention_stats_bytes * 2 + session->decode_q_bytes +
         session->decode_seq_len_bytes + session->linear_gdn_conv_state_bytes +
         session->linear_gdn_recurrent_state_bytes +
         session->packed_qkv_bytes + session->packed_gate_up_bytes + session->kv_bytes +
         session->kv_block_table_bytes +
         session->prompt_bytes + session->slots_bytes +
         session->experimental_rt_candidate_pages_bytes + sizeof(uint32_t) +
         kCublasWorkspaceBytes;
}

uint64_t session_fixed_footprint_without_prefill_chunk(
    const NervaCudaHfDecodeSequenceSession *session) {
  return session->arena_bytes + session->layout_bytes + session->scratch_bytes +
         session->projection_input_bytes + session->projection_batch_input_bytes +
         session->projection_batch_output_bytes + session->prefill_hidden_bytes * 2 +
         session->decode_attention_values_bytes +
         session->decode_attention_stats_bytes * 2 + session->decode_q_bytes +
         session->decode_seq_len_bytes + session->linear_gdn_conv_state_bytes +
         session->linear_gdn_recurrent_state_bytes +
         session->packed_qkv_bytes + session->packed_gate_up_bytes + session->kv_bytes +
         session->kv_block_table_bytes +
         session->prompt_bytes + session->slots_bytes +
         session->experimental_rt_candidate_pages_bytes + sizeof(uint32_t) +
         kCublasWorkspaceBytes;
}

uint64_t sat_add_u64(uint64_t lhs, uint64_t rhs) {
  if (UINT64_MAX - lhs < rhs) return UINT64_MAX;
  return lhs + rhs;
}

uint64_t sat_mul_u64(uint64_t lhs, uint64_t rhs) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs) return UINT64_MAX;
  return lhs * rhs;
}

uint64_t full_attention_scratch_elements(uint64_t hidden,
                                         uint64_t attention_hidden,
                                         uint64_t kv_hidden,
                                         uint64_t intermediate) {
  uint64_t total = 0;
  total = sat_add_u64(total, sat_mul_u64(hidden, 5));
  total = sat_add_u64(total, sat_mul_u64(attention_hidden, 3));
  total = sat_add_u64(total, sat_mul_u64(kv_hidden, 2));
  total = sat_add_u64(total, sat_mul_u64(intermediate, 3));
  return total;
}

uint64_t layout_linear_gdn_value_dim(const SequenceLayerLayout &layout) {
  return static_cast<uint64_t>(layout.linear_value_heads) *
         layout.linear_value_head_dim;
}

uint64_t layout_linear_gdn_key_dim(const SequenceLayerLayout &layout) {
  return static_cast<uint64_t>(layout.linear_key_heads) *
         layout.linear_key_head_dim;
}

uint64_t layout_linear_gdn_conv_dim(const SequenceLayerLayout &layout) {
  return sat_add_u64(sat_mul_u64(layout_linear_gdn_key_dim(layout), 2),
                     layout_linear_gdn_value_dim(layout));
}

uint64_t layer_scratch_elements(const SequenceLayerLayout &layout,
                                uint64_t hidden,
                                uint64_t attention_hidden,
                                uint64_t kv_hidden,
                                uint64_t intermediate) {
  if (layout.attention_kind != kAttentionKindLinearGdn) {
    return full_attention_scratch_elements(hidden, attention_hidden, kv_hidden,
                                           intermediate);
  }
  const uint64_t conv_dim = layout_linear_gdn_conv_dim(layout);
  const uint64_t value_dim = layout_linear_gdn_value_dim(layout);
  uint64_t total = 0;
  total = sat_add_u64(total, sat_mul_u64(hidden, 5));
  total = sat_add_u64(total, sat_mul_u64(conv_dim, 2));
  total = sat_add_u64(total, sat_mul_u64(value_dim, 3));
  total = sat_add_u64(
      total, sat_mul_u64(static_cast<uint64_t>(layout.linear_value_heads), 2));
  total = sat_add_u64(total, sat_mul_u64(intermediate, 3));
  return total;
}

uint64_t max_layer_scratch_elements(
    const std::vector<SequenceLayerLayout> &layouts, uint64_t hidden,
    uint64_t attention_hidden, uint64_t kv_hidden, uint64_t intermediate) {
  uint64_t max_scratch = full_attention_scratch_elements(
      hidden, attention_hidden, kv_hidden, intermediate);
  for (const SequenceLayerLayout &layout : layouts) {
    max_scratch = std::max(
        max_scratch, layer_scratch_elements(layout, hidden, attention_hidden,
                                            kv_hidden, intermediate));
  }
  return max_scratch;
}

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor);

uint32_t clamp_nonzero_u32(uint32_t value, uint32_t fallback) {
  return value == 0 ? fallback : value;
}

uint32_t normalize_experimental_rt_mode(uint32_t mode) {
  return mode == kExperimentalRtModeShadow || mode == kExperimentalRtModeSparse
             ? mode
             : kExperimentalRtModeAuto;
}

uint32_t experimental_rt_local_pages(
    const NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr || session->experimental_rt_page_tokens == 0) {
    return 0;
  }
  return ceil_div_u32(session->experimental_rt_local_window_tokens,
                      session->experimental_rt_page_tokens);
}

uint32_t experimental_rt_sink_pages(
    const NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr || session->experimental_rt_page_tokens == 0) {
    return 0;
  }
  return ceil_div_u32(session->experimental_rt_sink_tokens,
                      session->experimental_rt_page_tokens);
}

bool experimental_rt_sparse_available(
    const NervaCudaHfDecodeSequenceSession *session,
    uint32_t dense_attention_chunks) {
  return session != nullptr && session->experimental_rt_decode_enabled != 0 &&
         session->experimental_rt_selector != nullptr &&
         session->device_experimental_rt_candidate_pages != nullptr &&
         dense_attention_chunks != 0 &&
         session->experimental_rt_pages != 0 &&
         session->experimental_rt_pages < dense_attention_chunks &&
         (session->experimental_rt_mode == kExperimentalRtModeAuto ||
          session->experimental_rt_mode == kExperimentalRtModeSparse);
}

uint32_t experimental_rt_attention_chunks_for(
    const NervaCudaHfDecodeSequenceSession *session,
    uint32_t dense_attention_chunks) {
  return experimental_rt_sparse_available(session, dense_attention_chunks)
             ? session->experimental_rt_pages
             : dense_attention_chunks;
}

bool experimental_rt_should_launch_for(
    const NervaCudaHfDecodeSequenceSession *session,
    uint32_t dense_attention_chunks) {
  if (session == nullptr || session->experimental_rt_decode_enabled == 0 ||
      session->experimental_rt_selector == nullptr) {
    return false;
  }
  if (session->experimental_rt_mode == kExperimentalRtModeShadow) {
    return true;
  }
  return experimental_rt_sparse_available(session, dense_attention_chunks);
}

const uint32_t *experimental_rt_selected_chunks_for(
    const NervaCudaHfDecodeSequenceSession *session,
    uint32_t dense_attention_chunks) {
  return experimental_rt_sparse_available(session, dense_attention_chunks)
             ? session->device_experimental_rt_candidate_pages
             : nullptr;
}

cudaError_t initialize_experimental_rt_selector(
    NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr || session->experimental_rt_decode_requested == 0) {
    return cudaSuccess;
  }
  session->experimental_rt_mode =
      normalize_experimental_rt_mode(session->experimental_rt_mode);
  const uint32_t page_tokens =
      clamp_nonzero_u32(session->experimental_rt_page_tokens, 64u);
  const uint32_t pages = ceil_div_u32(session->max_context_tokens, page_tokens);
  if (pages == 0) {
    return cudaErrorInvalidValue;
  }
  const uint32_t requested_candidates =
      clamp_nonzero_u32(session->experimental_rt_pages, pages);
  const uint32_t candidates = requested_candidates < pages ? requested_candidates : pages;
  const uint32_t query_count = session->kv_heads == 0 ? 1u : session->kv_heads;
  session->experimental_rt_page_tokens = page_tokens;
  session->experimental_rt_pages = candidates;
  session->experimental_rt_query_count = query_count;
  if (session->experimental_rt_mode != kExperimentalRtModeShadow &&
      candidates >= pages) {
    return cudaSuccess;
  }
  const uint64_t candidate_count =
      static_cast<uint64_t>(query_count) * candidates;
  if (candidate_count == 0 ||
      candidate_count > UINT64_MAX / sizeof(uint32_t)) {
    return cudaErrorInvalidValue;
  }
  session->experimental_rt_candidate_pages_bytes =
      candidate_count * sizeof(uint32_t);
  cudaError_t err = cudaMalloc(
      reinterpret_cast<void **>(&session->device_experimental_rt_candidate_pages),
      session->experimental_rt_candidate_pages_bytes);
  if (err != cudaSuccess) {
    return err;
  }
  int32_t rt_cuda_error = static_cast<int32_t>(cudaSuccess);
  void *selector = nullptr;
  const int rt_status = nerva_cuda_rt_candidate_selector_create(
      pages, page_tokens, query_count, candidates,
      session->device_experimental_rt_candidate_pages, session->stream,
      &selector, &rt_cuda_error);
  if (rt_status != 0 || selector == nullptr) {
    cudaFree(session->device_experimental_rt_candidate_pages);
    session->device_experimental_rt_candidate_pages = nullptr;
    session->experimental_rt_candidate_pages_bytes = 0;
    return rt_cuda_error == static_cast<int32_t>(cudaSuccess)
               ? cudaErrorNotSupported
               : static_cast<cudaError_t>(rt_cuda_error);
  }
  session->experimental_rt_selector = selector;
  session->experimental_rt_decode_enabled = 1;
  return cudaSuccess;
}

cudaError_t launch_experimental_rt_selector(
    NervaCudaHfDecodeSequenceSession *session, uint32_t current_position,
    uint32_t dense_attention_chunks) {
  if (!experimental_rt_should_launch_for(session, dense_attention_chunks)) {
    return cudaSuccess;
  }
  const uint32_t active_pages =
      dense_attention_chunks == 0
          ? ceil_div_u32(current_position + 1u,
                         session->experimental_rt_page_tokens)
          : dense_attention_chunks;
  const uint32_t current_page =
      session->experimental_rt_page_tokens == 0
          ? 0u
          : current_position / session->experimental_rt_page_tokens;
  int32_t rt_cuda_error = static_cast<int32_t>(cudaSuccess);
  const int rt_status = nerva_cuda_rt_candidate_selector_launch(
      session->experimental_rt_selector, session->stream, active_pages,
      current_page, experimental_rt_local_pages(session),
      experimental_rt_sink_pages(session), &rt_cuda_error);
  if (rt_status != 0) {
    return rt_cuda_error == static_cast<int32_t>(cudaSuccess)
               ? cudaErrorUnknown
               : static_cast<cudaError_t>(rt_cuda_error);
  }
  session->experimental_rt_shadow_launches += 1;
  return cudaSuccess;
}

uint64_t prefill_chunk_scratch_bytes(uint64_t chunk_tokens,
                                     uint64_t projection_input_elements,
                                     uint64_t prefill_qkv_rows,
                                     uint64_t attention_hidden,
                                     uint64_t hidden,
                                     uint64_t prefill_q_gate_rows,
                                     uint64_t prefill_gate_up_rows,
                                     uint64_t intermediate) {
  uint64_t total = 0;
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(projection_input_elements, chunk_tokens), sizeof(uint16_t)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(prefill_qkv_rows, chunk_tokens), sizeof(float)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(prefill_qkv_rows, chunk_tokens), sizeof(uint16_t)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(attention_hidden, chunk_tokens), sizeof(uint16_t)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(hidden, chunk_tokens), sizeof(float)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(prefill_q_gate_rows, chunk_tokens), sizeof(float)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(prefill_gate_up_rows, chunk_tokens), sizeof(float)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(intermediate, chunk_tokens), sizeof(uint16_t)));
  total = sat_add_u64(total, sat_mul_u64(
      sat_mul_u64(hidden, chunk_tokens), sizeof(float)));
  return total;
}

uint32_t tune_prefill_chunk_tokens(uint64_t max_context_tokens,
                                   uint64_t fixed_device_bytes,
                                   uint64_t projection_input_elements,
                                   uint64_t prefill_qkv_rows,
                                   uint64_t attention_hidden,
                                   uint64_t hidden,
                                   uint64_t prefill_q_gate_rows,
                                   uint64_t prefill_gate_up_rows,
                                   uint64_t intermediate,
                                   uint64_t free_device_bytes) {
  if (max_context_tokens == 0) return 0;
  const uint64_t base =
      std::min<uint64_t>(kPrefillChunkBaseTokens, max_context_tokens);
  const uint64_t max_target =
      std::min<uint64_t>(kPrefillChunkMaxTokens, max_context_tokens);
  const uint64_t min_chunk = std::min<uint64_t>(base, max_context_tokens);
  if (free_device_bytes == 0) {
    return static_cast<uint32_t>(base);
  }
  const uint64_t budget =
      free_device_bytes > kPrefillAutotuneSafetyBytes
          ? free_device_bytes - kPrefillAutotuneSafetyBytes
          : free_device_bytes;
  auto fits = [&](uint64_t candidate) {
    const uint64_t footprint = sat_add_u64(
        fixed_device_bytes,
        prefill_chunk_scratch_bytes(candidate, projection_input_elements,
                                    prefill_qkv_rows, attention_hidden, hidden,
                                    prefill_q_gate_rows,
                                    prefill_gate_up_rows, intermediate));
    return footprint <= budget;
  };
  uint64_t chunk = base;
  while (chunk > min_chunk && !fits(chunk)) {
    chunk = std::max<uint64_t>(min_chunk, chunk / 2);
  }
  while (chunk < max_target) {
    const uint64_t next = std::min<uint64_t>(max_target, chunk * 2);
    if (next == chunk || !fits(next)) break;
    chunk = next;
  }
  return static_cast<uint32_t>(chunk);
}

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
  return divisor == 0 ? 0 : (value + divisor - 1u) / divisor;
}

uint32_t ceil_div_u64_to_u32(uint64_t value, uint32_t divisor) {
  if (divisor == 0) return 0;
  const uint64_t blocks = (value + divisor - 1u) / divisor;
  return blocks > 0xffffffffu ? 0xffffffffu : static_cast<uint32_t>(blocks);
}

cudaError_t deinterleave_descriptor_query_gate_weights(
    uint16_t *device_arena, const std::vector<SequenceLayerLayout> &layouts,
    uint32_t hidden, uint32_t heads, uint32_t head_dim, cudaStream_t stream,
    uint64_t *setup_sync_calls) {
  if (device_arena == nullptr || hidden == 0 || heads == 0 || head_dim == 0) {
    return cudaErrorInvalidValue;
  }
  bool has_query_gate = false;
  for (const SequenceLayerLayout &layout : layouts) {
    if (layout.w_q_gate != kMissingOffset) {
      has_query_gate = true;
      break;
    }
  }
  if (!has_query_gate) {
    return cudaSuccess;
  }

  const uint64_t attention_hidden = static_cast<uint64_t>(heads) * head_dim;
  const uint64_t q_elements = attention_hidden * static_cast<uint64_t>(hidden);
  const uint64_t packed_bytes = q_elements * 2u * sizeof(uint16_t);
  uint16_t *temporary = nullptr;
  cudaError_t err =
      cudaMalloc(reinterpret_cast<void **>(&temporary), packed_bytes);
  if (err != cudaSuccess) {
    return err;
  }

  const uint32_t blocks = ceil_div_u64_to_u32(q_elements, kDecodeThreads);
  for (const SequenceLayerLayout &layout : layouts) {
    if (layout.w_q_gate == kMissingOffset) {
      continue;
    }
    err = cudaMemcpyAsync(temporary, device_arena + layout.w_q, packed_bytes,
                          cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
      break;
    }
    hf_deinterleave_q_gate_projection_kernel<<<blocks, kDecodeThreads, 0,
                                               stream>>>(
        temporary, device_arena + layout.w_q, device_arena + layout.w_q_gate,
        heads, head_dim, hidden);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      break;
    }
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
    if (err == cudaSuccess && setup_sync_calls != nullptr) {
      *setup_sync_calls += 1;
    }
  }
  cudaFree(temporary);
  return err;
}

uint32_t next_pow2_at_least(uint32_t value, uint32_t minimum,
                            uint32_t maximum) {
  uint32_t out = minimum;
  while (out < value && out < maximum) {
    out <<= 1;
  }
  return out > maximum ? maximum : out;
}

uint32_t tuned_head_threads(uint32_t head_dim, const cudaDeviceProp &props) {
  const uint32_t warp_threads = props.warpSize > 0 ? props.warpSize : 32u;
  const uint32_t minimum = props.major >= 9 ? std::max(warp_threads, 64u)
                                            : warp_threads;
  const uint32_t exact_head_threads =
      next_pow2_at_least(head_dim, minimum, kHeadThreadsMax);
  const uint32_t compact_threads = next_pow2_at_least(
      ceil_div_u32(head_dim, kHeadThreadElements), minimum, kHeadThreadsMax);
  if (props.major >= 9 && compact_threads < exact_head_threads) {
    return compact_threads;
  }
  return exact_head_threads;
}

uint32_t decode_attention_chunks_for_cursor(
    const NervaCudaHfDecodeSequenceSession *session, uint32_t cursor) {
  const uint32_t kv_tokens = cursor >= session->max_context_tokens
                                 ? session->max_context_tokens
                                 : cursor + 1u;
  if (kv_tokens <= kChunkedDecodeAttentionThreshold ||
      session->decode_attention_max_chunks == 0 ||
      session->device_decode_attention_values == nullptr ||
      session->device_decode_attention_m == nullptr ||
      session->device_decode_attention_l == nullptr ||
      session->head_dim > kDecodeThreads) {
    return 0;
  }
  const uint32_t chunks =
      ceil_div_u32(kv_tokens, kDecodeAttentionChunkTokens);
  return std::min(chunks, session->decode_attention_max_chunks);
}

uint32_t decode_head_threads_for_session(
    const NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr) {
    return kHeadThreadsMax;
  }
  return next_pow2_at_least(session->head_dim, session->head_threads,
                            kHeadThreadsMax);
}

bool session_graph_matches(const NervaCudaHfDecodeSequenceSession *session,
                           uint32_t context_steps,
                           uint32_t prompt_token_count,
                           uint32_t has_eos_token,
                           uint32_t eos_token,
                           uint32_t attention_chunks,
                           NervaCudaHfDecodeSamplerConfig sampler) {
  return session->cached_graph_exec != nullptr &&
         session->cached_context_steps == context_steps &&
         session->cached_prompt_token_count == prompt_token_count &&
         session->cached_has_eos_token == has_eos_token &&
         session->cached_eos_token == eos_token &&
         session->cached_attention_chunks == attention_chunks &&
         session->cached_experimental_rt_sparse_attention_active ==
             session->experimental_rt_sparse_attention_active &&
         hf_decode_sampler_config_matches(session->cached_sampler, sampler);
}

bool session_has_sparse_moe_layers(
    const NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr ||
      session->host_layouts.size() != session->layer_count) {
    return false;
  }
  for (const SequenceLayerLayout &layout : session->host_layouts) {
    if (layout.mlp_kind == kMlpKindSparseMoe) {
      return true;
    }
  }
  return false;
}

bool session_has_dense_mlp_layers(
    const NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr ||
      session->host_layouts.size() != session->layer_count) {
    return false;
  }
  for (const SequenceLayerLayout &layout : session->host_layouts) {
    if (layout.mlp_kind == kMlpKindDense) {
      return true;
    }
  }
  return false;
}

bool session_has_query_gate_layers(
    const NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr ||
      session->host_layouts.size() != session->layer_count) {
    return false;
  }
  for (const SequenceLayerLayout &layout : session->host_layouts) {
    if (layout.w_q_gate != kMissingOffset) {
      return true;
    }
  }
  return false;
}

bool session_has_linear_gdn_layers(
    const NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr ||
      session->host_layouts.size() != session->layer_count) {
    return false;
  }
  for (const SequenceLayerLayout &layout : session->host_layouts) {
    if (layout.attention_kind == kAttentionKindLinearGdn) {
      return true;
    }
  }
  return false;
}

bool use_cublas_layer_path(const NervaCudaHfDecodeSequenceSession *session) {
  const uint32_t attention_hidden = session->heads * session->head_dim;
  return session->hidden >= 128 && attention_hidden == session->hidden &&
         session->host_layouts.size() == session->layer_count &&
         !session_has_linear_gdn_layers(session) &&
         session->device_projection_input != nullptr &&
         session->device_qkv_packed != nullptr &&
         (!session_has_dense_mlp_layers(session) ||
          session->device_gate_up_packed != nullptr) &&
         session->cublas != nullptr && session->cublas_lt != nullptr;
}

bool use_cublas_prefill_path(const NervaCudaHfDecodeSequenceSession *session) {
  return use_cublas_layer_path(session);
}

bool projection_batch_session_ready(
    const NervaCudaHfDecodeSequenceSession *session) {
  const uint32_t attention_hidden = session->heads * session->head_dim;
  return session->hidden >= 128 && attention_hidden == session->hidden &&
         session->host_layouts.size() == session->layer_count &&
         !session_has_linear_gdn_layers(session) &&
         !session_has_sparse_moe_layers(session) &&
         !session_has_query_gate_layers(session) &&
         session->device_projection_input != nullptr &&
         session->device_qkv_packed != nullptr &&
         session->device_gate_up_packed != nullptr;
}

cudaError_t autotune_session_lt_gemv_plans(
    NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr || !use_cublas_layer_path(session) ||
      session->layer_count == 0) {
    return cudaErrorInvalidValue;
  }
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const PackedProjectionShape packed_shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  const SequenceLayerLayout layout = session->host_layouts[0];
  const bool has_dense_mlp = session_has_dense_mlp_layers(session);
  int32_t first_dense_layer = -1;
  for (uint32_t layer_index = 0; layer_index < session->layer_count;
       ++layer_index) {
    if (session->host_layouts[layer_index].mlp_kind == kMlpKindDense) {
      first_dense_layer = static_cast<int32_t>(layer_index);
      break;
    }
  }
  LayerScratch scratch = layer_scratch_ptrs(
      session->device_scratch, session->hidden, attention_hidden, kv_hidden,
      session->intermediate);
  cudaError_t err = cudaMemsetAsync(
      session->device_projection_input, 0, session->projection_input_bytes,
      session->stream);
  if (err == cudaSuccess)
    err = create_lt_gemv_plan(
        &session->qkv_plan, static_cast<uint32_t>(packed_shape.qkv_rows),
        session->hidden, session->dtype);
  if (err == cudaSuccess)
    err = create_lt_gemv_plan(&session->attention_output_plan,
                              session->hidden, attention_hidden,
                              session->dtype);
  if (err == cudaSuccess)
    err = has_dense_mlp
              ? create_lt_gemv_plan(
                    &session->gate_up_plan,
                    static_cast<uint32_t>(packed_shape.gate_up_rows),
                    session->hidden, session->dtype)
              : cudaSuccess;
  if (err == cudaSuccess)
    err = has_dense_mlp
              ? create_lt_gemv_plan(&session->down_plan, session->hidden,
                                    session->intermediate, session->dtype)
              : cudaSuccess;
  if (err == cudaSuccess)
    err = create_lt_gemv_plan(&session->lm_head_plan, session->vocab_size,
                              session->hidden, session->dtype);
  if (err == cudaSuccess)
    err = autotune_lt_gemv_plan(
        session->cublas, session->cublas_lt, session->stream,
        session->cublas_workspace, kCublasWorkspaceBytes, &session->qkv_plan,
        session->device_qkv_packed, session->device_projection_input,
        scratch.q);
  if (err == cudaSuccess)
    err = autotune_lt_gemv_plan(
        session->cublas, session->cublas_lt, session->stream,
        session->cublas_workspace, kCublasWorkspaceBytes,
        &session->attention_output_plan,
        session->device_arena + layout.w_o, session->device_projection_input,
        scratch.residual);
  if (err == cudaSuccess)
    err = has_dense_mlp
              ? autotune_lt_gemv_plan(
                    session->cublas, session->cublas_lt, session->stream,
                    session->cublas_workspace, kCublasWorkspaceBytes,
                    &session->gate_up_plan,
                    session->device_gate_up_packed +
                        packed_shape.gate_up_elements_per_layer *
                            static_cast<uint32_t>(first_dense_layer),
                    session->device_projection_input, scratch.gate)
              : cudaSuccess;
  if (err == cudaSuccess)
    err = has_dense_mlp
              ? autotune_lt_gemv_plan(
                    session->cublas, session->cublas_lt, session->stream,
                    session->cublas_workspace, kCublasWorkspaceBytes,
                    &session->down_plan,
                    session->device_arena +
                        session->host_layouts[static_cast<uint32_t>(
                            first_dense_layer)]
                            .w_down,
                    session->device_projection_input, scratch.down)
              : cudaSuccess;
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = autotune_lt_gemv_plan(
        session->cublas, session->cublas_lt, session->stream,
        session->cublas_workspace, kCublasWorkspaceBytes,
        &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, device_logits);
  }
  return err;
}

cudaError_t ensure_session_cublas_resources(
    NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr || !projection_batch_session_ready(session)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t err = cudaSuccess;
  if (session->cublas_workspace == nullptr) {
    err = cudaMalloc(&session->cublas_workspace, kCublasWorkspaceBytes);
  }
  if (err == cudaSuccess && session->cublas == nullptr) {
    err = cublas_to_cuda(cublasCreate(&session->cublas));
  }
  if (err == cudaSuccess && session->cublas_lt == nullptr) {
    err = cublas_to_cuda(cublasLtCreate(&session->cublas_lt));
  }
  if (err == cudaSuccess) {
    err = configure_cublas(session->cublas, session->stream,
                           session->cublas_workspace,
                           kCublasWorkspaceBytes);
  }
  if (err == cudaSuccess && !use_cublas_layer_path(session)) {
    err = cudaErrorInvalidValue;
  }
  const bool dense_mlp_ready =
      !session_has_dense_mlp_layers(session) ||
      (session->gate_up_plan.ready && session->down_plan.ready);
  if (err == cudaSuccess && (!session->qkv_plan.ready ||
                             !session->attention_output_plan.ready ||
                             !dense_mlp_ready || !session->lm_head_plan.ready)) {
    err = autotune_session_lt_gemv_plans(session);
  }
  return err;
}

void copy_cached_profile(const NervaCudaHfDecodeSequenceSession *session,
                         NervaCudaHfDecodeSequenceResult *out) {
  out->projection_ns = session->cached_projection_ns;
  out->qkv_projection_ns = session->cached_qkv_projection_ns;
  out->attention_output_projection_ns =
      session->cached_attention_output_projection_ns;
  out->gate_up_projection_ns = session->cached_gate_up_projection_ns;
  out->down_projection_ns = session->cached_down_projection_ns;
  out->lm_head_projection_ns = session->cached_lm_head_projection_ns;
  out->attention_ns = session->cached_attention_ns;
  out->mlp_ns = session->cached_mlp_ns;
  out->norm_ns = session->cached_norm_ns;
  out->sampling_ns = session->cached_sampling_ns;
}

cudaError_t encoded_row_major_gemm_tokens_cached(
    NervaCudaHfDecodeSequenceSession *session, const uint16_t *matrix,
    const uint16_t *input, uint32_t rows, uint32_t cols, uint32_t tokens,
    uint32_t dtype, float beta, float *output);

cudaError_t project_encoded_rows(NervaCudaHfDecodeSequenceSession *session,
                                 const LtGemvPlan *single_token_plan,
                                 const uint16_t *matrix,
                                 const uint16_t *input, uint32_t rows,
                                 uint32_t cols, uint32_t tokens,
                                 uint32_t dtype, float beta,
                                 float *output) {
  if (session == nullptr || matrix == nullptr || input == nullptr ||
      output == nullptr || rows == 0 || cols == 0 || tokens == 0) {
    return cudaErrorInvalidValue;
  }
  if (tokens == 1) {
    if (single_token_plan != nullptr && single_token_plan->ready &&
        single_token_plan->rows == rows && single_token_plan->cols == cols &&
        single_token_plan->dtype == dtype) {
      return encoded_row_major_gemv_planned(
          session->cublas, session->cublas_lt, session->stream,
          session->cublas_workspace, kCublasWorkspaceBytes, single_token_plan,
          matrix, input, beta, output);
    }
    return encoded_row_major_gemv_beta(session->cublas, matrix, input, rows,
                                       cols, dtype, beta, output);
  }
  return encoded_row_major_gemm_tokens_cached(session, matrix, input, rows, cols,
                                             tokens, dtype, beta, output);
}
