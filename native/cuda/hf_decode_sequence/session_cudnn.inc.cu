#if NERVA_HAVE_CUDNN_FRONTEND
bool cudnn_decode_debug_enabled() {
  static int enabled = []() {
    const char *value = getenv("NERVA_CUDNN_DECODE_DEBUG");
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
  }();
  return enabled != 0;
}

bool cudnn_decode_runtime_enabled() {
  static int enabled = []() {
    const char *value = getenv("NERVA_CUDNN_DECODE");
    if (value == nullptr || value[0] == '\0') {
      return 1;
    }
    const bool is_disabled =
        strcmp(value, "0") == 0 || strcmp(value, "false") == 0 ||
        strcmp(value, "False") == 0 || strcmp(value, "FALSE") == 0;
    return is_disabled ? 0 : 1;
  }();
  return enabled != 0;
}

void log_cudnn_decode_status(const char *phase,
                             cudnn_frontend::error_object status) {
  if (!cudnn_decode_debug_enabled()) {
    return;
  }
  fprintf(stderr, "[nerva-cudnn-decode] %s failed code=%d message=%s\n",
          phase, static_cast<int>(status.get_code()),
          status.get_message().c_str());
}

void log_cudnn_decode_cuda_error(const char *phase, cudaError_t err) {
  if (!cudnn_decode_debug_enabled()) {
    return;
  }
  fprintf(stderr, "[nerva-cudnn-decode] %s failed cuda=%s: %s\n", phase,
          cudaGetErrorName(err), cudaGetErrorString(err));
}

cudaError_t ensure_cudnn_prefill_sdpa_plan(
    NervaCudaHfDecodeSequenceSession *session, uint32_t seq_tokens) {
  if (session == nullptr || session->cudnn == nullptr || seq_tokens == 0 ||
      session->dtype != kDTypeBF16 || session->head_dim == 0 ||
      session->heads == 0 || session->kv_heads == 0 ||
      session->heads % session->kv_heads != 0) {
    return cudaErrorNotSupported;
  }
  if (session->cudnn_prefill_sdpa != nullptr &&
      session->cudnn_prefill_sdpa->seq_tokens == seq_tokens &&
      session->cudnn_prefill_sdpa->heads == session->heads &&
      session->cudnn_prefill_sdpa->kv_heads == session->kv_heads &&
      session->cudnn_prefill_sdpa->head_dim == session->head_dim) {
    return cudaSuccess;
  }

  auto *plan = new (std::nothrow) CudnnPrefillSdpaPlan();
  if (plan == nullptr) {
    return cudaErrorMemoryAllocation;
  }
  plan->seq_tokens = seq_tokens;
  plan->heads = session->heads;
  plan->kv_heads = session->kv_heads;
  plan->head_dim = session->head_dim;
  const uint64_t attention_hidden =
      static_cast<uint64_t>(session->heads) * session->head_dim;
  const uint64_t kv_hidden =
      static_cast<uint64_t>(session->kv_heads) * session->head_dim;
  plan->rows = attention_hidden + kv_hidden * 2;
  plan->graph = std::make_unique<cudnn_frontend::graph::Graph>();
  plan->graph->set_io_data_type(cudnn_frontend::DataType_t::BFLOAT16)
      .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

  constexpr int64_t kTensorQ = 9001;
  constexpr int64_t kTensorK = 9002;
  constexpr int64_t kTensorV = 9003;
  constexpr int64_t kTensorO = 9004;
  const int64_t batch = 1;
  const int64_t heads = static_cast<int64_t>(session->heads);
  const int64_t kv_heads = static_cast<int64_t>(session->kv_heads);
  const int64_t seq = static_cast<int64_t>(seq_tokens);
  const int64_t dim = static_cast<int64_t>(session->head_dim);
  const int64_t rows = static_cast<int64_t>(plan->rows);
  const std::vector<int64_t> q_stride = {seq * rows, dim, rows, 1};
  const std::vector<int64_t> k_stride = {seq * rows, dim, rows, 1};
  const std::vector<int64_t> v_stride = {seq * rows, dim, rows, 1};
  const std::vector<int64_t> o_stride = {
      seq * heads * dim, dim, heads * dim, 1};

  auto q_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_q")
                    .set_uid(kTensorQ)
                    .set_dim({batch, heads, seq, dim})
                    .set_stride(q_stride);
  auto k_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_k")
                    .set_uid(kTensorK)
                    .set_dim({batch, kv_heads, seq, dim})
                    .set_stride(k_stride);
  auto v_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_v")
                    .set_uid(kTensorV)
                    .set_dim({batch, kv_heads, seq, dim})
                    .set_stride(v_stride);
  auto o_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_o")
                    .set_uid(kTensorO)
                    .set_dim({batch, heads, seq, dim})
                    .set_stride(o_stride);

  auto sdpa = cudnn_frontend::graph::SDPA_attributes()
                  .set_name("nerva_prefill_sdpa")
                  .set_generate_stats(false)
                  .set_causal_mask(true)
                  .set_attn_scale(rsqrtf(static_cast<float>(session->head_dim)));
  auto q = plan->graph->tensor(q_desc);
  auto k = plan->graph->tensor(k_desc);
  auto v = plan->graph->tensor(v_desc);
  auto outputs = plan->graph->sdpa(q, k, v, sdpa);
  outputs[0]->set_output(true)
      .set_dim({batch, heads, seq, dim})
      .set_stride(o_stride)
      .set_uid(kTensorO);

  cudnnStatus_t stream_status = cudnnSetStream(session->cudnn, session->stream);
  if (stream_status != CUDNN_STATUS_SUCCESS) {
    delete plan;
    return cudnn_to_cuda(stream_status);
  }
  auto status = plan->graph->build(session->cudnn,
                                   {cudnn_frontend::HeurMode_t::A});
  if (status.is_bad()) {
    delete plan;
    return cudaErrorNotSupported;
  }
  const int64_t workspace = plan->graph->get_workspace_size();
  if (workspace < 0 ||
      static_cast<uint64_t>(workspace) > kCublasWorkspaceBytes) {
    delete plan;
    return cudaErrorMemoryAllocation;
  }
  plan->workspace_bytes = static_cast<size_t>(workspace);
  delete session->cudnn_prefill_sdpa;
  session->cudnn_prefill_sdpa = plan;
  return cudaSuccess;
}

cudaError_t execute_cudnn_prefill_sdpa(
    NervaCudaHfDecodeSequenceSession *session, uint32_t seq_tokens) {
  cudaError_t err = ensure_cudnn_prefill_sdpa_plan(session, seq_tokens);
  if (err != cudaSuccess) {
    return err;
  }
  constexpr int64_t kTensorQ = 9001;
  constexpr int64_t kTensorK = 9002;
  constexpr int64_t kTensorV = 9003;
  constexpr int64_t kTensorO = 9004;
  const uint64_t attention_hidden =
      static_cast<uint64_t>(session->heads) * session->head_dim;
  const uint64_t kv_hidden =
      static_cast<uint64_t>(session->kv_heads) * session->head_dim;
  uint16_t *base = session->device_prefill_qkv_encoded;
  std::unordered_map<int64_t, void *> tensors = {
      {kTensorQ, base},
      {kTensorK, base + attention_hidden},
      {kTensorV, base + attention_hidden + kv_hidden},
      {kTensorO, session->device_prefill_attn},
  };
  cudnnStatus_t stream_status = cudnnSetStream(session->cudnn, session->stream);
  if (stream_status != CUDNN_STATUS_SUCCESS) {
    return cudnn_to_cuda(stream_status);
  }
  auto status = session->cudnn_prefill_sdpa->graph->execute(
      session->cudnn, tensors, session->cublas_workspace);
  return status.is_good() ? cudaSuccess : cudaErrorLaunchFailure;
}

bool can_use_cudnn_decode_sdpa(const NervaCudaHfDecodeSequenceSession *session,
                               uint32_t attention_chunks) {
  const bool usable =
      session != nullptr && attention_chunks != 0 &&
      cudnn_decode_runtime_enabled() &&
      session->cudnn_decode_sdpa_disabled == 0 &&
      session->cudnn != nullptr && session->dtype == kDTypeBF16 &&
      session->heads != 0 && session->kv_heads != 0 &&
      session->heads % session->kv_heads == 0 && session->head_dim != 0 &&
      session->device_decode_q != nullptr &&
      session->device_decode_seq_len_q != nullptr &&
      session->device_decode_seq_len_kv != nullptr;
  if (!usable && cudnn_decode_debug_enabled()) {
    fprintf(stderr,
            "[nerva-cudnn-decode] gate failed session=%d chunks=%u disabled=%u "
            "cudnn=%d dtype=%u heads=%u kv_heads=%u head_dim=%u q=%d "
            "seq_q=%d seq_kv=%d\n",
            session != nullptr, attention_chunks,
            session == nullptr ? 0 : session->cudnn_decode_sdpa_disabled,
            session != nullptr && session->cudnn != nullptr,
            session == nullptr ? 0 : session->dtype,
            session == nullptr ? 0 : session->heads,
            session == nullptr ? 0 : session->kv_heads,
            session == nullptr ? 0 : session->head_dim,
            session != nullptr && session->device_decode_q != nullptr,
            session != nullptr && session->device_decode_seq_len_q != nullptr,
            session != nullptr && session->device_decode_seq_len_kv != nullptr);
  }
  return usable;
}

cudaError_t ensure_cudnn_decode_sdpa_plan(
    NervaCudaHfDecodeSequenceSession *session) {
  if (!can_use_cudnn_decode_sdpa(session, 1)) {
    return cudaErrorNotSupported;
  }
  if (session->cudnn_decode_sdpa != nullptr &&
      session->cudnn_decode_sdpa->max_context_tokens ==
          session->max_context_tokens &&
      session->cudnn_decode_sdpa->kv_token_capacity ==
          session->kv_token_capacity &&
      session->cudnn_decode_sdpa->heads == session->heads &&
      session->cudnn_decode_sdpa->kv_heads == session->kv_heads &&
      session->cudnn_decode_sdpa->head_dim == session->head_dim) {
    return cudaSuccess;
  }

  auto *plan = new (std::nothrow) CudnnDecodeSdpaPlan();
  if (plan == nullptr) {
    return cudaErrorMemoryAllocation;
  }
  plan->max_context_tokens = session->max_context_tokens;
  plan->kv_token_capacity = session->kv_token_capacity;
  plan->heads = session->heads;
  plan->kv_heads = session->kv_heads;
  plan->head_dim = session->head_dim;
  plan->graph = std::make_unique<cudnn_frontend::graph::Graph>();
  plan->graph->set_io_data_type(cudnn_frontend::DataType_t::BFLOAT16)
      .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

  constexpr int64_t kTensorQ = 9101;
  constexpr int64_t kTensorK = 9102;
  constexpr int64_t kTensorV = 9103;
  constexpr int64_t kTensorO = 9104;
  constexpr int64_t kTensorSeqLenQ = 9105;
  constexpr int64_t kTensorSeqLenKv = 9106;
  const int64_t batch = 1;
  const int64_t heads = static_cast<int64_t>(session->heads);
  const int64_t kv_heads = static_cast<int64_t>(session->kv_heads);
  const int64_t seq = static_cast<int64_t>(session->kv_token_capacity);
  const int64_t dim = static_cast<int64_t>(session->head_dim);
  const int64_t attention_hidden = heads * dim;
  const int64_t kv_hidden = kv_heads * dim;
  const std::vector<int64_t> q_stride = {attention_hidden, dim,
                                         attention_hidden, 1};
  const std::vector<int64_t> kv_stride = {seq * kv_hidden, dim, kv_hidden, 1};

  auto q_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_decode_q")
                    .set_uid(kTensorQ)
                    .set_data_type(cudnn_frontend::DataType_t::BFLOAT16)
                    .set_dim({batch, heads, 1, dim})
                    .set_stride(q_stride);
  auto k_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_decode_k_cache")
                    .set_uid(kTensorK)
                    .set_data_type(cudnn_frontend::DataType_t::BFLOAT16)
                    .set_dim({batch, kv_heads, seq, dim})
                    .set_stride(kv_stride);
  auto v_desc = cudnn_frontend::graph::Tensor_attributes()
                    .set_name("nerva_decode_v_cache")
                    .set_uid(kTensorV)
                    .set_data_type(cudnn_frontend::DataType_t::BFLOAT16)
                    .set_dim({batch, kv_heads, seq, dim})
                    .set_stride(kv_stride);
  auto seq_len_q_desc = cudnn_frontend::graph::Tensor_attributes()
                            .set_name("nerva_decode_seq_len_q")
                            .set_uid(kTensorSeqLenQ)
                            .set_data_type(cudnn_frontend::DataType_t::INT32)
                            .set_dim({batch, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(false);
  auto seq_len_kv_desc = cudnn_frontend::graph::Tensor_attributes()
                             .set_name("nerva_decode_seq_len_kv")
                             .set_uid(kTensorSeqLenKv)
                             .set_data_type(cudnn_frontend::DataType_t::INT32)
                             .set_dim({batch, 1, 1, 1})
                             .set_stride({1, 1, 1, 1})
                             .set_is_pass_by_value(false);

  auto q = plan->graph->tensor(q_desc);
  auto k = plan->graph->tensor(k_desc);
  auto v = plan->graph->tensor(v_desc);
  auto seq_len_q = plan->graph->tensor(seq_len_q_desc);
  auto seq_len_kv = plan->graph->tensor(seq_len_kv_desc);
  auto sdpa = cudnn_frontend::graph::SDPA_attributes()
                  .set_name("nerva_decode_sdpa")
                  .set_generate_stats(false)
                  .set_padding_mask(true)
                  .set_seq_len_q(seq_len_q)
                  .set_seq_len_kv(seq_len_kv)
                  .set_attn_scale(rsqrtf(static_cast<float>(session->head_dim)));
  auto outputs = plan->graph->sdpa(q, k, v, sdpa);
  outputs[0]->set_output(true)
      .set_uid(kTensorO)
      .set_data_type(cudnn_frontend::DataType_t::BFLOAT16)
      .set_dim({batch, heads, 1, dim})
      .set_stride(q_stride);

  cudnnStatus_t stream_status = cudnnSetStream(session->cudnn, session->stream);
  if (stream_status != CUDNN_STATUS_SUCCESS) {
    delete plan;
    return cudnn_to_cuda(stream_status);
  }
  auto status = plan->graph->build(session->cudnn,
                                   {cudnn_frontend::HeurMode_t::A});
  if (status.is_bad()) {
    log_cudnn_decode_status("build", status);
    delete plan;
    return cudaErrorNotSupported;
  }
  const int64_t workspace = plan->graph->get_workspace_size();
  if (workspace < 0 ||
      static_cast<uint64_t>(workspace) > kCublasWorkspaceBytes) {
    delete plan;
    return cudaErrorMemoryAllocation;
  }
  plan->workspace_bytes = static_cast<size_t>(workspace);
  if (cudnn_decode_debug_enabled()) {
    fprintf(stderr,
            "[nerva-cudnn-decode] build ok max_context=%u kv_capacity=%u "
            "heads=%u kv_heads=%u head_dim=%u workspace=%zu\n",
            session->max_context_tokens, session->kv_token_capacity,
            session->heads, session->kv_heads, session->head_dim,
            plan->workspace_bytes);
  }
  delete session->cudnn_decode_sdpa;
  session->cudnn_decode_sdpa = plan;
  return cudaSuccess;
}

cudaError_t execute_cudnn_decode_sdpa(
    NervaCudaHfDecodeSequenceSession *session, uint32_t layer_index) {
  cudaError_t err = ensure_cudnn_decode_sdpa_plan(session);
  if (err != cudaSuccess) {
    return err;
  }
  constexpr int64_t kTensorQ = 9101;
  constexpr int64_t kTensorK = 9102;
  constexpr int64_t kTensorV = 9103;
  constexpr int64_t kTensorO = 9104;
  constexpr int64_t kTensorSeqLenQ = 9105;
  constexpr int64_t kTensorSeqLenKv = 9106;
  const uint64_t kv_hidden =
      static_cast<uint64_t>(session->kv_heads) * session->head_dim;
  const uint64_t layer_kv_elements =
      static_cast<uint64_t>(session->kv_token_capacity) * kv_hidden;
  uint16_t *layer_keys =
      session->device_kv_keys + layer_kv_elements * layer_index;
  uint16_t *layer_values =
      session->device_kv_values + layer_kv_elements * layer_index;
  std::unordered_map<int64_t, void *> tensors = {
      {kTensorQ, session->device_decode_q},
      {kTensorK, layer_keys},
      {kTensorV, layer_values},
      {kTensorO, session->device_projection_input},
      {kTensorSeqLenQ, session->device_decode_seq_len_q},
      {kTensorSeqLenKv, session->device_decode_seq_len_kv},
  };
  auto status = session->cudnn_decode_sdpa->graph->execute(
      session->cudnn, tensors, session->cublas_workspace);
  if (status.is_bad()) {
    log_cudnn_decode_status("execute", status);
    return cudaErrorLaunchFailure;
  }
  return cudaSuccess;
}
#endif
