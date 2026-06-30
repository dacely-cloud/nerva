struct NervaCudaHfDecodeSequenceSession {
  uint32_t dtype = 0;
  uint32_t hidden = 0;
  uint32_t heads = 0;
  uint32_t kv_heads = 0;
  uint32_t head_dim = 0;
  uint32_t head_threads = kHeadThreadsMax;
  uint32_t intermediate = 0;
  uint32_t vocab_size = 0;
  uint32_t layer_count = 0;
  uint32_t max_context_tokens = 0;
  uint32_t kv_block_count = 0;
  uint32_t kv_token_capacity = 0;
  uint32_t prefill_chunk_tokens = 0;
  uint32_t detailed_profile = 0;
  uint32_t experimental_rt_decode_requested = 0;
  uint32_t experimental_rt_decode_enabled = 0;
  uint32_t experimental_rt_mode = kExperimentalRtModeAuto;
  uint32_t experimental_rt_page_tokens = 0;
  uint32_t experimental_rt_pages = 0;
  uint32_t experimental_rt_local_window_tokens = 0;
  uint32_t experimental_rt_sink_tokens = 0;
  uint32_t experimental_rt_query_count = 0;
  uint64_t experimental_rt_candidate_pages_bytes = 0;
  uint64_t experimental_rt_shadow_launches = 0;
  uint32_t experimental_rt_sparse_attention_active = 0;
  uint32_t experimental_rt_selector_cache_valid = 0;
  uint32_t experimental_rt_selector_cached_active_pages = 0;
  uint32_t experimental_rt_selector_cached_current_page = 0;
  uint32_t experimental_rt_selector_cached_local_pages = 0;
  uint32_t experimental_rt_selector_cached_sink_pages = 0;
  void *experimental_rt_selector = nullptr;
  uint32_t *device_experimental_rt_candidate_pages = nullptr;
  uint32_t experimental_rt_query_key_selector = 0;
  uint32_t experimental_prefill_local_window_tokens = 0;
  float rms_eps = 0.0f;
  float rope_theta = 0.0f;
  SequenceArenaLayout arena_layout{};
  uint64_t arena_bytes = 0;
  uint64_t resident_weight_bytes = 0;
  uint64_t layout_bytes = 0;
  uint64_t scratch_bytes = 0;
  uint64_t projection_input_bytes = 0;
  uint64_t projection_batch_input_bytes = 0;
  uint64_t projection_batch_output_bytes = 0;
  uint64_t prefill_hidden_bytes = 0;
  uint64_t prefill_norm_bytes = 0;
  uint64_t prefill_qkv_bytes = 0;
  uint64_t prefill_qkv_encoded_bytes = 0;
  uint64_t prefill_attn_bytes = 0;
  uint64_t prefill_o_bytes = 0;
  uint64_t prefill_q_gate_bytes = 0;
  uint64_t prefill_gate_up_bytes = 0;
  uint64_t prefill_ff_bytes = 0;
  uint64_t prefill_down_bytes = 0;
  uint64_t decode_attention_values_bytes = 0;
  uint64_t decode_attention_stats_bytes = 0;
  uint32_t decode_attention_max_chunks = 0;
  uint64_t decode_q_bytes = 0;
  uint64_t decode_seq_len_bytes = 0;
  uint64_t linear_gdn_conv_state_bytes = 0;
  uint64_t linear_gdn_recurrent_state_bytes = 0;
  uint64_t packed_qkv_bytes = 0;
  uint64_t packed_gate_up_bytes = 0;
  uint64_t kv_bytes = 0;
  uint64_t kv_block_table_bytes = 0;
  uint64_t slots_bytes = 0;
  uint64_t prompt_bytes = 0;
  uint64_t h2d_bytes = 0;
  uint64_t load_staging_bytes = 0;
  uint64_t setup_sync_calls = 0;
  uint64_t descriptor_gpu_resident_h2d_bytes = 0;
  uint64_t descriptor_gpu_staged_h2d_bytes = 0;
  uint32_t planned_weight_blocks = 0;
  uint32_t planned_gpu_resident_blocks = 0;
  uint32_t planned_gpu_staged_blocks = 0;
  uint64_t planned_weight_bytes = 0;
  uint64_t planned_gpu_resident_weight_bytes = 0;
  uint64_t planned_gpu_staged_weight_bytes = 0;
  uint32_t planned_weight_descriptor_count = 0;
  uint64_t planned_weight_descriptor_hash = 0;
  std::vector<SequenceLayerLayout> host_layouts;
  uint16_t *device_arena = nullptr;
  SequenceLayerLayout *device_layouts = nullptr;
  float *device_scratch = nullptr;
  uint16_t *device_projection_input = nullptr;
  uint16_t *device_projection_batch_input = nullptr;
  float *device_projection_batch_output = nullptr;
  uint16_t *device_prefill_hidden_a = nullptr;
  uint16_t *device_prefill_hidden_b = nullptr;
  uint16_t *device_prefill_norm = nullptr;
  float *device_prefill_qkv = nullptr;
  uint16_t *device_prefill_qkv_encoded = nullptr;
  uint16_t *device_prefill_attn = nullptr;
  float *device_prefill_o = nullptr;
  float *device_prefill_q_gate = nullptr;
  float *device_prefill_gate_up = nullptr;
  uint16_t *device_prefill_ff = nullptr;
  float *device_prefill_down = nullptr;
  float *device_decode_attention_values = nullptr;
  float *device_decode_attention_m = nullptr;
  float *device_decode_attention_l = nullptr;
  uint16_t *device_decode_q = nullptr;
  int32_t *device_decode_seq_len_q = nullptr;
  int32_t *device_decode_seq_len_kv = nullptr;
  float *device_linear_gdn_conv_state = nullptr;
  float *device_linear_gdn_recurrent_state = nullptr;
  uint16_t *device_qkv_packed = nullptr;
  uint16_t *device_gate_up_packed = nullptr;
  uint16_t *device_kv_keys = nullptr;
  uint16_t *device_kv_values = nullptr;
  uint32_t *device_kv_block_table = nullptr;
  uint32_t *device_prompt_tokens = nullptr;
  NervaCudaSyntheticTokenSlot *host_slots = nullptr;
  NervaCudaSyntheticTokenSlot *device_slots = nullptr;
  uint32_t *device_step = nullptr;
  void *cublas_workspace = nullptr;
  std::shared_ptr<SessionSharedWeights> shared_weights;
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas = nullptr;
  cublasLtHandle_t cublas_lt = nullptr;
#if NERVA_HAVE_CUDNN_FRONTEND
  cudnnHandle_t cudnn = nullptr;
  CudnnPrefillSdpaPlan *cudnn_prefill_sdpa = nullptr;
  uint32_t cudnn_prefill_sdpa_disabled = 0;
  CudnnDecodeSdpaPlan *cudnn_decode_sdpa = nullptr;
  uint32_t cudnn_decode_sdpa_disabled = 0;
#endif
  LtGemvPlan qkv_plan;
  LtGemvPlan attention_output_plan;
  LtGemvPlan gate_up_plan;
  LtGemvPlan down_plan;
  LtGemvPlan lm_head_plan;
  std::vector<LtGemmTokensPlan> projection_block_plans;
  cudaEvent_t device_start = nullptr;
  cudaEvent_t device_stop = nullptr;
  cudaEvent_t profile_start = nullptr;
  cudaEvent_t profile_stop = nullptr;
  cudaGraph_t cached_graph = nullptr;
  cudaGraphExec_t cached_graph_exec = nullptr;
  uint32_t cached_context_steps = 0;
  uint32_t cached_prompt_token_count = 0;
  uint32_t cached_has_eos_token = 0;
  uint32_t cached_eos_token = 0;
  uint32_t cached_attention_chunks = 0;
  uint32_t cached_experimental_rt_sparse_attention_active = 0;
  NervaCudaHfDecodeSamplerConfig cached_sampler = {};
  uint32_t projection_batch_peer_streams_synchronized = 0;
  uint32_t projection_batch_defer_layer_sync = 0;
  uint32_t projection_batch_own_stream_synchronized = 0;
  uint64_t cached_graph_nodes = 0;
  uint64_t cached_projection_ns = 0;
  uint64_t cached_qkv_projection_ns = 0;
  uint64_t cached_attention_output_projection_ns = 0;
  uint64_t cached_gate_up_projection_ns = 0;
  uint64_t cached_down_projection_ns = 0;
  uint64_t cached_lm_head_projection_ns = 0;
  uint64_t cached_attention_ns = 0;
  uint64_t cached_mlp_ns = 0;
  uint64_t cached_norm_ns = 0;
  uint64_t cached_sampling_ns = 0;
  uint64_t pending_prefill_kernel_launches = 0;
  uint64_t pending_prefill_device_elapsed_ns = 0;
  uint64_t pending_prefill_sync_calls = 0;
  uint64_t pending_prefill_graph_replays = 0;
  uint64_t pending_prefill_graph_launches = 0;
  uint64_t pending_prefill_graph_nodes = 0;
  uint32_t pending_prefill_available = 0;
  uint32_t active_prompt_token_count = 0;
  uint32_t active_has_eos_token = 0;
  uint32_t active_eos_token = 0;
  uint32_t active_seed_token = 0;
  NervaCudaHfDecodeSamplerConfig active_sampler = {};
  uint32_t active_observed_tokens = 0;
  uint32_t active_cursor = 0;
  bool active_started = false;
  bool active_finished = false;
};

struct ScopedProjectionBatchFlags {
  NervaCudaHfDecodeSequenceSession *session = nullptr;
  uint32_t previous_peer_streams_synchronized = 0;
  uint32_t previous_defer_layer_sync = 0;

  ScopedProjectionBatchFlags(NervaCudaHfDecodeSequenceSession *session_arg,
                             bool peer_streams_synchronized,
                             bool defer_layer_sync)
      : session(session_arg) {
    if (session == nullptr) {
      return;
    }
    previous_peer_streams_synchronized =
        session->projection_batch_peer_streams_synchronized;
    previous_defer_layer_sync = session->projection_batch_defer_layer_sync;
    if (peer_streams_synchronized) {
      session->projection_batch_peer_streams_synchronized = 1;
    }
    if (defer_layer_sync) {
      session->projection_batch_defer_layer_sync = 1;
    }
  }

  ~ScopedProjectionBatchFlags() {
    if (session == nullptr) {
      return;
    }
    session->projection_batch_peer_streams_synchronized =
        previous_peer_streams_synchronized;
    session->projection_batch_defer_layer_sync = previous_defer_layer_sync;
  }
};
