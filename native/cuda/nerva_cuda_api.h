#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NervaCudaDeviceSmokeResult {
  int32_t status;
  int32_t cuda_error;
  uint32_t value;
  int32_t device_count;
  int32_t device_ordinal;
  int32_t driver_version;
  int32_t runtime_version;
  int32_t compute_capability_major;
  int32_t compute_capability_minor;
  int32_t posix_fd_handle_supported;
  int32_t vmm_posix_fd_export_verified;
  int32_t gpu_direct_rdma_supported;
  int32_t gpu_direct_rdma_with_cuda_vmm_supported;
  uint64_t total_global_mem;
  uint64_t free_global_mem;
  char gpu_name[128];
  char pci_bus_id[32];
} NervaCudaDeviceSmokeResult;

typedef struct NervaCudaSyntheticTokenSlot {
  uint32_t request_id;
  uint32_t sequence_id;
  uint64_t token_index;
  uint32_t token;
  uint64_t version;
  uint32_t completion;
  uint32_t host_copied;
} NervaCudaSyntheticTokenSlot;

typedef struct NervaCudaSyntheticGraphResult {
  int32_t status;
  int32_t cuda_error;
  uint32_t steps;
  uint32_t ring_capacity;
  uint32_t seed_token;
  uint32_t last_token;
  uint64_t graph_replays;
  uint64_t graph_nodes;
  uint64_t observed_tokens;
  uint64_t observed_token_hash;
  uint64_t token_ring_slots_touched;
  uint64_t token_ring_reuses;
  uint64_t token_ring_max_slot_version;
  uint64_t stale_tokens;
  uint64_t missing_tokens;
  uint64_t extra_tokens;
  uint64_t mismatched_tokens;
  uint64_t host_causality_edges;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t graph_launches;
  uint64_t sync_calls;
  uint64_t d2h_bytes;
  uint64_t hot_path_allocations;
} NervaCudaSyntheticGraphResult;

typedef struct NervaCudaTinyBlockResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t hidden;
  uint32_t intermediate;
  uint16_t output[2];
  uint64_t output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t d2h_bytes;
  uint64_t hot_path_allocations;
} NervaCudaTinyBlockResult;

typedef struct NervaCudaLoadedTinyBlockResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t hidden;
  uint32_t intermediate;
  uint16_t output[2];
  uint64_t output_hash;
  uint64_t resident_weight_bytes;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaLoadedTinyBlockResult;

typedef struct NervaCudaBlockForwardRequest {
  uint32_t dtype;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint32_t position;
  float rms_eps;
  float rope_theta;
  const uint16_t *input;
  const uint16_t *rms_attn_weight;
  const uint16_t *rms_mlp_weight;
  const uint16_t *w_q;
  const uint16_t *w_k;
  const uint16_t *w_v;
  const uint16_t *w_o;
  const uint16_t *q_bias;
  const uint16_t *k_bias;
  const uint16_t *v_bias;
  const uint16_t *o_bias;
  const uint16_t *w_gate;
  const uint16_t *w_up;
  const uint16_t *w_down;
  uint16_t *output;
} NervaCudaBlockForwardRequest;

typedef struct NervaCudaBlockForwardResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t dtype;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint64_t output_hash;
  uint64_t resident_weight_bytes;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaBlockForwardResult;

typedef struct NervaCudaGreedySamplerResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t vocab_size;
  uint64_t token_index;
  uint32_t token;
  uint64_t slot_version;
  uint32_t completion;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaGreedySamplerResult;

typedef struct NervaCudaHfSamplerRequest {
  uint32_t dtype;
  uint32_t hidden;
  uint32_t vocab_size;
  uint64_t token_index;
  float rms_eps;
  const uint16_t *hidden_bits;
  const uint16_t *final_norm_weight;
  const uint16_t *lm_head;
} NervaCudaHfSamplerRequest;

typedef struct NervaCudaHfSamplerResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t dtype;
  uint32_t hidden;
  uint32_t vocab_size;
  uint64_t token_index;
  uint32_t token;
  uint64_t slot_version;
  uint32_t completion;
  uint64_t output_hash;
  uint64_t resident_weight_bytes;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaHfSamplerResult;

typedef struct NervaCudaHfDecodeStepRequest {
  uint32_t dtype;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint32_t vocab_size;
  uint32_t position;
  uint64_t token_index;
  float rms_eps;
  float rope_theta;
  const uint16_t *input;
  const uint16_t *rms_attn_weight;
  const uint16_t *rms_mlp_weight;
  const uint16_t *w_q;
  const uint16_t *w_k;
  const uint16_t *w_v;
  const uint16_t *w_o;
  const uint16_t *q_bias;
  const uint16_t *k_bias;
  const uint16_t *v_bias;
  const uint16_t *o_bias;
  const uint16_t *w_gate;
  const uint16_t *w_up;
  const uint16_t *w_down;
  const uint16_t *final_norm_weight;
  const uint16_t *lm_head;
} NervaCudaHfDecodeStepRequest;

typedef struct NervaCudaHfDecodeStepResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t dtype;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint32_t vocab_size;
  uint64_t token_index;
  uint32_t token;
  uint64_t slot_version;
  uint32_t completion;
  uint64_t output_hash;
  uint64_t resident_weight_bytes;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaHfDecodeStepResult;

typedef struct NervaCudaHfDecodeChainLayer {
  const uint16_t *rms_attn_weight;
  const uint16_t *rms_mlp_weight;
  const uint16_t *w_q;
  const uint16_t *w_q_gate;
  const uint16_t *w_k;
  const uint16_t *q_norm_weight;
  const uint16_t *k_norm_weight;
  const uint16_t *w_v;
  const uint16_t *w_o;
  const uint16_t *q_bias;
  const uint16_t *k_bias;
  const uint16_t *v_bias;
  const uint16_t *o_bias;
  const uint16_t *w_gate;
  const uint16_t *w_up;
  const uint16_t *w_down;
  const uint16_t *w_router;
  const uint16_t *w_expert_gate_up;
  const uint16_t *w_expert_down;
  const uint16_t *w_shared_expert_gate;
  const uint16_t *w_shared_expert_up;
  const uint16_t *w_shared_expert_down;
  const uint16_t *w_shared_expert_router;
  uint32_t linear_key_heads;
  uint32_t linear_value_heads;
  uint32_t linear_key_head_dim;
  uint32_t linear_value_head_dim;
  uint32_t linear_conv_kernel;
  const uint16_t *w_linear_conv;
  const uint16_t *w_linear_qkv;
  const uint16_t *w_linear_z;
  const uint16_t *w_linear_b;
  const uint16_t *w_linear_a;
  const uint16_t *w_linear_dt_bias;
  const float *w_linear_a_log;
  const uint16_t *w_linear_norm;
  const uint16_t *w_linear_out;
  uint32_t mlp_kind;
  uint32_t moe_intermediate;
  uint32_t shared_expert_intermediate;
  uint32_t num_experts;
  uint32_t experts_per_token;
  uint32_t norm_topk_prob;
  uint32_t attention_kind;
  uint32_t deepseek_mode;
  uint32_t deepseek_flags;
  uint32_t deepseek_hc_mult;
  uint32_t deepseek_q_lora_rank;
  uint32_t deepseek_kv_lora_rank;
  uint32_t deepseek_o_lora_rank;
  uint32_t deepseek_o_groups;
  uint32_t deepseek_qk_nope_head_dim;
  uint32_t deepseek_qk_rope_head_dim;
  uint32_t deepseek_v_head_dim;
  uint32_t deepseek_compress_ratio;
  uint32_t deepseek_index_topk;
  uint32_t deepseek_index_n_heads;
  uint32_t deepseek_index_head_dim;
  uint32_t deepseek_router_num_groups;
  uint32_t deepseek_router_topk_groups;
  float deepseek_routed_scaling_factor;
} NervaCudaHfDecodeChainLayer;

typedef struct NervaCudaHfDecodeChainRequest {
  uint32_t dtype;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint32_t vocab_size;
  uint32_t layer_count;
  uint32_t position;
  uint64_t token_index;
  float rms_eps;
  float rope_theta;
  const uint16_t *input;
  const NervaCudaHfDecodeChainLayer *layers;
  const uint16_t *final_norm_weight;
  const uint16_t *lm_head;
} NervaCudaHfDecodeChainRequest;

typedef struct NervaCudaHfDecodeChainResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t dtype;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint32_t vocab_size;
  uint32_t layer_count;
  uint64_t token_index;
  uint32_t token;
  uint64_t slot_version;
  uint32_t completion;
  uint64_t output_hash;
  uint64_t resident_weight_bytes;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaHfDecodeChainResult;

typedef struct NervaCudaHfDecodeSequenceWeightBlock {
  const uint16_t *host_source;
  const char *source_file;
  uint64_t source_file_len;
  uint64_t file_offset_begin;
  uint64_t block_id;
  uint64_t block_version;
  uint64_t offset_bytes;
  uint64_t bytes;
  uint32_t strategy;
  uint32_t reserved;
} NervaCudaHfDecodeSequenceWeightBlock;

typedef struct NervaCudaHfDecodeSamplerConfig {
  float temperature;
  float top_p;
  uint32_t top_k;
  uint32_t reserved;
  uint64_t seed;
} NervaCudaHfDecodeSamplerConfig;

typedef struct NervaCudaHfDecodeSequenceRequest {
  uint32_t dtype;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint32_t vocab_size;
  uint32_t layer_count;
  uint32_t steps;
  uint32_t seed_token;
  const uint32_t *prompt_tokens;
  uint32_t prompt_token_count;
  uint32_t has_eos_token;
  uint32_t eos_token;
  float rms_eps;
  float rope_theta;
  const uint16_t *embeddings;
  const NervaCudaHfDecodeChainLayer *layers;
  const uint16_t *final_norm_weight;
  const uint16_t *lm_head;
  uint32_t planned_weight_blocks;
  uint32_t planned_gpu_resident_blocks;
  uint32_t planned_gpu_staged_blocks;
  uint64_t planned_weight_bytes;
  uint64_t planned_gpu_resident_weight_bytes;
  uint64_t planned_gpu_staged_weight_bytes;
  const NervaCudaHfDecodeSequenceWeightBlock *planned_weight_descriptors;
  uint32_t planned_weight_descriptor_count;
  uint64_t planned_weight_descriptor_hash;
  uint32_t *output_tokens;
  uint32_t output_token_capacity;
  NervaCudaHfDecodeSamplerConfig sampler;
} NervaCudaHfDecodeSequenceRequest;

typedef struct NervaCudaHfDecodeSequenceResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t dtype;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint32_t vocab_size;
  uint32_t layer_count;
  uint32_t steps;
  uint32_t seed_token;
  uint32_t observed_tokens;
  uint32_t last_token;
  uint64_t observed_token_hash;
  uint64_t resident_weight_bytes;
  uint32_t planned_weight_blocks;
  uint32_t planned_gpu_resident_blocks;
  uint32_t planned_gpu_staged_blocks;
  uint64_t planned_weight_bytes;
  uint64_t planned_gpu_resident_weight_bytes;
  uint64_t planned_gpu_staged_weight_bytes;
  uint64_t descriptor_gpu_resident_h2d_bytes;
  uint64_t descriptor_gpu_staged_h2d_bytes;
  uint32_t planned_weight_descriptor_count;
  uint64_t planned_weight_descriptor_hash;
  uint64_t resident_kv_bytes;
  uint64_t kv_tokens;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t graph_replays;
  uint64_t graph_nodes;
  uint64_t graph_launches;
  uint64_t graph_captures;
  uint64_t graph_cache_hits;
  uint64_t kernel_launches;
  uint64_t experimental_rt_selector_launches;
  uint32_t experimental_rt_sparse_attention_active;
  uint32_t experimental_rt_dense_attention_chunks;
  uint32_t experimental_rt_attention_chunks;
  uint32_t experimental_rt_reserved;
  uint64_t device_elapsed_ns;
  uint64_t projection_ns;
  uint64_t qkv_projection_ns;
  uint64_t attention_output_projection_ns;
  uint64_t gate_up_projection_ns;
  uint64_t down_projection_ns;
  uint64_t lm_head_projection_ns;
  uint64_t attention_ns;
  uint64_t mlp_ns;
  uint64_t norm_ns;
  uint64_t sampling_ns;
  uint64_t sync_calls;
  uint64_t host_causality_edges;
  uint64_t hot_path_allocations;
  uint64_t deepseek_compressor_state_writes;
  uint64_t deepseek_compressed_kv_writes;
  uint64_t deepseek_indexer_state_writes;
  uint64_t deepseek_indexer_kv_writes;
  uint64_t deepseek_compressed_kv_attention_reads;
  uint64_t deepseek_compressed_kv_attention_slots_scanned;
  uint64_t deepseek_sparse_topk_selections;
  uint64_t deepseek_sparse_topk_slots_selected;
  uint64_t deepseek_sparse_topk_candidates_scored;
  uint64_t deepseek_v3_grouped_router_selections;
  uint64_t deepseek_v4_bias_router_selections;
  uint64_t deepseek_v4_hash_router_selections;
  uint64_t deepseek_raw_attention_tokens_scanned;
} NervaCudaHfDecodeSequenceResult;

typedef struct NervaCudaHfDecodeSequenceLayoutPlanRequest {
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint32_t vocab_size;
  uint32_t layer_count;
  uint32_t layer_index;
  const NervaCudaHfDecodeChainLayer *layers;
} NervaCudaHfDecodeSequenceLayoutPlanRequest;

typedef struct NervaCudaHfDecodeSequenceLayoutPlanResult {
  int32_t status;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint32_t vocab_size;
  uint32_t layer_count;
  uint32_t layer_index;
  uint32_t attention_kind;
  uint32_t deepseek_mode;
  uint32_t deepseek_flags;
  uint32_t deepseek_qk_head_dim;
  uint32_t deepseek_q_rows;
  uint32_t deepseek_kv_cache_width;
  uint32_t deepseek_kv_b_rows;
  uint32_t deepseek_value_rows;
  uint64_t resident_weight_bytes;
  uint64_t layout_bytes;
  uint64_t rms_attn;
  uint64_t rms_mlp;
  uint64_t w_q;
  uint64_t q_norm;
  uint64_t w_k;
  uint64_t k_norm;
  uint64_t w_v;
  uint64_t w_o;
  uint64_t w_router;
  uint64_t w_expert_gate_up;
  uint64_t w_expert_down;
  uint64_t deepseek_q_a_scale;
  uint64_t deepseek_q_b;
  uint64_t deepseek_q_b_scale;
  uint64_t deepseek_kv_a_scale;
  uint64_t deepseek_kv_b_scale;
  uint64_t deepseek_o_a_scale;
  uint64_t deepseek_o_b;
  uint64_t deepseek_o_b_scale;
  uint64_t deepseek_attention_sink;
  uint64_t deepseek_indexer_q;
  uint64_t deepseek_indexer_q_scale;
  uint64_t deepseek_indexer_k;
  uint64_t deepseek_indexer_k_scale;
  uint64_t deepseek_indexer_k_norm;
  uint64_t deepseek_indexer_k_norm_bias;
  uint64_t deepseek_indexer_weights;
  uint64_t deepseek_compressor_ape;
  uint64_t deepseek_compressor_wkv;
  uint64_t deepseek_compressor_wgate;
  uint64_t deepseek_compressor_norm;
  uint64_t deepseek_indexer_compressor_ape;
  uint64_t deepseek_indexer_compressor_wkv;
  uint64_t deepseek_indexer_compressor_wgate;
  uint64_t deepseek_indexer_compressor_norm;
} NervaCudaHfDecodeSequenceLayoutPlanResult;

typedef struct NervaCudaHfDecodeSequenceSession NervaCudaHfDecodeSequenceSession;

typedef struct NervaCudaHfDecodeSequenceSessionCreateRequest {
  uint32_t dtype;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint32_t vocab_size;
  uint32_t layer_count;
  uint32_t max_context_tokens;
  float rms_eps;
  float rope_theta;
  const uint16_t *embeddings;
  const NervaCudaHfDecodeChainLayer *layers;
  const uint16_t *final_norm_weight;
  const uint16_t *lm_head;
  uint32_t planned_weight_blocks;
  uint32_t planned_gpu_resident_blocks;
  uint32_t planned_gpu_staged_blocks;
  uint64_t planned_weight_bytes;
  uint64_t planned_gpu_resident_weight_bytes;
  uint64_t planned_gpu_staged_weight_bytes;
  const NervaCudaHfDecodeSequenceWeightBlock *planned_weight_descriptors;
  uint32_t planned_weight_descriptor_count;
  uint64_t planned_weight_descriptor_hash;
  uint32_t detailed_profile;
  uint32_t experimental_rt_decode;
  uint32_t experimental_rt_mode;
  uint32_t experimental_rt_page_tokens;
  uint32_t experimental_rt_pages;
  uint32_t experimental_rt_local_window_tokens;
  uint32_t experimental_rt_sink_tokens;
} NervaCudaHfDecodeSequenceSessionCreateRequest;

typedef struct NervaCudaHfDecodeSequenceSessionCreateResult {
  int32_t status;
  int32_t cuda_error;
  int32_t failure_stage;
  int32_t device_count;
  uint32_t dtype;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint32_t vocab_size;
  uint32_t layer_count;
  uint32_t max_context_tokens;
  uint32_t prefill_chunk_tokens;
  uint32_t head_threads;
  uint64_t resident_weight_bytes;
  uint32_t planned_weight_blocks;
  uint32_t planned_gpu_resident_blocks;
  uint32_t planned_gpu_staged_blocks;
  uint64_t planned_weight_bytes;
  uint64_t planned_gpu_resident_weight_bytes;
  uint64_t planned_gpu_staged_weight_bytes;
  uint64_t descriptor_gpu_resident_h2d_bytes;
  uint64_t descriptor_gpu_staged_h2d_bytes;
  uint32_t planned_weight_descriptor_count;
  uint64_t planned_weight_descriptor_hash;
  uint32_t experimental_rt_decode_requested;
  uint32_t experimental_rt_decode_enabled;
  uint32_t experimental_rt_mode;
  uint32_t experimental_rt_page_tokens;
  uint32_t experimental_rt_pages;
  uint32_t experimental_rt_local_window_tokens;
  uint32_t experimental_rt_sink_tokens;
  uint32_t deepseek_v4_attention_aux_streams;
  uint32_t deepseek_v4_attention_events;
  uint64_t deepseek_v4_swa_kv_bytes;
  uint64_t resident_kv_bytes;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaHfDecodeSequenceSessionCreateResult;

typedef struct NervaCudaHfDecodeSequenceSessionForkSharedWeightsRequest {
  NervaCudaHfDecodeSequenceSession *parent;
  uint32_t detailed_profile;
} NervaCudaHfDecodeSequenceSessionForkSharedWeightsRequest;

typedef struct NervaCudaHfDecodeSequenceSessionRunRequest {
  NervaCudaHfDecodeSequenceSession *session;
  uint32_t steps;
  uint32_t seed_token;
  const uint32_t *prompt_tokens;
  uint32_t prompt_token_count;
  uint32_t has_eos_token;
  uint32_t eos_token;
  uint32_t *output_tokens;
  uint32_t output_token_capacity;
  NervaCudaHfDecodeSamplerConfig sampler;
} NervaCudaHfDecodeSequenceSessionRunRequest;

typedef struct NervaCudaHfDecodeSequenceSessionStartRequest {
  NervaCudaHfDecodeSequenceSession *session;
  const uint32_t *prompt_tokens;
  uint32_t prompt_token_count;
  uint32_t has_eos_token;
  uint32_t eos_token;
  NervaCudaHfDecodeSamplerConfig sampler;
} NervaCudaHfDecodeSequenceSessionStartRequest;

typedef struct NervaCudaHfDecodeSequenceSessionAdvanceRequest {
  NervaCudaHfDecodeSequenceSession *session;
  uint32_t steps;
  uint32_t *output_tokens;
  uint32_t output_token_capacity;
} NervaCudaHfDecodeSequenceSessionAdvanceRequest;

typedef struct NervaCudaHfDecodeSequenceDeepSeekV4SwaKvSnapshotRequest {
  NervaCudaHfDecodeSequenceSession *session;
  uint32_t layer_index;
  uint8_t *output_bytes;
  uint64_t output_byte_capacity;
} NervaCudaHfDecodeSequenceDeepSeekV4SwaKvSnapshotRequest;

typedef struct NervaCudaHfDecodeSequenceDeepSeekV4SwaKvSnapshotResult {
  int32_t status;
  int32_t cuda_error;
  uint32_t layer_index;
  uint32_t block_count;
  uint64_t layer_offset_bytes;
  uint64_t layer_bytes;
  uint64_t page_bytes;
  uint64_t copied_bytes;
  uint64_t output_hash;
} NervaCudaHfDecodeSequenceDeepSeekV4SwaKvSnapshotResult;

typedef struct NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotRequest {
  NervaCudaHfDecodeSequenceSession *session;
  uint32_t layer_index;
  uint8_t *output_bytes;
  uint64_t output_byte_capacity;
} NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotRequest;

typedef struct NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotResult {
  int32_t status;
  int32_t cuda_error;
  uint32_t layer_index;
  uint32_t block_count;
  uint64_t layer_offset_bytes;
  uint64_t layer_bytes;
  uint64_t page_bytes;
  uint64_t copied_bytes;
  uint64_t output_hash;
} NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotResult;

typedef struct NervaCudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshotRequest {
  NervaCudaHfDecodeSequenceSession *session;
  uint32_t layer_index;
  uint8_t *output_bytes;
  uint64_t output_byte_capacity;
} NervaCudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshotRequest;

typedef struct NervaCudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshotResult {
  int32_t status;
  int32_t cuda_error;
  uint32_t layer_index;
  uint32_t block_count;
  uint64_t layer_offset_bytes;
  uint64_t layer_bytes;
  uint64_t page_bytes;
  uint64_t copied_bytes;
  uint64_t output_hash;
} NervaCudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshotResult;

typedef struct NervaCudaHfDecodeSequenceProjectionBatchPlanRequest {
  NervaCudaHfDecodeSequenceSession **sessions;
  uint32_t session_count;
  uint32_t target_block_tokens;
  uint32_t min_block_tokens;
} NervaCudaHfDecodeSequenceProjectionBatchPlanRequest;

typedef struct NervaCudaHfDecodeSequenceProjectionBatchPlanResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t reason;
  uint32_t exact;
  uint32_t requested_session_count;
  uint32_t eligible_session_count;
  uint32_t block_tokens;
  uint32_t target_block_tokens;
  uint32_t min_block_tokens;
  uint32_t dtype;
  uint32_t hidden;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t intermediate;
  uint32_t vocab_size;
  uint32_t layer_count;
  uint32_t max_context_tokens;
  uint64_t planned_weight_descriptor_hash;
  uint64_t resident_weight_bytes;
  uint64_t qkv_rows;
  uint64_t gate_up_rows;
  uint64_t qkv_input_bytes;
  uint64_t qkv_output_bytes;
  uint64_t attention_output_input_bytes;
  uint64_t attention_output_output_bytes;
  uint64_t gate_up_input_bytes;
  uint64_t gate_up_output_bytes;
  uint64_t down_input_bytes;
  uint64_t down_output_bytes;
  uint64_t lm_head_input_bytes;
  uint64_t lm_head_output_bytes;
  uint64_t pack_input_bytes;
  uint64_t max_projection_output_bytes;
  uint64_t hot_path_allocations;
} NervaCudaHfDecodeSequenceProjectionBatchPlanResult;

#define NERVA_CUDA_PROJECTION_BATCH_KIND_QKV 1u
#define NERVA_CUDA_PROJECTION_BATCH_KIND_ATTENTION_OUTPUT 2u
#define NERVA_CUDA_PROJECTION_BATCH_KIND_GATE_UP 3u
#define NERVA_CUDA_PROJECTION_BATCH_KIND_DOWN 4u
#define NERVA_CUDA_PROJECTION_BATCH_KIND_LM_HEAD 5u

typedef struct NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest {
  NervaCudaHfDecodeSequenceSession **sessions;
  uint32_t session_count;
  uint32_t target_block_tokens;
  uint32_t min_block_tokens;
  uint32_t projection_kind;
  uint32_t layer_index;
} NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest;

typedef struct NervaCudaHfDecodeSequenceProjectionBatchExecuteResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t reason;
  uint32_t exact;
  uint32_t projection_kind;
  uint32_t layer_index;
  uint32_t requested_session_count;
  uint32_t eligible_session_count;
  uint32_t block_tokens;
  uint32_t target_block_tokens;
  uint32_t min_block_tokens;
  uint32_t dtype;
  uint32_t rows;
  uint32_t cols;
  uint64_t input_bytes;
  uint64_t output_bytes;
  uint64_t elapsed_ns;
  uint64_t pack_kernel_launches;
  uint64_t projection_kernel_launches;
  uint64_t scatter_kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaHfDecodeSequenceProjectionBatchExecuteResult;

typedef struct NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteRequest {
  NervaCudaHfDecodeSequenceSession **sessions;
  uint32_t session_count;
  uint32_t target_block_tokens;
  uint32_t min_block_tokens;
  uint32_t layer_index;
} NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteRequest;

typedef struct NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t reason;
  uint32_t exact;
  uint32_t layer_index;
  uint32_t requested_session_count;
  uint32_t eligible_session_count;
  uint32_t block_tokens;
  uint32_t target_block_tokens;
  uint32_t min_block_tokens;
  uint32_t dtype;
  uint32_t qkv_rows;
  uint32_t attention_output_rows;
  uint32_t gate_up_rows;
  uint32_t down_rows;
  uint32_t hidden_cols;
  uint32_t attention_output_cols;
  uint32_t down_cols;
  uint64_t input_bytes;
  uint64_t output_bytes;
  uint64_t elapsed_ns;
  uint64_t qkv_elapsed_ns;
  uint64_t attention_output_elapsed_ns;
  uint64_t gate_up_elapsed_ns;
  uint64_t down_elapsed_ns;
  uint64_t pack_kernel_launches;
  uint64_t projection_kernel_launches;
  uint64_t scatter_kernel_launches;
  uint64_t dependency_kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult;

typedef struct NervaCudaHfDecodeSequenceBatchAdvanceRequest {
  NervaCudaHfDecodeSequenceSession **sessions;
  uint32_t session_count;
  uint32_t target_block_tokens;
  uint32_t min_block_tokens;
  uint32_t *output_tokens;
  uint32_t output_token_capacity;
} NervaCudaHfDecodeSequenceBatchAdvanceRequest;

typedef struct NervaCudaHfDecodeSequenceBatchAdvanceResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t reason;
  uint32_t exact;
  uint32_t requested_session_count;
  uint32_t eligible_session_count;
  uint32_t block_tokens;
  uint32_t target_block_tokens;
  uint32_t min_block_tokens;
  uint32_t dtype;
  uint32_t layer_count;
  uint32_t observed_tokens;
  uint32_t last_token;
  uint64_t observed_token_hash;
  uint64_t d2h_bytes;
  uint64_t projection_elapsed_ns;
  uint64_t qkv_elapsed_ns;
  uint64_t attention_output_elapsed_ns;
  uint64_t gate_up_elapsed_ns;
  uint64_t down_elapsed_ns;
  uint64_t lm_head_elapsed_ns;
  uint64_t pack_kernel_launches;
  uint64_t projection_kernel_launches;
  uint64_t scatter_kernel_launches;
  uint64_t dependency_kernel_launches;
  uint64_t sampling_kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaHfDecodeSequenceBatchAdvanceResult;

typedef struct NervaCudaTinyDecodeResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t steps;
  uint32_t ring_capacity;
  uint32_t seed_token;
  uint32_t vocab_size;
  uint32_t hidden;
  uint32_t last_token;
  uint64_t graph_replays;
  uint64_t graph_nodes;
  uint64_t observed_tokens;
  uint64_t observed_token_hash;
  uint64_t token_ring_slots_touched;
  uint64_t token_ring_reuses;
  uint64_t token_ring_max_slot_version;
  uint64_t stale_tokens;
  uint64_t missing_tokens;
  uint64_t extra_tokens;
  uint64_t mismatched_tokens;
  uint64_t host_causality_edges;
  uint64_t resident_weight_bytes;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t graph_launches;
  uint64_t sync_calls;
  uint64_t kernel_launches;
  uint64_t hot_path_allocations;
  uint64_t token_ledgers;
  uint64_t graph_replay_events;
  uint64_t device_activity_events;
  uint64_t copy_events;
  uint64_t soft_visibility_syncs;
  uint64_t hard_syncs;
  uint64_t host_event_wait_ns;
  uint64_t gpu_active_ns;
  uint64_t gpu_idle_ns;
  uint64_t wall_latency_ns;
} NervaCudaTinyDecodeResult;

typedef struct NervaCudaTieredAttentionResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t hidden;
  uint32_t heads;
  uint32_t blocks;
  uint32_t tokens;
  float output[2];
  uint64_t output_hash;
  uint64_t cpu_block_events;
  uint64_t device_block_events;
  uint64_t resident_kv_bytes;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaTieredAttentionResult;

typedef struct NervaCudaBackendContractResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  int32_t device_ordinal;
  int32_t driver_version;
  int32_t runtime_version;
  int32_t compute_capability_major;
  int32_t compute_capability_minor;
  uint64_t total_global_mem;
  uint64_t free_global_mem;
  uint64_t requested_device_bytes;
  uint64_t requested_pinned_bytes;
  uint64_t allocated_device_bytes;
  uint64_t allocated_pinned_bytes;
  uint64_t stream_creations;
  uint64_t stream_destroys;
  uint64_t event_creations;
  uint64_t event_destroys;
  uint64_t device_allocations;
  uint64_t device_frees;
  uint64_t pinned_allocations;
  uint64_t pinned_frees;
  uint64_t memset_bytes;
  uint64_t d2h_bytes;
  uint64_t sync_calls;
  uint64_t observed_word;
  uint64_t hot_path_allocations;
  char gpu_name[128];
  char pci_bus_id[32];
} NervaCudaBackendContractResult;

typedef struct NervaCudaProjectionBenchRequest {
  uint32_t dtype;
  uint32_t rows;
  uint32_t cols;
  uint32_t iterations;
  uint32_t warmup_iterations;
  uint32_t block_tokens;
} NervaCudaProjectionBenchRequest;

typedef struct NervaCudaProjectionBenchResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  int32_t device_ordinal;
  int32_t compute_capability_major;
  int32_t compute_capability_minor;
  uint32_t dtype;
  uint32_t rows;
  uint32_t cols;
  uint32_t block_tokens;
  uint32_t iterations;
  uint32_t warmup_iterations;
  uint64_t matrix_bytes;
  uint64_t input_bytes;
  uint64_t output_bytes;
  uint64_t cublaslt_total_ns;
  uint64_t cublaslt_avg_ns;
  uint64_t cublaslt_default_total_ns;
  uint64_t cublaslt_default_avg_ns;
  uint32_t cublaslt_heuristic_count;
  uint32_t cublaslt_best_heuristic_index;
  uint64_t cublaslt_best_heuristic_total_ns;
  uint64_t cublaslt_best_heuristic_avg_ns;
  uint64_t custom_total_ns;
  uint64_t custom_avg_ns;
  uint64_t cublaslt_graph_total_ns;
  uint64_t cublaslt_graph_avg_ns;
  uint64_t cublaslt_default_graph_total_ns;
  uint64_t cublaslt_default_graph_avg_ns;
  uint64_t cublaslt_best_heuristic_graph_total_ns;
  uint64_t cublaslt_best_heuristic_graph_avg_ns;
  uint64_t custom_graph_total_ns;
  uint64_t custom_graph_avg_ns;
  uint64_t cublaslt_graph_nodes;
  uint64_t custom_graph_nodes;
  uint64_t graph_replays;
  uint64_t graph_captures;
  uint32_t selected_graph_strategy;
  uint64_t cublaslt_effective_bandwidth_bps;
  uint64_t custom_effective_bandwidth_bps;
  uint32_t selected_strategy;
  uint32_t mismatch_count;
  float max_abs_diff;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t device_allocations;
  uint64_t device_frees;
  uint64_t device_arena_bytes;
  uint64_t hot_path_allocations;
  uint64_t block_cublaslt_total_ns;
  uint64_t block_cublaslt_avg_ns;
  uint64_t block_cublaslt_per_token_ns;
  uint64_t block_cublaslt_graph_total_ns;
  uint64_t block_cublaslt_graph_avg_ns;
  uint64_t block_cublaslt_graph_per_token_ns;
  uint64_t block_cublaslt_graph_nodes;
  uint64_t block_cublaslt_speedup_x1000;
  uint64_t block_cublaslt_graph_speedup_x1000;
  uint64_t block_cublaslt_effective_bandwidth_bps;
} NervaCudaProjectionBenchResult;

typedef struct NervaCudaExperimentalRtCandidateBenchRequest {
  uint32_t pages;
  uint32_t page_tokens;
  uint32_t dims;
  uint32_t query_count;
  uint32_t candidates_per_query;
  uint32_t iterations;
  uint32_t warmup_iterations;
} NervaCudaExperimentalRtCandidateBenchRequest;

typedef struct NervaCudaExperimentalRtCandidateBenchResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  int32_t device_ordinal;
  int32_t compute_capability_major;
  int32_t compute_capability_minor;
  uint32_t pages;
  uint32_t page_tokens;
  uint32_t dims;
  uint32_t query_count;
  uint32_t candidates_per_query;
  uint32_t iterations;
  uint32_t warmup_iterations;
  uint32_t rt_core_capable;
  uint32_t real_rt_backend_available;
  uint32_t rt_headers_available;
  uint32_t optix_headers_available;
  uint32_t vulkan_headers_available;
  uint32_t vulkan_shader_compiler_available;
  uint32_t vulkan_loader_available;
  uint32_t vulkan_rt_extensions_available;
  uint32_t vulkan_physical_devices;
  uint64_t descriptor_bytes;
  uint64_t query_bytes;
  uint64_t kv_cache_bytes;
  uint64_t candidate_id_bytes;
  uint64_t output_bytes;
  uint64_t dense_selector_total_ns;
  uint64_t dense_selector_avg_ns;
  uint64_t software_selector_total_ns;
  uint64_t software_selector_avg_ns;
  uint64_t candidate_selector_total_ns;
  uint64_t candidate_selector_avg_ns;
  uint64_t rerank_total_ns;
  uint64_t rerank_avg_ns;
  uint64_t selector_plus_rerank_avg_ns;
  uint64_t dense_vs_selector_speedup_x1000;
  uint64_t dense_vs_selector_plus_rerank_speedup_x1000;
  uint64_t candidate_fraction_ppm;
  uint64_t candidate_parity_checked;
  uint64_t candidate_parity_mismatches;
  uint64_t candidate_parity_first_mismatch_index;
  uint64_t candidate_parity_first_expected;
  uint64_t candidate_parity_first_actual;
  uint64_t candidate_query_hashes_distinct;
  uint64_t candidate_query_hash_repeats;
  uint64_t local_window_tokens;
  uint64_t local_attention_total_ns;
  uint64_t local_attention_avg_ns;
  uint64_t kv_page_access_total_ns;
  uint64_t kv_page_access_avg_ns;
  uint64_t far_sparse_attention_total_ns;
  uint64_t far_sparse_attention_avg_ns;
  uint64_t softmax_merge_total_ns;
  uint64_t softmax_merge_avg_ns;
  uint64_t dense_full_attention_total_ns;
  uint64_t dense_full_attention_avg_ns;
  uint64_t attention_mass_recall_min_ppm;
  uint64_t attention_mass_recall_avg_ppm;
  uint64_t page_level_attention_mass_recall_min_ppm;
  uint64_t page_level_attention_mass_recall_avg_ppm;
  uint64_t far_oracle_topk_tokens;
  uint64_t far_oracle_topk_token_recall_min_ppm;
  uint64_t far_oracle_topk_token_recall_avg_ppm;
  uint64_t page_level_far_oracle_topk_token_recall_min_ppm;
  uint64_t page_level_far_oracle_topk_token_recall_avg_ppm;
  uint64_t far_oracle_topk_importance_scatter_min_pages;
  uint64_t far_oracle_topk_importance_scatter_avg_pages_x1000;
  uint64_t far_oracle_topk_importance_scatter_max_pages;
  uint64_t fine_token_projected_topk_tokens;
  uint64_t fine_token_projected_candidate_tokens;
  uint64_t fine_token_projected_token_recall_min_ppm;
  uint64_t fine_token_projected_token_recall_avg_ppm;
  uint64_t fine_token_learned_projected_topk_tokens;
  uint64_t fine_token_learned_projected_candidate_tokens;
  uint64_t fine_token_learned_projected_token_recall_min_ppm;
  uint64_t fine_token_learned_projected_token_recall_avg_ppm;
  uint64_t norm_stress_topk_tokens;
  uint64_t norm_stress_no_augmentation_token_recall_min_ppm;
  uint64_t norm_stress_no_augmentation_token_recall_avg_ppm;
  uint64_t norm_stress_synthetic_norm_augmented_token_recall_min_ppm;
  uint64_t norm_stress_synthetic_norm_augmented_token_recall_avg_ppm;
  uint64_t dense_selector_attention_stage_avg_ns;
  uint64_t rt_selector_attention_stage_avg_ns;
  uint64_t rt_selector_overlapped_attention_stage_avg_ns;
  uint64_t dense_vs_rt_attention_stage_speedup_x1000;
  uint64_t dense_vs_rt_overlapped_attention_stage_speedup_x1000;
  uint64_t dense_full_vs_rt_attention_stage_speedup_x1000;
  uint64_t dense_full_vs_rt_overlapped_attention_stage_speedup_x1000;
  uint64_t selected_hash;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t device_allocations;
  uint64_t device_frees;
  uint64_t device_arena_bytes;
  uint64_t hot_path_allocations;
  char backend[64];
  char reason[192];
} NervaCudaExperimentalRtCandidateBenchResult;

typedef struct NervaCudaDeepSeekQuantSmokeResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t fp8_rows;
  uint32_t fp8_cols;
  uint32_t fp8_block_rows;
  uint32_t fp8_block_cols;
  uint32_t mxfp4_rows;
  uint32_t mxfp4_packed_cols;
  uint32_t mxfp4_scale_packed_cols;
  uint64_t fp8_output_hash;
  uint64_t mxfp4_output_hash;
  uint64_t fp8_mismatches;
  uint64_t mxfp4_mismatches;
  float fp8_max_abs_diff;
  float mxfp4_max_abs_diff;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekQuantSmokeResult;

typedef struct NervaCudaDeepSeekQuantFp8DequantRequest {
  uint32_t rows;
  uint32_t cols;
  uint32_t block_rows;
  uint32_t block_cols;
  const uint8_t *weights;
  const uint8_t *scales;
  float *output;
} NervaCudaDeepSeekQuantFp8DequantRequest;

typedef struct NervaCudaDeepSeekQuantMxfp4DequantRequest {
  uint32_t rows;
  uint32_t packed_cols;
  uint32_t scale_packed_cols;
  const uint8_t *packed;
  const uint8_t *scales;
  float *output;
} NervaCudaDeepSeekQuantMxfp4DequantRequest;

typedef struct NervaCudaDeepSeekQuantDequantResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t rows;
  uint32_t cols;
  uint32_t block_rows;
  uint32_t block_cols;
  uint64_t output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekQuantDequantResult;

typedef struct NervaCudaDeepSeekQuantFp8F32ScaleMatvecRequest {
  uint32_t rows;
  uint32_t cols;
  uint32_t block_rows;
  uint32_t block_cols;
  const uint8_t *weights;
  const float *scales;
  const float *input;
  float *output;
} NervaCudaDeepSeekQuantFp8F32ScaleMatvecRequest;

typedef struct NervaCudaDeepSeekQuantFp8F32ScaleEncodedMatvecRequest {
  uint32_t rows;
  uint32_t cols;
  uint32_t block_rows;
  uint32_t block_cols;
  uint32_t input_dtype;
  const uint8_t *weights;
  const float *scales;
  const uint16_t *input;
  float *output;
} NervaCudaDeepSeekQuantFp8F32ScaleEncodedMatvecRequest;

typedef struct NervaCudaDeepSeekFusedInvRopeFp8QuantRequest {
  uint32_t num_tokens;
  uint32_t n_groups;
  uint32_t heads_per_group;
  uint32_t head_dim;
  uint32_t rope_dim;
  uint32_t quant_group_size;
  uint32_t cos_sin_stride;
  float fp8_max;
  float eps;
  const float *input;
  const int64_t *positions;
  const float *cos_sin_cache;
  uint8_t *fp8_output;
  float *scale_output;
  uint32_t *packed_scale_output;
} NervaCudaDeepSeekFusedInvRopeFp8QuantRequest;

typedef struct NervaCudaDeepSeekFusedInvRopeFp8QuantResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t num_tokens;
  uint32_t n_groups;
  uint32_t heads_per_group;
  uint32_t head_dim;
  uint32_t rope_dim;
  uint32_t quant_group_size;
  uint32_t scale_blocks;
  uint64_t fp8_output_hash;
  uint64_t scale_output_hash;
  uint64_t packed_scale_output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekFusedInvRopeFp8QuantResult;

typedef struct NervaCudaDeepSeekRouterSmokeResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t v3_num_experts;
  uint32_t v3_num_groups;
  uint32_t v3_top_k_groups;
  uint32_t v3_top_k;
  uint32_t v4_num_experts;
  uint32_t v4_top_k;
  uint32_t v4_hash_top_k;
  uint32_t v3_expert_ids[2];
  uint32_t v4_expert_ids[2];
  uint32_t v4_hash_expert_ids[3];
  float v3_weights[2];
  float v4_weights[2];
  float v4_hash_weights[3];
  uint64_t v3_output_hash;
  uint64_t v4_output_hash;
  uint64_t v4_hash_output_hash;
  uint64_t v3_mismatches;
  uint64_t v4_mismatches;
  uint64_t v4_hash_mismatches;
  float v3_max_abs_diff;
  float v4_max_abs_diff;
  float v4_hash_max_abs_diff;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekRouterSmokeResult;

typedef struct NervaCudaDeepSeekRouterRouteRequest {
  uint32_t router_kind;
  uint32_t num_experts;
  uint32_t num_groups;
  uint32_t top_k_groups;
  uint32_t top_k;
  uint32_t norm_topk_prob;
  uint32_t route_token;
  float routed_scaling_factor;
  const float *logits;
  const float *correction_bias;
  const uint32_t *hash_route_table;
  uint32_t hash_route_table_len;
  uint32_t *expert_ids;
  float *weights;
} NervaCudaDeepSeekRouterRouteRequest;

typedef struct NervaCudaDeepSeekRouterRouteResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  int32_t route_error;
  uint32_t router_kind;
  uint32_t num_experts;
  uint32_t num_groups;
  uint32_t top_k_groups;
  uint32_t top_k;
  uint64_t output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekRouterRouteResult;

typedef struct NervaCudaDeepSeekMlaSmokeResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t heads;
  uint32_t tokens;
  uint32_t kv_lora_rank;
  uint32_t qk_nope_head_dim;
  uint32_t qk_rope_head_dim;
  uint32_t v_head_dim;
  float softmax_scale;
  float output[4];
  uint64_t output_hash;
  uint64_t mismatches;
  float max_abs_diff;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekMlaSmokeResult;

typedef struct NervaCudaDeepSeekMlaDecodeRequest {
  uint32_t heads;
  uint32_t tokens;
  uint32_t kv_lora_rank;
  uint32_t qk_nope_head_dim;
  uint32_t qk_rope_head_dim;
  uint32_t v_head_dim;
  float softmax_scale;
  const float *q_nope;
  const float *q_pe;
  const float *kv_c;
  const float *k_pe;
  const float *w_uk;
  const float *w_uv;
  float *output;
} NervaCudaDeepSeekMlaDecodeRequest;

typedef struct NervaCudaDeepSeekMlaDecodeResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  int32_t decode_error;
  uint32_t heads;
  uint32_t tokens;
  uint32_t kv_lora_rank;
  uint32_t qk_nope_head_dim;
  uint32_t qk_rope_head_dim;
  uint32_t v_head_dim;
  float softmax_scale;
  uint64_t output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekMlaDecodeResult;

typedef struct NervaCudaDeepSeekQKvRmsNormRequest {
  uint32_t num_tokens;
  uint32_t q_size;
  uint32_t kv_size;
  float eps;
  const float *q;
  const float *kv;
  const float *q_weight;
  const float *kv_weight;
  float *q_out;
  float *kv_out;
} NervaCudaDeepSeekQKvRmsNormRequest;

typedef struct NervaCudaDeepSeekQKvRmsNormResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t num_tokens;
  uint32_t q_size;
  uint32_t kv_size;
  float eps;
  uint64_t output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekQKvRmsNormResult;

typedef struct NervaCudaDeepSeekKvFp8DsMlaPackRequest {
  uint32_t block_size;
  uint32_t token_index;
  uint32_t nope_bytes;
  uint32_t rope_bf16_values;
  uint32_t scale_dim;
  const uint8_t *nope_fp8;
  const uint16_t *rope_bf16;
  const uint8_t *scales;
  uint8_t *output_block;
} NervaCudaDeepSeekKvFp8DsMlaPackRequest;

typedef struct NervaCudaDeepSeekKvFp8DsMlaPackResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t block_size;
  uint32_t token_index;
  uint32_t token_stride;
  uint32_t scale_dim;
  uint64_t block_bytes;
  uint64_t output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekKvFp8DsMlaPackResult;

typedef struct NervaCudaDeepSeekCompressedSlotMappingRequest {
  uint32_t num_tokens;
  uint32_t num_reqs;
  uint32_t block_table_stride;
  uint32_t block_size;
  uint32_t compress_ratio;
  const int32_t *query_start_loc;
  const int32_t *seq_lens;
  const int32_t *block_table;
  int64_t *output_slots;
} NervaCudaDeepSeekCompressedSlotMappingRequest;

typedef struct NervaCudaDeepSeekCompressedSlotMappingResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t num_tokens;
  uint32_t num_reqs;
  uint32_t block_table_stride;
  uint32_t block_size;
  uint32_t compress_ratio;
  uint32_t valid_slots;
  uint32_t pad_slots;
  uint64_t output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekCompressedSlotMappingResult;

typedef struct NervaCudaDeepSeekC128TopkMetadataRequest {
  uint32_t num_tokens;
  uint32_t num_decode_tokens;
  uint32_t num_reqs;
  uint32_t block_table_stride;
  uint32_t block_size;
  uint32_t compress_ratio;
  uint32_t max_compressed_tokens;
  const int64_t *positions;
  const int32_t *token_to_req_indices;
  const int32_t *block_table;
  const int64_t *slot_mapping;
  int32_t *global_decode;
  int32_t *decode_lens;
  int32_t *prefill_local;
} NervaCudaDeepSeekC128TopkMetadataRequest;

typedef struct NervaCudaDeepSeekC128TopkMetadataResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t num_tokens;
  uint32_t num_decode_tokens;
  uint32_t num_prefill_tokens;
  uint32_t num_reqs;
  uint32_t block_table_stride;
  uint32_t block_size;
  uint32_t compress_ratio;
  uint32_t max_compressed_tokens;
  uint32_t valid_decode_tokens;
  uint32_t decode_entries;
  uint32_t prefill_entries;
  uint64_t output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekC128TopkMetadataResult;

typedef struct NervaCudaDeepSeekC4IndexerTopkRequest {
  uint32_t num_tokens;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t max_compressed_tokens;
  uint32_t topk_tokens;
  const float *query;
  const float *key_cache;
  const float *weights;
  const int32_t *context_lens;
  int32_t *topk_indices;
  float *topk_scores;
} NervaCudaDeepSeekC4IndexerTopkRequest;

typedef struct NervaCudaDeepSeekC4IndexerTopkResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t num_tokens;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t max_compressed_tokens;
  uint32_t topk_tokens;
  uint32_t valid_tokens;
  uint32_t selected_entries;
  uint64_t output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekC4IndexerTopkResult;

typedef struct NervaCudaDeepSeekSavePartialStatesRequest {
  uint32_t num_tokens;
  uint32_t block_size;
  uint32_t head_size;
  uint32_t state_width;
  uint32_t compress_ratio;
  uint32_t num_blocks;
  const float *kv;
  const float *score;
  const float *ape;
  const int64_t *positions;
  const int64_t *slot_mapping;
  float *state_cache;
} NervaCudaDeepSeekSavePartialStatesRequest;

typedef struct NervaCudaDeepSeekSavePartialStatesResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t num_tokens;
  uint32_t block_size;
  uint32_t head_size;
  uint32_t state_width;
  uint32_t compress_ratio;
  uint32_t num_blocks;
  uint32_t written_tokens;
  uint32_t skipped_tokens;
  uint64_t output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekSavePartialStatesResult;

typedef struct NervaCudaDeepSeekCompressNormRopeFp8CacheRequest {
  uint32_t num_tokens;
  uint32_t num_reqs;
  uint32_t block_table_stride;
  uint32_t state_block_size;
  uint32_t kv_cache_block_size;
  uint32_t head_size;
  uint32_t state_width;
  uint32_t rope_head_dim;
  uint32_t compress_ratio;
  uint32_t overlap;
  uint32_t quant_block;
  uint32_t token_stride;
  uint32_t scale_dim;
  uint32_t scale_format;
  uint32_t num_state_blocks;
  uint32_t num_kv_blocks;
  uint32_t kv_cache_block_stride;
  uint32_t cos_sin_stride;
  uint32_t cos_sin_values;
  float rms_norm_eps;
  float fp8_max;
  const float *state_cache;
  const int32_t *token_to_req_indices;
  const int64_t *positions;
  const int64_t *slot_mapping;
  const int32_t *block_table;
  const int64_t *kv_slot_mapping;
  const float *rms_norm_weight;
  const float *cos_sin_cache;
  uint8_t *kv_cache;
} NervaCudaDeepSeekCompressNormRopeFp8CacheRequest;

typedef struct NervaCudaDeepSeekCompressNormRopeFp8CacheResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t num_tokens;
  uint32_t head_size;
  uint32_t rope_head_dim;
  uint32_t compress_ratio;
  uint32_t quant_block;
  uint32_t token_stride;
  uint32_t scale_dim;
  uint32_t scale_format;
  uint32_t written_tokens;
  uint32_t skipped_tokens;
  uint64_t kv_cache_bytes;
  uint64_t output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekCompressNormRopeFp8CacheResult;

typedef struct NervaCudaDeepSeekMoeSmokeResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  uint32_t hidden_size;
  uint32_t intermediate_size;
  uint32_t num_experts;
  uint32_t top_k;
  float swiglu_limit;
  uint32_t expert_ids[2];
  float expert_weights[2];
  float output[3];
  uint64_t output_hash;
  uint64_t mismatches;
  float max_abs_diff;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekMoeSmokeResult;

typedef struct NervaCudaDeepSeekMoeForwardRequest {
  uint32_t hidden_size;
  uint32_t intermediate_size;
  uint32_t num_experts;
  uint32_t top_k;
  uint32_t clamp_swiglu;
  float swiglu_limit;
  const float *input;
  const uint32_t *expert_ids;
  const float *expert_weights;
  const float *w_gate;
  const float *w_up;
  const float *w_down;
  float *output;
} NervaCudaDeepSeekMoeForwardRequest;

typedef struct NervaCudaDeepSeekMoeForwardResult {
  int32_t status;
  int32_t cuda_error;
  int32_t device_count;
  int32_t moe_error;
  uint32_t hidden_size;
  uint32_t intermediate_size;
  uint32_t num_experts;
  uint32_t top_k;
  uint32_t clamp_swiglu;
  float swiglu_limit;
  uint64_t output_hash;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t kernel_launches;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaDeepSeekMoeForwardResult;

int nerva_cuda_device_smoke(NervaCudaDeviceSmokeResult *out);
int nerva_cuda_synthetic_graph_smoke(uint32_t steps,
                                     uint32_t ring_capacity,
                                     uint32_t seed_token,
                                     NervaCudaSyntheticGraphResult *out);
int nerva_cuda_tiny_block_smoke(NervaCudaTinyBlockResult *out);
int nerva_cuda_loaded_tiny_block_smoke(NervaCudaLoadedTinyBlockResult *out);
int nerva_cuda_block_forward_u16(const NervaCudaBlockForwardRequest *request,
                                 NervaCudaBlockForwardResult *out);
int nerva_cuda_greedy_sampler_smoke(NervaCudaGreedySamplerResult *out);
int nerva_cuda_hf_sample_u16(const NervaCudaHfSamplerRequest *request,
                             NervaCudaHfSamplerResult *out);
int nerva_cuda_hf_decode_step_u16(const NervaCudaHfDecodeStepRequest *request,
                                  NervaCudaHfDecodeStepResult *out);
int nerva_cuda_hf_decode_chain_u16(const NervaCudaHfDecodeChainRequest *request,
                                   NervaCudaHfDecodeChainResult *out);
int nerva_cuda_hf_decode_sequence_u16(
    const NervaCudaHfDecodeSequenceRequest *request,
    NervaCudaHfDecodeSequenceResult *out);
int nerva_cuda_hf_decode_sequence_plan_layout(
    const NervaCudaHfDecodeSequenceLayoutPlanRequest *request,
    NervaCudaHfDecodeSequenceLayoutPlanResult *out);
int nerva_cuda_hf_decode_sequence_session_create(
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request,
    NervaCudaHfDecodeSequenceSessionCreateResult *out,
    NervaCudaHfDecodeSequenceSession **session);
int nerva_cuda_hf_decode_sequence_session_run(
    const NervaCudaHfDecodeSequenceSessionRunRequest *request,
    NervaCudaHfDecodeSequenceResult *out);
int nerva_cuda_hf_decode_sequence_session_start(
    const NervaCudaHfDecodeSequenceSessionStartRequest *request,
    NervaCudaHfDecodeSequenceResult *out);
int nerva_cuda_hf_decode_sequence_session_advance(
    const NervaCudaHfDecodeSequenceSessionAdvanceRequest *request,
    NervaCudaHfDecodeSequenceResult *out);
int nerva_cuda_hf_decode_sequence_deepseek_v4_swa_kv_snapshot(
    const NervaCudaHfDecodeSequenceDeepSeekV4SwaKvSnapshotRequest *request,
    NervaCudaHfDecodeSequenceDeepSeekV4SwaKvSnapshotResult *out);
int nerva_cuda_hf_decode_sequence_deepseek_v3_mla_kv_snapshot(
    const NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotRequest *request,
    NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotResult *out);
int nerva_cuda_hf_decode_sequence_deepseek_v32_mla_packed_kv_snapshot(
    const NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotRequest *request,
    NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotResult *out);
int nerva_cuda_hf_decode_sequence_deepseek_v32_indexer_kv_snapshot(
    const NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotRequest *request,
    NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotResult *out);
int nerva_cuda_hf_decode_sequence_deepseek_v4_compressed_kv_snapshot(
    const NervaCudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshotRequest *request,
    NervaCudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshotResult *out);
int nerva_cuda_hf_decode_sequence_projection_batch_plan(
    const NervaCudaHfDecodeSequenceProjectionBatchPlanRequest *request,
    NervaCudaHfDecodeSequenceProjectionBatchPlanResult *out);
int nerva_cuda_hf_decode_sequence_projection_batch_execute(
    const NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest *request,
    NervaCudaHfDecodeSequenceProjectionBatchExecuteResult *out);
int nerva_cuda_hf_decode_sequence_layer_projection_batch_execute(
    const NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteRequest *request,
    NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult *out);
int nerva_cuda_hf_decode_sequence_batch_advance_one(
    const NervaCudaHfDecodeSequenceBatchAdvanceRequest *request,
    NervaCudaHfDecodeSequenceBatchAdvanceResult *out);
int nerva_cuda_hf_decode_sequence_session_fork_shared_weights(
    const NervaCudaHfDecodeSequenceSessionForkSharedWeightsRequest *request,
    NervaCudaHfDecodeSequenceSessionCreateResult *out,
    NervaCudaHfDecodeSequenceSession **session_out);
int nerva_cuda_hf_decode_sequence_session_destroy(
    NervaCudaHfDecodeSequenceSession *session,
    NervaCudaHfDecodeSequenceSessionCreateResult *out);
int nerva_cuda_tiny_decode_smoke(uint32_t steps,
                                 uint32_t ring_capacity,
                                 uint32_t seed_token,
                                 NervaCudaTinyDecodeResult *out);
int nerva_cuda_tiered_attention_smoke(NervaCudaTieredAttentionResult *out);
int nerva_cuda_backend_contract_smoke(NervaCudaBackendContractResult *out,
                                      uint64_t device_bytes,
                                      uint64_t pinned_bytes);
int nerva_cuda_projection_bench(const NervaCudaProjectionBenchRequest *request,
                                NervaCudaProjectionBenchResult *out);
int nerva_cuda_experimental_rt_candidate_bench(
    const NervaCudaExperimentalRtCandidateBenchRequest *request,
    NervaCudaExperimentalRtCandidateBenchResult *out);
int nerva_cuda_deepseek_quant_smoke(NervaCudaDeepSeekQuantSmokeResult *out);
int nerva_cuda_deepseek_quant_fp8_dequant(
    const NervaCudaDeepSeekQuantFp8DequantRequest *request,
    NervaCudaDeepSeekQuantDequantResult *out);
int nerva_cuda_deepseek_quant_mxfp4_dequant(
    const NervaCudaDeepSeekQuantMxfp4DequantRequest *request,
    NervaCudaDeepSeekQuantDequantResult *out);
int nerva_cuda_deepseek_quant_fp8_f32_scale_matvec(
    const NervaCudaDeepSeekQuantFp8F32ScaleMatvecRequest *request,
    NervaCudaDeepSeekQuantDequantResult *out);
int nerva_cuda_deepseek_quant_fp8_f32_scale_encoded_matvec(
    const NervaCudaDeepSeekQuantFp8F32ScaleEncodedMatvecRequest *request,
    NervaCudaDeepSeekQuantDequantResult *out);
int nerva_cuda_deepseek_fused_inv_rope_fp8_quant(
    const NervaCudaDeepSeekFusedInvRopeFp8QuantRequest *request,
    NervaCudaDeepSeekFusedInvRopeFp8QuantResult *out);
int nerva_cuda_deepseek_router_smoke(NervaCudaDeepSeekRouterSmokeResult *out);
int nerva_cuda_deepseek_router_route(
    const NervaCudaDeepSeekRouterRouteRequest *request,
    NervaCudaDeepSeekRouterRouteResult *out);
int nerva_cuda_deepseek_mla_smoke(NervaCudaDeepSeekMlaSmokeResult *out);
int nerva_cuda_deepseek_mla_decode(
    const NervaCudaDeepSeekMlaDecodeRequest *request,
    NervaCudaDeepSeekMlaDecodeResult *out);
int nerva_cuda_deepseek_qkv_rmsnorm(
    const NervaCudaDeepSeekQKvRmsNormRequest *request,
    NervaCudaDeepSeekQKvRmsNormResult *out);
int nerva_cuda_deepseek_kv_fp8_ds_mla_pack(
    const NervaCudaDeepSeekKvFp8DsMlaPackRequest *request,
    NervaCudaDeepSeekKvFp8DsMlaPackResult *out);
int nerva_cuda_deepseek_v32_kv_fp8_ds_mla_pack(
    const NervaCudaDeepSeekKvFp8DsMlaPackRequest *request,
    NervaCudaDeepSeekKvFp8DsMlaPackResult *out);
int nerva_cuda_deepseek_compressed_slot_mapping(
    const NervaCudaDeepSeekCompressedSlotMappingRequest *request,
    NervaCudaDeepSeekCompressedSlotMappingResult *out);
int nerva_cuda_deepseek_c128_topk_metadata(
    const NervaCudaDeepSeekC128TopkMetadataRequest *request,
    NervaCudaDeepSeekC128TopkMetadataResult *out);
int nerva_cuda_deepseek_c4_indexer_topk(
    const NervaCudaDeepSeekC4IndexerTopkRequest *request,
    NervaCudaDeepSeekC4IndexerTopkResult *out);
int nerva_cuda_deepseek_save_partial_states(
    const NervaCudaDeepSeekSavePartialStatesRequest *request,
    NervaCudaDeepSeekSavePartialStatesResult *out);
int nerva_cuda_deepseek_compress_norm_rope_fp8_cache(
    const NervaCudaDeepSeekCompressNormRopeFp8CacheRequest *request,
    NervaCudaDeepSeekCompressNormRopeFp8CacheResult *out);
int nerva_cuda_deepseek_moe_smoke(NervaCudaDeepSeekMoeSmokeResult *out);
int nerva_cuda_deepseek_moe_forward(
    const NervaCudaDeepSeekMoeForwardRequest *request,
    NervaCudaDeepSeekMoeForwardResult *out);

#ifdef __cplusplus
}
#endif
