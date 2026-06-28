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
} NervaCudaHfDecodeSequenceResult;

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
  uint64_t resident_kv_bytes;
  uint64_t device_arena_bytes;
  uint64_t pinned_host_bytes;
  uint64_t h2d_bytes;
  uint64_t sync_calls;
  uint64_t hot_path_allocations;
} NervaCudaHfDecodeSequenceSessionCreateResult;

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
} NervaCudaHfDecodeSequenceSessionRunRequest;

typedef struct NervaCudaHfDecodeSequenceSessionStartRequest {
  NervaCudaHfDecodeSequenceSession *session;
  const uint32_t *prompt_tokens;
  uint32_t prompt_token_count;
  uint32_t has_eos_token;
  uint32_t eos_token;
} NervaCudaHfDecodeSequenceSessionStartRequest;

typedef struct NervaCudaHfDecodeSequenceSessionAdvanceRequest {
  NervaCudaHfDecodeSequenceSession *session;
  uint32_t steps;
  uint32_t *output_tokens;
  uint32_t output_token_capacity;
} NervaCudaHfDecodeSequenceSessionAdvanceRequest;

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
int nerva_cuda_hf_decode_sequence_projection_batch_plan(
    const NervaCudaHfDecodeSequenceProjectionBatchPlanRequest *request,
    NervaCudaHfDecodeSequenceProjectionBatchPlanResult *out);
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

#ifdef __cplusplus
}
#endif
