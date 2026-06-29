#include "nerva_cuda_api.h"
#include "hf_decode_sequence/device_ops.cuh"
#include "hf_decode_sequence/kernels.cuh"
#include "hf_decode_sequence/projection.cuh"
#include "hf_decode_sequence/sampler.cuh"
#include "hf_decode_sequence/types.cuh"
#include "hf_decode_sequence/weights.cuh"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#if NERVA_HAVE_CUDNN_FRONTEND
#include <cudnn.h>
#include <cudnn_frontend.h>
#endif

namespace {

enum CreateFailureStage : int32_t {
  kCreateStageNone = 0,
  kCreateStageInvalidRequest = 1,
  kCreateStageGetDeviceCount = 2,
  kCreateStageSetDevice = 3,
  kCreateStageSessionAlloc = 4,
  kCreateStageHostWeightAlloc = 5,
  kCreateStageHostSlotsAlloc = 6,
  kCreateStageDeviceArenaAlloc = 7,
  kCreateStageDeviceLayoutsAlloc = 8,
  kCreateStageDeviceScratchAlloc = 9,
  kCreateStageProjectionInputAlloc = 10,
  kCreateStagePackedQkvAlloc = 11,
  kCreateStagePackedGateUpAlloc = 12,
  kCreateStageKvKeysAlloc = 13,
  kCreateStageKvValuesAlloc = 14,
  kCreateStagePromptTokensAlloc = 15,
  kCreateStageDeviceSlotsAlloc = 16,
  kCreateStageDeviceStepAlloc = 17,
  kCreateStageCublasWorkspaceAlloc = 18,
  kCreateStageStreamCreate = 19,
  kCreateStageCublasCreate = 20,
  kCreateStageCublasLtCreate = 21,
  kCreateStageCublasConfigure = 22,
  kCreateStageStartEventCreate = 23,
  kCreateStageStopEventCreate = 24,
  kCreateStageDescriptorCopy = 25,
  kCreateStageLayoutCopy = 26,
  kCreateStagePackReplicas = 27,
  kCreateStageWarmCublas = 28,
  kCreateStageSetupSynchronize = 29,
  kCreateStagePrefillHiddenAlloc = 30,
  kCreateStagePrefillChunkAlloc = 31,
  kCreateStageDecodeAttentionAlloc = 32,
  kCreateStageDecodeSdpaAlloc = 33,
  kCreateStageProjectionPlanAutotune = 34,
  kCreateStageExperimentalRtDecodeUnsupported = 35,
};

struct SessionSharedWeights {
  uint16_t *device_arena = nullptr;
  SequenceLayerLayout *device_layouts = nullptr;
  uint16_t *device_qkv_packed = nullptr;
  uint16_t *device_gate_up_packed = nullptr;

  ~SessionSharedWeights() {
    cudaFree(device_gate_up_packed);
    cudaFree(device_qkv_packed);
    cudaFree(device_layouts);
    cudaFree(device_arena);
  }
};

constexpr uint32_t kProjectionBatchPlanReady = 0;
constexpr uint32_t kProjectionBatchPlanInvalidRequest = 1;
constexpr uint32_t kProjectionBatchPlanNoSessions = 2;
constexpr uint32_t kProjectionBatchPlanNoReadySessions = 3;
constexpr uint32_t kProjectionBatchPlanSharedWeightsUnproven = 4;
constexpr uint32_t kProjectionBatchPlanInsufficientCompatibleReady = 5;
constexpr uint32_t kProjectionBatchPlanUnsupportedProjection = 6;
constexpr uint32_t kProjectionBatchPlanInvalidLayer = 7;
constexpr uint32_t kProjectionBatchPlanInsufficientScratch = 8;
constexpr uint32_t kProjectionBatchKindQkv = 1;
constexpr uint32_t kProjectionBatchKindAttentionOutput = 2;
constexpr uint32_t kProjectionBatchKindGateUp = 3;
constexpr uint32_t kProjectionBatchKindDown = 4;
constexpr uint32_t kProjectionBatchKindLmHead = 5;































int fail(NervaCudaHfDecodeSequenceResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  return -1;
}

#if NERVA_HAVE_CUDNN_FRONTEND
cudaError_t cudnn_to_cuda(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return cudaSuccess;
    case CUDNN_STATUS_ALLOC_FAILED:
      return cudaErrorMemoryAllocation;
    case CUDNN_STATUS_BAD_PARAM:
      return cudaErrorInvalidValue;
    case CUDNN_STATUS_NOT_SUPPORTED:
      return cudaErrorNotSupported;
    case CUDNN_STATUS_EXECUTION_FAILED:
      return cudaErrorLaunchFailure;
    default:
      return cudaErrorUnknown;
  }
}
#endif

cudaError_t final_head_gemv(cublasHandle_t handle, uint16_t *arena,
                            SequenceArenaLayout arena_layout, uint32_t dtype,
                            uint32_t hidden, uint32_t vocab_size,
                            float *device_logits) {
  return encoded_row_major_gemv(handle, arena + arena_layout.lm_head,
                                arena + arena_layout.input, vocab_size, hidden,
                                dtype, device_logits);
}

cudaError_t warm_cublas_gemv(cublasHandle_t handle, uint16_t *arena,
                             SequenceArenaLayout arena_layout, uint32_t dtype,
                             float *scratch, cudaStream_t stream) {
  cudaError_t err = cudaMemsetAsync(arena + arena_layout.input, 0,
                                    sizeof(uint16_t), stream);
  if (err != cudaSuccess) {
    return err;
  }
  return encoded_row_major_gemv(handle, arena + arena_layout.lm_head,
                                arena + arena_layout.input, 1, 1, dtype,
                                scratch);
}

uint32_t observed_count_for(uint32_t steps, uint32_t prompt_token_count,
                            uint32_t has_eos_token, uint32_t eos_token,
                            const NervaCudaSyntheticTokenSlot *slots) {
  uint32_t count = steps;
  if (has_eos_token == 0) {
    return count;
  }
  const uint32_t output_start = prompt_token_count - 1u;
  for (uint32_t index = 0; index < steps; ++index) {
    if (slots[output_start + index].token == eos_token) {
      count = index + 1;
      break;
    }
  }
  return count;
}

uint32_t observed_count(const NervaCudaHfDecodeSequenceRequest *request,
                        const NervaCudaSyntheticTokenSlot *slots) {
  return observed_count_for(request->steps, request->prompt_token_count,
                            request->has_eos_token, request->eos_token, slots);
}

uint64_t saturating_mul_profile_value(uint64_t value, uint64_t scale) {
  if (value != 0 && scale > UINT64_MAX / value) {
    return UINT64_MAX;
  }
  return value * scale;
}

void scale_profile_counters(NervaCudaHfDecodeSequenceResult *out,
                            uint64_t scale) {
  if (out == nullptr || scale <= 1) {
    return;
  }
  out->projection_ns = saturating_mul_profile_value(out->projection_ns, scale);
  out->qkv_projection_ns =
      saturating_mul_profile_value(out->qkv_projection_ns, scale);
  out->attention_output_projection_ns =
      saturating_mul_profile_value(out->attention_output_projection_ns, scale);
  out->gate_up_projection_ns =
      saturating_mul_profile_value(out->gate_up_projection_ns, scale);
  out->down_projection_ns =
      saturating_mul_profile_value(out->down_projection_ns, scale);
  out->lm_head_projection_ns =
      saturating_mul_profile_value(out->lm_head_projection_ns, scale);
  out->attention_ns = saturating_mul_profile_value(out->attention_ns, scale);
  out->mlp_ns = saturating_mul_profile_value(out->mlp_ns, scale);
  out->norm_ns = saturating_mul_profile_value(out->norm_ns, scale);
  out->sampling_ns = saturating_mul_profile_value(out->sampling_ns, scale);
}

#if NERVA_HAVE_CUDNN_FRONTEND
struct CudnnPrefillSdpaPlan {
  std::unique_ptr<cudnn_frontend::graph::Graph> graph;
  uint32_t seq_tokens = 0;
  uint32_t heads = 0;
  uint32_t kv_heads = 0;
  uint32_t head_dim = 0;
  uint64_t rows = 0;
  size_t workspace_bytes = 0;
};

struct CudnnDecodeSdpaPlan {
  std::unique_ptr<cudnn_frontend::graph::Graph> graph;
  uint32_t max_context_tokens = 0;
  uint32_t kv_token_capacity = 0;
  uint32_t heads = 0;
  uint32_t kv_heads = 0;
  uint32_t head_dim = 0;
  size_t workspace_bytes = 0;
};
#endif

}  // namespace

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
  uint32_t experimental_rt_page_tokens = 0;
  uint32_t experimental_rt_pages = 0;
  uint32_t experimental_rt_local_window_tokens = 0;
  uint32_t experimental_rt_sink_tokens = 0;
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
  uint64_t prefill_gate_up_bytes = 0;
  uint64_t prefill_ff_bytes = 0;
  uint64_t prefill_down_bytes = 0;
  uint64_t decode_attention_values_bytes = 0;
  uint64_t decode_attention_stats_bytes = 0;
  uint32_t decode_attention_max_chunks = 0;
  uint64_t decode_q_bytes = 0;
  uint64_t decode_seq_len_bytes = 0;
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
  float *device_prefill_gate_up = nullptr;
  uint16_t *device_prefill_ff = nullptr;
  float *device_prefill_down = nullptr;
  float *device_decode_attention_values = nullptr;
  float *device_decode_attention_m = nullptr;
  float *device_decode_attention_l = nullptr;
  uint16_t *device_decode_q = nullptr;
  int32_t *device_decode_seq_len_q = nullptr;
  int32_t *device_decode_seq_len_kv = nullptr;
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

namespace {

void free_session_fields(NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr) {
    return;
  }
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
         session->prefill_gate_up_bytes + session->prefill_ff_bytes +
         session->prefill_down_bytes + session->decode_attention_values_bytes +
         session->decode_attention_stats_bytes * 2 + session->decode_q_bytes +
         session->decode_seq_len_bytes +
         session->packed_qkv_bytes + session->packed_gate_up_bytes + session->kv_bytes +
         session->kv_block_table_bytes +
         session->prompt_bytes + session->slots_bytes + sizeof(uint32_t) +
         kCublasWorkspaceBytes;
}

uint64_t session_fixed_footprint_without_prefill_chunk(
    const NervaCudaHfDecodeSequenceSession *session) {
  return session->arena_bytes + session->layout_bytes + session->scratch_bytes +
         session->projection_input_bytes + session->projection_batch_input_bytes +
         session->projection_batch_output_bytes + session->prefill_hidden_bytes * 2 +
         session->decode_attention_values_bytes +
         session->decode_attention_stats_bytes * 2 + session->decode_q_bytes +
         session->decode_seq_len_bytes +
         session->packed_qkv_bytes + session->packed_gate_up_bytes + session->kv_bytes +
         session->kv_block_table_bytes +
         session->prompt_bytes + session->slots_bytes + sizeof(uint32_t) +
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

uint64_t prefill_chunk_scratch_bytes(uint64_t chunk_tokens,
                                     uint64_t projection_input_elements,
                                     uint64_t prefill_qkv_rows,
                                     uint64_t attention_hidden,
                                     uint64_t hidden,
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
         hf_decode_sampler_config_matches(session->cached_sampler, sampler);
}

bool use_cublas_layer_path(const NervaCudaHfDecodeSequenceSession *session) {
  const uint32_t attention_hidden = session->heads * session->head_dim;
  return session->hidden >= 128 && attention_hidden == session->hidden &&
         session->host_layouts.size() == session->layer_count &&
         session->device_projection_input != nullptr &&
         session->device_qkv_packed != nullptr &&
         session->device_gate_up_packed != nullptr &&
         session->cublas != nullptr && session->cublas_lt != nullptr;
}

bool projection_batch_session_ready(
    const NervaCudaHfDecodeSequenceSession *session) {
  const uint32_t attention_hidden = session->heads * session->head_dim;
  return session->hidden >= 128 && attention_hidden == session->hidden &&
         session->host_layouts.size() == session->layer_count &&
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
    err = create_lt_gemv_plan(
        &session->gate_up_plan,
        static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden,
        session->dtype);
  if (err == cudaSuccess)
    err = create_lt_gemv_plan(&session->down_plan, session->hidden,
                              session->intermediate, session->dtype);
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
    err = autotune_lt_gemv_plan(
        session->cublas, session->cublas_lt, session->stream,
        session->cublas_workspace, kCublasWorkspaceBytes,
        &session->gate_up_plan,
        session->device_gate_up_packed, session->device_projection_input,
        scratch.gate);
  if (err == cudaSuccess)
    err = autotune_lt_gemv_plan(
        session->cublas, session->cublas_lt, session->stream,
        session->cublas_workspace, kCublasWorkspaceBytes, &session->down_plan,
        session->device_arena + layout.w_down, session->device_projection_input,
        scratch.down);
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
  if (err == cudaSuccess && (!session->qkv_plan.ready ||
                             !session->attention_output_plan.ready ||
                             !session->gate_up_plan.ready ||
                             !session->down_plan.ready ||
                             !session->lm_head_plan.ready)) {
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

void stash_prefill_metrics(NervaCudaHfDecodeSequenceSession *session,
                           const NervaCudaHfDecodeSequenceResult *out) {
  session->pending_prefill_kernel_launches = out->kernel_launches;
  session->pending_prefill_device_elapsed_ns = out->device_elapsed_ns;
  session->pending_prefill_sync_calls = out->sync_calls;
  session->pending_prefill_graph_replays = out->graph_replays;
  session->pending_prefill_graph_launches = out->graph_launches;
  session->pending_prefill_graph_nodes = out->graph_nodes;
  session->pending_prefill_available = 1;
}

void drain_prefill_metrics(NervaCudaHfDecodeSequenceSession *session,
                           NervaCudaHfDecodeSequenceResult *out) {
  if (session->pending_prefill_available == 0) {
    return;
  }
  out->kernel_launches += session->pending_prefill_kernel_launches;
  out->device_elapsed_ns += session->pending_prefill_device_elapsed_ns;
  out->sync_calls += session->pending_prefill_sync_calls;
  out->graph_replays += session->pending_prefill_graph_replays;
  out->graph_launches += session->pending_prefill_graph_launches;
  if (out->graph_nodes == 0) {
    out->graph_nodes = session->pending_prefill_graph_nodes;
  }
  session->pending_prefill_available = 0;
  session->pending_prefill_kernel_launches = 0;
  session->pending_prefill_device_elapsed_ns = 0;
  session->pending_prefill_sync_calls = 0;
  session->pending_prefill_graph_replays = 0;
  session->pending_prefill_graph_launches = 0;
  session->pending_prefill_graph_nodes = 0;
}

cudaError_t profile_begin(NervaCudaHfDecodeSequenceSession *session) {
  return cudaEventRecord(session->profile_start, session->stream);
}

cudaError_t profile_end(NervaCudaHfDecodeSequenceSession *session,
                        uint64_t *bucket) {
  cudaError_t err = cudaEventRecord(session->profile_stop, session->stream);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaEventSynchronize(session->profile_stop);
  if (err != cudaSuccess) {
    return err;
  }
  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, session->profile_start,
                             session->profile_stop);
  if (err == cudaSuccess && elapsed_ms > 0.0f) {
    uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_ms * 1000000.0f);
    *bucket += elapsed_ns == 0 ? 1 : elapsed_ns;
  }
  return err;
}

cudaError_t pack_session_weight_replicas(
    NervaCudaHfDecodeSequenceSession *session) {
  if (!use_cublas_layer_path(session)) {
    return cudaSuccess;
  }
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const PackedProjectionShape shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  hf_pack_qkv_weights_kernel<<<
      static_cast<uint32_t>(shape.qkv_rows * session->layer_count),
      kDecodeThreads, 0, session->stream>>>(
      session->device_qkv_packed, session->device_arena,
      session->device_layouts, session->layer_count, session->hidden,
      attention_hidden, kv_hidden);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  hf_pack_gate_up_weights_kernel<<<
      static_cast<uint32_t>(shape.gate_up_rows * session->layer_count),
      kDecodeThreads, 0, session->stream>>>(
      session->device_gate_up_packed, session->device_arena,
      session->device_layouts, session->layer_count, session->hidden,
      session->intermediate);
  return cudaGetLastError();
}

cudaError_t launch_monolithic_session_step(
    NervaCudaHfDecodeSequenceSession *session, uint32_t max_steps,
    uint32_t prompt_token_count, uint32_t has_eos_token, uint32_t eos_token) {
  hf_decode_sequence_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
      session->device_arena, session->arena_layout, session->device_layouts,
      session->layer_count, session->dtype, session->hidden, session->heads,
      session->kv_heads, session->head_dim, session->intermediate, 0,
      session->device_step, max_steps, session->device_prompt_tokens,
      prompt_token_count, session->rms_eps, session->rope_theta,
      session->device_scratch, session->device_kv_keys,
      session->device_kv_values, session->kv_block_count,
      session->device_kv_block_table,
      session->device_slots);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = final_head_gemv(session->cublas, session->device_arena,
                          session->arena_layout, session->dtype,
                          session->hidden, session->vocab_size, device_logits);
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        session->stream, session->device_step, max_steps, has_eos_token,
        eos_token, device_logits, session->vocab_size, session->device_slots,
        session->active_sampler);
  }
  return err;
}

cudaError_t launch_cublas_layer_session_step(
    NervaCudaHfDecodeSequenceSession *session, uint32_t max_steps,
    uint32_t prompt_token_count, uint32_t has_eos_token, uint32_t eos_token,
    uint32_t attention_chunks) {
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const uint32_t decode_head_threads = decode_head_threads_for_session(session);
  cudaError_t err = cudaSuccess;
  uint64_t input_offset = session->arena_layout.input;
  uint64_t output_offset = session->arena_layout.scratch;
  LayerScratch scratch = layer_scratch_ptrs(
      session->device_scratch, session->hidden, attention_hidden, kv_hidden,
      session->intermediate);
  const PackedProjectionShape packed_shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  if (err == cudaSuccess && session->layer_count > 0) {
    const SequenceLayerLayout first_layout = session->host_layouts[0];
    hf_decode_prepare_first_attn_norm_encode_kernel<<<
        1, kDecodeNormThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, first_layout,
        session->dtype, session->hidden, attention_hidden, kv_hidden,
        session->intermediate, session->device_step, max_steps,
        session->device_prompt_tokens, prompt_token_count, session->device_slots,
        session->rms_eps, session->device_scratch,
        session->device_projection_input);
    err = cudaGetLastError();
  } else if (err == cudaSuccess) {
    hf_decode_prepare_input_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, session->hidden,
        session->device_step, max_steps, session->device_prompt_tokens,
        prompt_token_count, session->device_slots);
    err = cudaGetLastError();
  }
  for (uint32_t layer_index = 0;
       err == cudaSuccess && layer_index < session->layer_count;
       ++layer_index) {
    const SequenceLayerLayout layout = session->host_layouts[layer_index];
    err = project_encoded_rows(
        session, &session->qkv_plan,
        session->device_qkv_packed +
            packed_shape.qkv_elements_per_layer * layer_index,
        session->device_projection_input,
        static_cast<uint32_t>(packed_shape.qkv_rows), session->hidden, 1,
        session->dtype, 0.0f, scratch.q);
    if (err == cudaSuccess && attention_chunks == 0) {
      hf_layer_qkv_attention_encode_kernel<<<session->heads, decode_head_threads, 0,
                                             session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
#if NERVA_HAVE_CUDNN_FRONTEND
      const bool use_cudnn_decode_sdpa =
          can_use_cudnn_decode_sdpa(session, attention_chunks);
#else
      const bool use_cudnn_decode_sdpa = false;
#endif
      hf_layer_qkv_prepare_kernel<<<session->heads, decode_head_threads, 0,
                                    session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          use_cudnn_decode_sdpa ? session->device_decode_q : nullptr,
          use_cudnn_decode_sdpa ? session->device_decode_seq_len_q : nullptr,
          use_cudnn_decode_sdpa ? session->device_decode_seq_len_kv : nullptr);
      err = cudaGetLastError();
      bool ran_cudnn_decode_sdpa = false;
#if NERVA_HAVE_CUDNN_FRONTEND
      if (err == cudaSuccess && use_cudnn_decode_sdpa) {
        err = execute_cudnn_decode_sdpa(session, layer_index);
        if (err == cudaSuccess) {
          ran_cudnn_decode_sdpa = true;
        }
      }
#endif
      if (err == cudaSuccess && !ran_cudnn_decode_sdpa) {
        const uint32_t query_group = session->heads / session->kv_heads;
        const bool use_shared_warp_gqa =
            query_group == kGroupedGqaHeads &&
            session->heads % session->kv_heads == 0 &&
            session->head_dim <= kSharedWarpGqaHeadDimMax;
        const bool use_grouped_gqa =
            query_group == kGroupedGqaHeads &&
            session->heads % session->kv_heads == 0 &&
            session->head_dim <= kGroupedGqaHeadDimMax;
        const dim3 grid((use_shared_warp_gqa || use_grouped_gqa)
                            ? session->kv_heads
                            : session->heads,
                        attention_chunks);
        launch_hf_layer_attention_chunk_kernel(
            session->stream, grid, session->dtype, use_shared_warp_gqa,
            use_grouped_gqa, decode_head_threads, layer_index, session->hidden,
            session->heads, session->kv_heads, session->head_dim,
            session->intermediate, session->device_step, max_steps,
            attention_chunks, session->device_scratch,
            session->device_kv_keys, session->device_kv_values,
            session->device_decode_attention_values,
            session->device_decode_attention_m,
            session->device_decode_attention_l, session->kv_block_count,
            session->device_kv_block_table);

        err = cudaGetLastError();
      }
      if (err == cudaSuccess && !ran_cudnn_decode_sdpa) {
        const size_t reduce_shared_bytes =
            static_cast<size_t>(attention_chunks) * sizeof(float);
        hf_layer_attention_reduce_kernel<<<session->heads, decode_head_threads,
                                           reduce_shared_bytes,
                                           session->stream>>>(
            session->dtype, session->hidden, session->heads, session->kv_heads,
            session->head_dim, session->intermediate, session->device_step,
            max_steps, attention_chunks, session->device_scratch,
            session->device_decode_attention_values,
            session->device_decode_attention_m,
            session->device_decode_attention_l,
            session->device_projection_input);
        err = cudaGetLastError();
      }
    }
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->attention_output_plan,
          session->device_arena + layout.w_o, session->device_projection_input,
          session->hidden, attention_hidden, 1, session->dtype, 0.0f,
          scratch.residual);
    if (err == cudaSuccess) {
      hf_layer_mlp_norm_encode_kernel<<<1, kDecodeNormThreads, 0,
                                        session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->gate_up_plan,
          session->device_gate_up_packed +
              packed_shape.gate_up_elements_per_layer * layer_index,
          session->device_projection_input,
          static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden, 1,
          session->dtype, 0.0f, scratch.gate);
    if (err == cudaSuccess) {
      const uint32_t ff_blocks =
          (session->intermediate + kDecodeThreads - 1) / kDecodeThreads;
      hf_layer_ff_encode_kernel<<<ff_blocks, kDecodeThreads, 0,
                                  session->stream>>>(
          session->dtype, session->hidden, attention_hidden, kv_hidden,
          session->intermediate, session->device_step, max_steps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->down_plan,
          session->device_arena + layout.w_down, session->device_projection_input,
          session->hidden, session->intermediate, 1, session->dtype, 0.0f,
          scratch.down);
    if (err == cudaSuccess && layer_index + 1 < session->layer_count) {
      const SequenceLayerLayout next_layout =
          session->host_layouts[layer_index + 1];
      hf_layer_finish_next_attn_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, output_offset, next_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
      hf_layer_finish_final_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, session->arena_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    const uint64_t next_input = output_offset;
    output_offset = input_offset;
    input_offset = next_input;
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = project_encoded_rows(
        session, &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        1, session->dtype, 0.0f, device_logits);
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        session->stream, session->device_step, max_steps, has_eos_token,
        eos_token, device_logits, session->vocab_size, session->device_slots,
        session->active_sampler);
  }
  return err;
}

cudaError_t profile_cublas_layer_session_step(
    NervaCudaHfDecodeSequenceSession *session, uint32_t max_steps,
    uint32_t prompt_token_count, uint32_t has_eos_token, uint32_t eos_token,
    uint32_t attention_chunks, uint32_t cursor) {
  uint64_t projection_ns = 0;
  uint64_t qkv_projection_ns = 0;
  uint64_t attention_output_projection_ns = 0;
  uint64_t gate_up_projection_ns = 0;
  uint64_t down_projection_ns = 0;
  uint64_t lm_head_projection_ns = 0;
  uint64_t attention_ns = 0;
  uint64_t mlp_ns = 0;
  uint64_t norm_ns = 0;
  uint64_t sampling_ns = 0;
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const uint32_t decode_head_threads = decode_head_threads_for_session(session);
  const PackedProjectionShape packed_shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  LayerScratch scratch = layer_scratch_ptrs(
      session->device_scratch, session->hidden, attention_hidden, kv_hidden,
      session->intermediate);

  hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(session->device_step,
                                                          cursor);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) err = profile_begin(session);
  if (err == cudaSuccess && session->layer_count > 0) {
    const SequenceLayerLayout first_layout = session->host_layouts[0];
    hf_decode_prepare_first_attn_norm_encode_kernel<<<
        1, kDecodeNormThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, first_layout,
        session->dtype, session->hidden, attention_hidden, kv_hidden,
        session->intermediate, session->device_step, max_steps,
        session->device_prompt_tokens, prompt_token_count, session->device_slots,
        session->rms_eps, session->device_scratch,
        session->device_projection_input);
    err = cudaGetLastError();
  } else if (err == cudaSuccess) {
    hf_decode_prepare_input_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, session->hidden,
        session->device_step, max_steps, session->device_prompt_tokens,
        prompt_token_count, session->device_slots);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = profile_end(session, &norm_ns);

  uint64_t input_offset = session->arena_layout.input;
  uint64_t output_offset = session->arena_layout.scratch;
  for (uint32_t layer_index = 0;
       err == cudaSuccess && layer_index < session->layer_count;
       ++layer_index) {
    const SequenceLayerLayout layout = session->host_layouts[layer_index];
    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->qkv_plan,
          session->device_qkv_packed +
              packed_shape.qkv_elements_per_layer * layer_index,
          session->device_projection_input,
          static_cast<uint32_t>(packed_shape.qkv_rows), session->hidden, 1,
          session->dtype, 0.0f, scratch.q);
    if (err == cudaSuccess) err = profile_end(session, &qkv_projection_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess && attention_chunks == 0) {
      hf_layer_qkv_attention_encode_kernel<<<session->heads, decode_head_threads, 0,
                                             session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
#if NERVA_HAVE_CUDNN_FRONTEND
      const bool use_cudnn_decode_sdpa =
          can_use_cudnn_decode_sdpa(session, attention_chunks);
#else
      const bool use_cudnn_decode_sdpa = false;
#endif
      hf_layer_qkv_prepare_kernel<<<session->heads, decode_head_threads, 0,
                                    session->stream>>>(
          session->device_arena, layout, layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          use_cudnn_decode_sdpa ? session->device_decode_q : nullptr,
          use_cudnn_decode_sdpa ? session->device_decode_seq_len_q : nullptr,
          use_cudnn_decode_sdpa ? session->device_decode_seq_len_kv : nullptr);
      err = cudaGetLastError();
      bool ran_cudnn_decode_sdpa = false;
#if NERVA_HAVE_CUDNN_FRONTEND
      if (err == cudaSuccess && use_cudnn_decode_sdpa) {
        err = execute_cudnn_decode_sdpa(session, layer_index);
        if (err == cudaSuccess) {
          ran_cudnn_decode_sdpa = true;
        }
      }
#endif
      if (err == cudaSuccess && !ran_cudnn_decode_sdpa) {
        const uint32_t query_group = session->heads / session->kv_heads;
        const bool use_shared_warp_gqa =
            query_group == kGroupedGqaHeads &&
            session->heads % session->kv_heads == 0 &&
            session->head_dim <= kSharedWarpGqaHeadDimMax;
        const bool use_grouped_gqa =
            query_group == kGroupedGqaHeads &&
            session->heads % session->kv_heads == 0 &&
            session->head_dim <= kGroupedGqaHeadDimMax;
        const dim3 grid((use_shared_warp_gqa || use_grouped_gqa)
                            ? session->kv_heads
                            : session->heads,
                        attention_chunks);
        launch_hf_layer_attention_chunk_kernel(
            session->stream, grid, session->dtype, use_shared_warp_gqa,
            use_grouped_gqa, decode_head_threads, layer_index, session->hidden,
            session->heads, session->kv_heads, session->head_dim,
            session->intermediate, session->device_step, max_steps,
            attention_chunks, session->device_scratch,
            session->device_kv_keys, session->device_kv_values,
            session->device_decode_attention_values,
            session->device_decode_attention_m,
            session->device_decode_attention_l, session->kv_block_count,
            session->device_kv_block_table);

        err = cudaGetLastError();
      }
      if (err == cudaSuccess && !ran_cudnn_decode_sdpa) {
        const size_t reduce_shared_bytes =
            static_cast<size_t>(attention_chunks) * sizeof(float);
        hf_layer_attention_reduce_kernel<<<session->heads, decode_head_threads,
                                           reduce_shared_bytes,
                                           session->stream>>>(
            session->dtype, session->hidden, session->heads, session->kv_heads,
            session->head_dim, session->intermediate, session->device_step,
            max_steps, attention_chunks, session->device_scratch,
            session->device_decode_attention_values,
            session->device_decode_attention_m,
            session->device_decode_attention_l,
            session->device_projection_input);
        err = cudaGetLastError();
      }
    }
    if (err == cudaSuccess) err = profile_end(session, &attention_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->attention_output_plan,
          session->device_arena + layout.w_o, session->device_projection_input,
          session->hidden, attention_hidden, 1, session->dtype, 0.0f,
          scratch.residual);
    if (err == cudaSuccess) {
      err = profile_end(session, &attention_output_projection_ns);
    }

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess) {
      hf_layer_mlp_norm_encode_kernel<<<1, kDecodeNormThreads, 0,
                                        session->stream>>>(
          session->device_arena, layout, session->dtype, session->hidden,
          attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = profile_end(session, &norm_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->gate_up_plan,
          session->device_gate_up_packed +
              packed_shape.gate_up_elements_per_layer * layer_index,
          session->device_projection_input,
          static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden, 1,
          session->dtype, 0.0f, scratch.gate);
    if (err == cudaSuccess) err = profile_end(session, &gate_up_projection_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess) {
      const uint32_t ff_blocks =
          (session->intermediate + kDecodeThreads - 1) / kDecodeThreads;
      hf_layer_ff_encode_kernel<<<ff_blocks, kDecodeThreads, 0,
                                  session->stream>>>(
          session->dtype, session->hidden, attention_hidden, kv_hidden,
          session->intermediate, session->device_step, max_steps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = profile_end(session, &mlp_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess)
      err = project_encoded_rows(
          session, &session->down_plan,
          session->device_arena + layout.w_down, session->device_projection_input,
          session->hidden, session->intermediate, 1, session->dtype, 0.0f,
          scratch.down);
    if (err == cudaSuccess) err = profile_end(session, &down_projection_ns);

    if (err == cudaSuccess) err = profile_begin(session);
    if (err == cudaSuccess && layer_index + 1 < session->layer_count) {
      const SequenceLayerLayout next_layout =
          session->host_layouts[layer_index + 1];
      hf_layer_finish_next_attn_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, output_offset, next_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    } else if (err == cudaSuccess) {
      hf_layer_finish_final_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, session->stream>>>(
          session->device_arena, session->arena_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, max_steps, session->rms_eps,
          session->device_scratch, session->device_projection_input);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = profile_end(session, &norm_ns);
    const uint64_t next_input = output_offset;
    output_offset = input_offset;
    input_offset = next_input;
  }

  if (err == cudaSuccess) err = profile_begin(session);
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = project_encoded_rows(
        session, &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        1, session->dtype, 0.0f, device_logits);
  }
  if (err == cudaSuccess) err = profile_end(session, &lm_head_projection_ns);

  if (err == cudaSuccess) err = profile_begin(session);
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        session->stream, session->device_step, max_steps, has_eos_token,
        eos_token, device_logits, session->vocab_size, session->device_slots,
        session->active_sampler);
  }
  if (err == cudaSuccess) err = profile_end(session, &sampling_ns);

  if (err == cudaSuccess) {
    hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(
        session->device_step, cursor);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(session->stream);
  if (err == cudaSuccess) {
    projection_ns = qkv_projection_ns + attention_output_projection_ns +
                    gate_up_projection_ns + down_projection_ns +
                    lm_head_projection_ns;
    session->cached_projection_ns = projection_ns;
    session->cached_qkv_projection_ns = qkv_projection_ns;
    session->cached_attention_output_projection_ns =
        attention_output_projection_ns;
    session->cached_gate_up_projection_ns = gate_up_projection_ns;
    session->cached_down_projection_ns = down_projection_ns;
    session->cached_lm_head_projection_ns = lm_head_projection_ns;
    session->cached_attention_ns = attention_ns;
    session->cached_mlp_ns = mlp_ns;
    session->cached_norm_ns = norm_ns;
    session->cached_sampling_ns = sampling_ns;
  }
  return err;
}

cudaError_t ensure_session_graph(NervaCudaHfDecodeSequenceSession *session,
                                 uint32_t max_steps,
                                 uint32_t prompt_token_count,
                                 uint32_t has_eos_token,
                                 uint32_t eos_token,
                                 uint32_t attention_chunks,
                                 uint32_t profile_cursor,
                                 NervaCudaHfDecodeSequenceResult *out);

cudaError_t encoded_row_major_gemm_tokens_cached(
    NervaCudaHfDecodeSequenceSession *session, const uint16_t *matrix,
    const uint16_t *input, uint32_t rows, uint32_t cols, uint32_t tokens,
    uint32_t dtype, float beta, float *output) {
  if (session == nullptr) {
    return cudaErrorInvalidValue;
  }
  return encoded_row_major_gemm_tokens(session->cublas, matrix, input, rows,
                                       cols, tokens, dtype, beta, output);
}

cudaError_t launch_cublas_session_prefill(
    NervaCudaHfDecodeSequenceSession *session, uint32_t prompt_token_count,
    uint32_t has_eos_token, uint32_t eos_token,
    NervaCudaHfDecodeSequenceResult *out) {
  if (prompt_token_count == 0 || prompt_token_count > session->max_context_tokens ||
      !use_cublas_layer_path(session)) {
    return cudaErrorInvalidValue;
  }
  const uint32_t attention_hidden = session->heads * session->head_dim;
  const uint32_t kv_hidden = session->kv_heads * session->head_dim;
  const PackedProjectionShape packed_shape = packed_projection_shape(
      session->hidden, attention_hidden, kv_hidden, session->intermediate);
  const bool collect_profile = session->detailed_profile != 0;
  uint64_t qkv_projection_ns = 0;
  uint64_t attention_output_projection_ns = 0;
  uint64_t gate_up_projection_ns = 0;
  uint64_t down_projection_ns = 0;
  uint64_t lm_head_projection_ns = 0;
  uint64_t attention_ns = 0;
  uint64_t mlp_ns = 0;
  uint64_t norm_ns = 0;
  uint64_t sampling_ns = 0;
  auto profile_stage_begin = [&]() -> cudaError_t {
    return collect_profile ? profile_begin(session) : cudaSuccess;
  };
  auto profile_stage_end = [&](uint64_t *bucket) -> cudaError_t {
    return collect_profile ? profile_end(session, bucket) : cudaSuccess;
  };
  cudaError_t err = cudaEventRecord(session->device_start, session->stream);
  if (err == cudaSuccess) {
    hf_prefill_embed_kernel<<<prompt_token_count, kDecodeThreads, 0,
                              session->stream>>>(
        session->device_arena, session->arena_layout, session->hidden,
        session->device_prompt_tokens, prompt_token_count,
        session->device_prefill_hidden_a);
    err = cudaGetLastError();
    out->kernel_launches += 1;
  }
  uint16_t *hidden_in = session->device_prefill_hidden_a;
  uint16_t *hidden_out = session->device_prefill_hidden_b;
  for (uint32_t layer_index = 0;
       err == cudaSuccess && layer_index < session->layer_count;
       ++layer_index) {
    const SequenceLayerLayout layout = session->host_layouts[layer_index];
    for (uint32_t chunk_start = 0;
         err == cudaSuccess && chunk_start < prompt_token_count;
         chunk_start += session->prefill_chunk_tokens) {
      const uint32_t chunk_tokens =
          std::min(session->prefill_chunk_tokens, prompt_token_count - chunk_start);
      if (err == cudaSuccess) err = profile_stage_begin();
      if (err == cudaSuccess) {
        hf_prefill_attn_norm_kernel<<<chunk_tokens, kDecodeThreads, 0,
                                      session->stream>>>(
            session->device_arena, layout, session->dtype, session->hidden,
            chunk_start, chunk_tokens, session->rms_eps, hidden_in,
            session->device_prefill_norm);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&norm_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        err = project_encoded_rows(
            session, &session->qkv_plan,
            session->device_qkv_packed +
                packed_shape.qkv_elements_per_layer * layer_index,
            session->device_prefill_norm,
            static_cast<uint32_t>(packed_shape.qkv_rows), session->hidden,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_qkv);
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&qkv_projection_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      const uint32_t query_group =
          session->kv_heads == 0 ? 0 : session->heads / session->kv_heads;
      const bool use_grouped_gqa =
          query_group == kGroupedGqaHeads &&
          session->heads % session->kv_heads == 0 &&
          session->head_dim <= kSharedWarpGqaHeadDimMax;
#if NERVA_HAVE_CUDNN_FRONTEND
      const bool use_cudnn_sdpa =
          session->cudnn_prefill_sdpa_disabled == 0 &&
          session->cudnn != nullptr &&
          session->device_prefill_qkv_encoded != nullptr &&
          session->dtype == kDTypeBF16 && use_grouped_gqa &&
          chunk_start == 0 && chunk_tokens == prompt_token_count &&
          session->head_dim <= 128;
#endif
      if (err == cudaSuccess) {
        const dim3 grid(chunk_tokens, std::max(session->heads, session->kv_heads));
        hf_prefill_qkv_publish_kernel<<<grid, session->head_threads, 0,
                                      session->stream>>>(
            session->device_arena, layout, layer_index, session->dtype,
            session->heads, session->kv_heads, session->head_dim,
            session->max_context_tokens, chunk_start, chunk_tokens,
            session->rms_eps, session->rope_theta, session->device_prefill_qkv,
            session->device_kv_keys, session->device_kv_values,
#if NERVA_HAVE_CUDNN_FRONTEND
            use_cudnn_sdpa ? session->device_prefill_qkv_encoded : nullptr,
#else
            nullptr,
#endif
            session->kv_block_count, session->device_kv_block_table);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&norm_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
#if NERVA_HAVE_CUDNN_FRONTEND
        bool ran_cudnn_sdpa = false;
        if (use_cudnn_sdpa) {
          err = execute_cudnn_prefill_sdpa(session, chunk_tokens);
          if (err == cudaSuccess) {
            out->kernel_launches += 1;
            ran_cudnn_sdpa = true;
          } else if (err == cudaErrorNotSupported ||
                     err == cudaErrorMemoryAllocation) {
            session->cudnn_prefill_sdpa_disabled = 1;
            err = cudaSuccess;
          }
        }
        if (!ran_cudnn_sdpa) {
#endif
        if (use_grouped_gqa) {
          const dim3 grid(chunk_tokens, session->kv_heads);
          launch_hf_prefill_grouped_gqa_attention_direct_kernel(
              session->stream, grid, session->dtype, layer_index,
              session->heads, session->kv_heads, session->head_dim,
              session->max_context_tokens, chunk_start, chunk_tokens,
              session->device_prefill_qkv, session->device_kv_keys,
              session->device_kv_values, session->kv_block_count,
              session->device_kv_block_table, session->device_prefill_attn);
        } else {
          const dim3 grid(chunk_tokens, session->heads);
          hf_prefill_attention_kernel<<<grid, session->head_threads,
                                        session->head_dim * sizeof(float),
                                        session->stream>>>(
              layer_index, session->dtype, session->heads, session->kv_heads,
              session->head_dim, session->max_context_tokens, chunk_start,
              chunk_tokens, session->device_prefill_qkv, session->device_kv_keys,
              session->device_kv_values, session->kv_block_count,
              session->device_kv_block_table, session->device_prefill_attn);
        }
          err = cudaGetLastError();
          out->kernel_launches += 1;
#if NERVA_HAVE_CUDNN_FRONTEND
        }
#endif
      }
      if (err == cudaSuccess) err = profile_stage_end(&attention_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        err = project_encoded_rows(
            session, &session->attention_output_plan,
            session->device_arena + layout.w_o,
            session->device_prefill_attn, session->hidden, attention_hidden,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_o);
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) {
        err = profile_stage_end(&attention_output_projection_ns);
      }
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        hf_prefill_mlp_norm_kernel<<<chunk_tokens, kDecodeThreads, 0,
                                     session->stream>>>(
            session->device_arena, layout, session->dtype, session->hidden,
            chunk_start, chunk_tokens, session->rms_eps, hidden_in,
            session->device_prefill_o, session->device_prefill_norm);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&norm_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        err = project_encoded_rows(
            session, &session->gate_up_plan,
            session->device_gate_up_packed +
                packed_shape.gate_up_elements_per_layer * layer_index,
            session->device_prefill_norm,
            static_cast<uint32_t>(packed_shape.gate_up_rows), session->hidden,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_gate_up);
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&gate_up_projection_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        const uint32_t blocks =
            static_cast<uint32_t>(
                (static_cast<uint64_t>(chunk_tokens) * session->intermediate +
                 kDecodeThreads - 1) /
                kDecodeThreads);
        hf_prefill_ff_kernel<<<blocks, kDecodeThreads, 0, session->stream>>>(
            session->dtype, session->intermediate, chunk_tokens,
            session->device_prefill_gate_up, session->device_prefill_ff);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&mlp_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        err = project_encoded_rows(
            session, &session->down_plan,
            session->device_arena + layout.w_down,
            session->device_prefill_ff, session->hidden, session->intermediate,
            chunk_tokens, session->dtype, 0.0f, session->device_prefill_down);
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&down_projection_ns);
      if (err == cudaSuccess) {
        err = profile_stage_begin();
      }
      if (err == cudaSuccess) {
        const uint32_t blocks =
            static_cast<uint32_t>(
                (static_cast<uint64_t>(chunk_tokens) * session->hidden +
                 kDecodeThreads - 1) /
                kDecodeThreads);
        hf_prefill_finish_kernel<<<blocks, kDecodeThreads, 0, session->stream>>>(
            session->dtype, session->hidden, chunk_start, chunk_tokens,
            session->device_prefill_o, session->device_prefill_down, hidden_out);
        err = cudaGetLastError();
        out->kernel_launches += 1;
      }
      if (err == cudaSuccess) err = profile_stage_end(&mlp_ns);
    }
    std::swap(hidden_in, hidden_out);
  }
  if (err == cudaSuccess) {
    err = profile_stage_begin();
  }
  if (err == cudaSuccess) {
    hf_prefill_final_norm_last_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
        session->device_arena, session->arena_layout, session->dtype,
        session->hidden, prompt_token_count, session->rms_eps, hidden_in,
        session->device_projection_input);
    err = cudaGetLastError();
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) err = profile_stage_end(&norm_ns);
  if (err == cudaSuccess) {
    hf_decode_set_step_kernel<<<1, 1, 0, session->stream>>>(
        session->device_step, prompt_token_count - 1u);
    err = cudaGetLastError();
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) {
    err = profile_stage_begin();
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = project_encoded_rows(
        session, &session->lm_head_plan,
        session->device_arena + session->arena_layout.lm_head,
        session->device_projection_input, session->vocab_size, session->hidden,
        1, session->dtype, 0.0f, device_logits);
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) err = profile_stage_end(&lm_head_projection_ns);
  if (err == cudaSuccess) {
    err = profile_stage_begin();
  }
  if (err == cudaSuccess) {
    float *device_logits = session->device_scratch + session->hidden * 2;
    err = launch_hf_decode_final_head_sampler(
        session->stream, session->device_step, session->max_context_tokens,
        has_eos_token, eos_token, device_logits, session->vocab_size,
        session->device_slots, session->active_sampler);
    out->kernel_launches += 1;
  }
  if (err == cudaSuccess) err = profile_stage_end(&sampling_ns);
  if (err == cudaSuccess) {
    err = cudaEventRecord(session->device_stop, session->stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(session->stream);
    out->sync_calls += 1;
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
  if (err == cudaSuccess && collect_profile) {
    out->qkv_projection_ns = qkv_projection_ns;
    out->attention_output_projection_ns = attention_output_projection_ns;
    out->gate_up_projection_ns = gate_up_projection_ns;
    out->down_projection_ns = down_projection_ns;
    out->lm_head_projection_ns = lm_head_projection_ns;
    out->projection_ns = qkv_projection_ns + attention_output_projection_ns +
                         gate_up_projection_ns + down_projection_ns +
                         lm_head_projection_ns;
    out->attention_ns = attention_ns;
    out->mlp_ns = mlp_ns;
    out->norm_ns = norm_ns;
    out->sampling_ns = sampling_ns;
  }
  return err;
}

cudaError_t launch_serial_session_prefill(
    NervaCudaHfDecodeSequenceSession *session, uint32_t prompt_token_count,
    uint32_t has_eos_token, uint32_t eos_token,
    NervaCudaHfDecodeSequenceResult *out) {
  cudaError_t err =
      ensure_session_graph(session, session->max_context_tokens, prompt_token_count,
                           has_eos_token, eos_token, 0, 0, out);
  if (err == cudaSuccess) {
    err = cudaEventRecord(session->device_start, session->stream);
  }
  for (uint32_t step = 0; err == cudaSuccess && step < prompt_token_count; ++step) {
    err = cudaGraphLaunch(session->cached_graph_exec, session->stream);
    if (err == cudaSuccess) {
      out->graph_replays += 1;
      out->graph_launches += 1;
      out->kernel_launches += out->graph_nodes == 0 ? 1 : out->graph_nodes;
    }
  }
  if (err == cudaSuccess) {
    err = cudaEventRecord(session->device_stop, session->stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(session->stream);
    out->sync_calls += 1;
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
  return err;
}

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

}  // namespace

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
  const uint64_t block_scratch =
      hidden * 5 + attention_hidden * 2 + kv_hidden * 2 + intermediate * 3;
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
        device_kv_block_table, device_slots);
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
        kv_block_table_bytes + prompt_bytes + slots_bytes + sizeof(uint32_t) +
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
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    if (!valid_layer(request->layers[index], !descriptor_mode)) {
      out->failure_stage = kCreateStageInvalidRequest;
      return -1;
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
  if (request->experimental_rt_decode != 0) {
    out->failure_stage = kCreateStageExperimentalRtDecodeUnsupported;
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
  std::vector<SequenceLayerLayout> layouts(request->layer_count);
  uint64_t elements = 0;
  session->arena_layout.embeddings = push(elements, vocab_size * hidden);
  session->arena_layout.input = push(elements, hidden);
  session->arena_layout.scratch = push(elements, hidden);
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    pack_layer(layouts[index], elements, request->layers[index], hidden,
               attention_hidden, kv_hidden, request->head_dim, intermediate);
  }
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

  const uint64_t block_scratch =
      hidden * 5 + attention_hidden * 2 + kv_hidden * 2 + intermediate * 3;
  const uint64_t final_scratch = hidden * 2 + vocab_size;
  const uint64_t scratch_elements =
      block_scratch > final_scratch ? block_scratch : final_scratch;
  const uint64_t projection_input_elements =
      intermediate > attention_hidden
          ? (intermediate > hidden ? intermediate : hidden)
          : (attention_hidden > hidden ? attention_hidden : hidden);
  const uint64_t prefill_qkv_rows = attention_hidden + kv_hidden * 2;
  const uint64_t prefill_gate_up_rows = intermediate * 2;
  const bool pack_cublas =
      should_pack_cublas_weights(request->hidden, attention_hidden);
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
  session->experimental_rt_page_tokens = request->experimental_rt_page_tokens;
  session->experimental_rt_pages = request->experimental_rt_pages;
  session->experimental_rt_local_window_tokens =
      request->experimental_rt_local_window_tokens;
  session->experimental_rt_sink_tokens = request->experimental_rt_sink_tokens;
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
      static_cast<uint64_t>(request->heads) *
      session->decode_attention_max_chunks * request->head_dim * sizeof(float);
  session->decode_attention_stats_bytes =
      static_cast<uint64_t>(request->heads) *
      session->decode_attention_max_chunks * sizeof(float);
  session->decode_q_bytes =
      static_cast<uint64_t>(attention_hidden) * sizeof(uint16_t);
  session->decode_seq_len_bytes = sizeof(int32_t) * 2u;
  if (pack_cublas) {
    session->packed_qkv_bytes =
        packed_shape.qkv_elements_per_layer * request->layer_count *
        sizeof(uint16_t);
    session->packed_gate_up_bytes =
        packed_shape.gate_up_elements_per_layer * request->layer_count *
        sizeof(uint16_t);
  }
  session->kv_bytes =
      request->layer_count * static_cast<uint64_t>(session->kv_token_capacity) *
      kv_hidden * sizeof(uint16_t) * 2;
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
      prefill_qkv_rows, attention_hidden, hidden, prefill_gate_up_rows,
      intermediate, static_cast<uint64_t>(device_free_before_alloc));
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
      hf_decode_sequence_kernel<<<1, kDecodeThreads, 0, session->stream>>>(
          session->device_arena, session->arena_layout, session->device_layouts,
          session->layer_count, session->dtype, session->hidden, session->heads,
          session->kv_heads, session->head_dim, session->intermediate, 0,
          session->device_step, context_steps, session->device_prompt_tokens,
          request->prompt_token_count, session->rms_eps, session->rope_theta,
          session->device_scratch, session->device_kv_keys,
          session->device_kv_values, session->kv_block_count,
          session->device_kv_block_table,
          session->device_slots);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
      float *device_logits = session->device_scratch + session->hidden * 2;
      err = final_head_gemv(session->cublas, session->device_arena,
                            session->arena_layout, session->dtype,
                            session->hidden, session->vocab_size,
                            device_logits);
    }
    if (err == cudaSuccess) {
      float *device_logits = session->device_scratch + session->hidden * 2;
      err = launch_hf_decode_final_head_sampler(
          session->stream, session->device_step, context_steps,
          request->has_eos_token, request->eos_token, device_logits,
          session->vocab_size, session->device_slots, session->active_sampler);
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
    err = cudaGraphLaunch(session->cached_graph_exec, session->stream);
    if (err == cudaSuccess) {
      out->graph_replays += 1;
      out->graph_launches += 1;
      out->kernel_launches += out->graph_nodes == 0 ? 1 : out->graph_nodes;
    }
  }
  if (err == cudaSuccess) err = cudaEventRecord(session->device_stop, session->stream);
  const uint64_t slots_bytes =
      static_cast<uint64_t>(context_steps) * sizeof(NervaCudaSyntheticTokenSlot);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(session->host_slots, session->device_slots, slots_bytes,
                          cudaMemcpyDeviceToHost, session->stream);
    out->d2h_bytes = slots_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(session->stream);
    out->sync_calls += 1;
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
    out->resident_kv_bytes = session->kv_bytes;
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
  cudaError_t err = cudaMemsetAsync(session->device_slots, 0,
                                    session->slots_bytes, session->stream);
  if (err == cudaSuccess)
    err = cudaMemsetAsync(session->device_step, 0, sizeof(uint32_t),
                          session->stream);
  const uint64_t prompt_bytes =
      static_cast<uint64_t>(request->prompt_token_count) * sizeof(uint32_t);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(session->device_prompt_tokens, request->prompt_tokens,
                          prompt_bytes, cudaMemcpyHostToDevice, session->stream);
    out->h2d_bytes = prompt_bytes;
  }
  if (err == cudaSuccess) {
    err = use_cublas_layer_path(session)
              ? launch_cublas_session_prefill(
                    session, request->prompt_token_count,
                    request->has_eos_token, request->eos_token, out)
              : launch_serial_session_prefill(
                    session, request->prompt_token_count,
                    request->has_eos_token, request->eos_token, out);
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
    out->resident_kv_bytes = session->kv_bytes;
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
  const uint32_t run_count = target_cursor - session->active_cursor;
  const uint32_t seed_token =
      session->active_observed_tokens == 0
          ? session->active_seed_token
          : session->host_slots[slot_start - 1u].token;
  fill_session_result_header(session, out, request->steps, seed_token);

  cudaError_t err = cudaSuccess;
  if (run_count != 0 && projection_batch_session_ready(session) &&
      !use_cublas_layer_path(session)) {
    err = ensure_session_cublas_resources(session);
  }
  if (run_count != 0) {
    const uint32_t attention_chunks =
        decode_attention_chunks_for_cursor(session, target_cursor);
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
    err = cudaGraphLaunch(session->cached_graph_exec, session->stream);
    if (err == cudaSuccess) {
      out->graph_replays += 1;
      out->graph_launches += 1;
      out->kernel_launches += out->graph_nodes == 0 ? 1 : out->graph_nodes;
    }
  }
  if (err == cudaSuccess && run_count != 0)
    err = cudaEventRecord(session->device_stop, session->stream);
  const uint64_t slots_bytes =
      static_cast<uint64_t>(request->steps) * sizeof(NervaCudaSyntheticTokenSlot);
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(session->host_slots + slot_start,
                          session->device_slots + slot_start, slots_bytes,
                          cudaMemcpyDeviceToHost, session->stream);
    out->d2h_bytes = slots_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(session->stream);
    out->sync_calls += 1;
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
    out->resident_kv_bytes = session->kv_bytes;
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

extern "C" int nerva_cuda_hf_decode_sequence_layer_projection_batch_execute(
    const NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteRequest *request,
    NervaCudaHfDecodeSequenceLayerProjectionBatchExecuteResult *out) {
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
  out->layer_index = request->layer_index;
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
        request->layer_index >= candidate->layer_count) {
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
  if (request->layer_index >= best->layer_count) {
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
  std::vector<NervaCudaHfDecodeSequenceSession *> selected;
  selected.reserve(max_block_tokens);
  for (uint32_t index = 0; index < request->session_count &&
                           selected.size() < max_block_tokens;
       ++index) {
    NervaCudaHfDecodeSequenceSession *session = request->sessions[index];
    if (ready_session(session) && same_projection_model(best, session)) {
      selected.push_back(session);
    }
  }
  if (selected.size() < out->min_block_tokens) {
    out->reason = kProjectionBatchPlanInsufficientCompatibleReady;
    out->status = 0;
    return 0;
  }
  out->block_tokens = static_cast<uint32_t>(selected.size());
  out->dtype = best->dtype;

  if (best->projection_batch_peer_streams_synchronized == 0) {
    for (NervaCudaHfDecodeSequenceSession *session : selected) {
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
  }
  ScopedProjectionBatchFlags layer_scope(best, true, false);

  auto run_stage =
      [&](uint32_t projection_kind,
          NervaCudaHfDecodeSequenceProjectionBatchExecuteResult *stage_out)
          -> int {
    NervaCudaHfDecodeSequenceProjectionBatchExecuteRequest stage_request{};
    stage_request.sessions = request->sessions;
    stage_request.session_count = request->session_count;
    stage_request.target_block_tokens = request->target_block_tokens;
    stage_request.min_block_tokens = request->min_block_tokens;
    stage_request.projection_kind = projection_kind;
    stage_request.layer_index = request->layer_index;
    const int rc = nerva_cuda_hf_decode_sequence_projection_batch_execute(
        &stage_request, stage_out);
    out->cuda_error = stage_out->cuda_error;
    out->device_count = stage_out->device_count;
    out->reason = stage_out->reason;
    out->eligible_session_count = stage_out->eligible_session_count;
    out->block_tokens = stage_out->block_tokens;
    out->target_block_tokens = stage_out->target_block_tokens;
    out->min_block_tokens = stage_out->min_block_tokens;
    out->dtype = stage_out->dtype;
    return rc;
  };

  auto launch_attention_encode =
      [&](NervaCudaHfDecodeSequenceSession *session) -> cudaError_t {
    const uint32_t attention_hidden = session->heads * session->head_dim;
    const uint32_t kv_hidden = session->kv_heads * session->head_dim;
    const uint32_t decode_head_threads = decode_head_threads_for_session(session);
    const uint32_t attention_chunks =
        decode_attention_chunks_for_cursor(session, session->active_cursor);
    const SequenceLayerLayout layout =
        session->host_layouts[request->layer_index];
    const uint32_t max_steps = session->max_context_tokens;
    if (attention_chunks == 0) {
      hf_layer_qkv_attention_encode_kernel<<<session->heads,
                                             decode_head_threads, 0,
                                             best->stream>>>(
          session->device_arena, layout, request->layer_index, session->dtype,
          session->hidden, session->heads, session->kv_heads, session->head_dim,
          session->intermediate, session->device_step, max_steps,
          session->rms_eps, session->rope_theta, session->device_scratch,
          session->device_kv_keys, session->device_kv_values,
          session->kv_block_count, session->device_kv_block_table,
          session->device_projection_input);
      out->dependency_kernel_launches += 1;
      return cudaGetLastError();
    }

    hf_layer_qkv_prepare_kernel<<<session->heads, decode_head_threads, 0,
                                  best->stream>>>(
        session->device_arena, layout, request->layer_index, session->dtype,
        session->hidden, session->heads, session->kv_heads, session->head_dim,
        session->intermediate, session->device_step, max_steps,
        session->rms_eps, session->rope_theta, session->device_scratch,
        session->device_kv_keys, session->device_kv_values,
        session->kv_block_count, session->device_kv_block_table, nullptr,
        nullptr, nullptr);
    out->dependency_kernel_launches += 1;
    cudaError_t local_err = cudaGetLastError();
    if (local_err != cudaSuccess) {
      return local_err;
    }

    const uint32_t query_group = session->heads / session->kv_heads;
    const bool use_shared_warp_gqa =
        query_group == kGroupedGqaHeads &&
        session->heads % session->kv_heads == 0 &&
        session->head_dim <= kSharedWarpGqaHeadDimMax;
    const bool use_grouped_gqa =
        query_group == kGroupedGqaHeads &&
        session->heads % session->kv_heads == 0 &&
        session->head_dim <= kGroupedGqaHeadDimMax;
    const dim3 grid((use_shared_warp_gqa || use_grouped_gqa) ? session->kv_heads
                                                             : session->heads,
                    attention_chunks);
    launch_hf_layer_attention_chunk_kernel(
        best->stream, grid, session->dtype, use_shared_warp_gqa,
        use_grouped_gqa, decode_head_threads, request->layer_index, session->hidden,
        session->heads, session->kv_heads, session->head_dim,
        session->intermediate, session->device_step, max_steps,
        attention_chunks, session->device_scratch,
        session->device_kv_keys, session->device_kv_values,
        session->device_decode_attention_values,
        session->device_decode_attention_m,
        session->device_decode_attention_l, session->kv_block_count,
        session->device_kv_block_table);

    out->dependency_kernel_launches += 1;
    local_err = cudaGetLastError();
    if (local_err != cudaSuccess) {
      return local_err;
    }
    const size_t reduce_shared_bytes =
        static_cast<size_t>(attention_chunks) * sizeof(float);
    hf_layer_attention_reduce_kernel<<<session->heads, decode_head_threads,
                                       reduce_shared_bytes, best->stream>>>(
        session->dtype, session->hidden, session->heads, session->kv_heads,
        session->head_dim, session->intermediate, session->device_step,
        max_steps, attention_chunks, session->device_scratch,
        session->device_decode_attention_values, session->device_decode_attention_m,
        session->device_decode_attention_l, session->device_projection_input);
    out->dependency_kernel_launches += 1;
    return cudaGetLastError();
  };

  if (request->layer_index == 0) {
    for (NervaCudaHfDecodeSequenceSession *session : selected) {
      hf_decode_set_step_kernel<<<1, 1, 0, best->stream>>>(
          session->device_step, session->active_cursor);
      out->dependency_kernel_launches += 1;
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        out->cuda_error = static_cast<int32_t>(err);
        return -1;
      }
      const uint32_t attention_hidden = session->heads * session->head_dim;
      const uint32_t kv_hidden = session->kv_heads * session->head_dim;
      const SequenceLayerLayout first_layout = session->host_layouts[0];
      hf_decode_prepare_first_attn_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, best->stream>>>(
          session->device_arena, session->arena_layout, first_layout,
          session->dtype, session->hidden, attention_hidden, kv_hidden,
          session->intermediate, session->device_step, session->max_context_tokens,
          session->device_prompt_tokens, session->active_prompt_token_count,
          session->device_slots, session->rms_eps, session->device_scratch,
          session->device_projection_input);
      out->dependency_kernel_launches += 1;
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        out->cuda_error = static_cast<int32_t>(err);
        return -1;
      }
    }
  }

  constexpr uint32_t kLayerProjectionKinds[] = {
      kProjectionBatchKindQkv,
      kProjectionBatchKindAttentionOutput,
      kProjectionBatchKindGateUp,
      kProjectionBatchKindDown,
  };
  NervaCudaHfDecodeSequenceProjectionBatchExecuteResult stages[4];

  int rc = run_stage(kLayerProjectionKinds[0], &stages[0]);
  if (rc != 0 || stages[0].status != 0 || stages[0].exact == 0) {
    out->exact = 0;
    out->status = stages[0].status;
    return rc;
  }

  for (NervaCudaHfDecodeSequenceSession *session : selected) {
    err = launch_attention_encode(session);
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  rc = run_stage(kLayerProjectionKinds[1], &stages[1]);
  if (rc != 0 || stages[1].status != 0 || stages[1].exact == 0) {
    out->exact = 0;
    out->status = stages[1].status;
    return rc;
  }

  for (NervaCudaHfDecodeSequenceSession *session : selected) {
    const uint32_t attention_hidden = session->heads * session->head_dim;
    const uint32_t kv_hidden = session->kv_heads * session->head_dim;
    const SequenceLayerLayout layout =
        session->host_layouts[request->layer_index];
    hf_layer_mlp_norm_encode_kernel<<<1, kDecodeNormThreads, 0, best->stream>>>(
        session->device_arena, layout, session->dtype, session->hidden,
        attention_hidden, kv_hidden, session->intermediate, session->device_step,
        session->max_context_tokens, session->rms_eps, session->device_scratch,
        session->device_projection_input);
    out->dependency_kernel_launches += 1;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  rc = run_stage(kLayerProjectionKinds[2], &stages[2]);
  if (rc != 0 || stages[2].status != 0 || stages[2].exact == 0) {
    out->exact = 0;
    out->status = stages[2].status;
    return rc;
  }

  for (NervaCudaHfDecodeSequenceSession *session : selected) {
    const uint32_t attention_hidden = session->heads * session->head_dim;
    const uint32_t kv_hidden = session->kv_heads * session->head_dim;
    const uint32_t ff_blocks =
        (session->intermediate + kDecodeThreads - 1) / kDecodeThreads;
    hf_layer_ff_encode_kernel<<<ff_blocks, kDecodeThreads, 0, best->stream>>>(
        session->dtype, session->hidden, attention_hidden, kv_hidden,
        session->intermediate, session->device_step, session->max_context_tokens,
        session->device_scratch, session->device_projection_input);
    out->dependency_kernel_launches += 1;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  rc = run_stage(kLayerProjectionKinds[3], &stages[3]);
  if (rc != 0 || stages[3].status != 0 || stages[3].exact == 0) {
    out->exact = 0;
    out->status = stages[3].status;
    return rc;
  }

  for (NervaCudaHfDecodeSequenceSession *session : selected) {
    const uint32_t attention_hidden = session->heads * session->head_dim;
    const uint32_t kv_hidden = session->kv_heads * session->head_dim;
    if (request->layer_index + 1 < session->layer_count) {
      const SequenceLayerLayout next_layout =
          session->host_layouts[request->layer_index + 1];
      const uint64_t output_offset =
          (request->layer_index % 2 == 0) ? session->arena_layout.scratch
                                          : session->arena_layout.input;
      hf_layer_finish_next_attn_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, best->stream>>>(
          session->device_arena, output_offset, next_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, session->max_context_tokens, session->rms_eps,
          session->device_scratch, session->device_projection_input);
    } else {
      hf_layer_finish_final_norm_encode_kernel<<<
          1, kDecodeNormThreads, 0, best->stream>>>(
          session->device_arena, session->arena_layout, session->dtype,
          session->hidden, attention_hidden, kv_hidden, session->intermediate,
          session->device_step, session->max_context_tokens, session->rms_eps,
          session->device_scratch, session->device_projection_input);
    }
    out->dependency_kernel_launches += 1;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  if (best->projection_batch_defer_layer_sync == 0) {
    err = cudaStreamSynchronize(best->stream);
    out->sync_calls += 1;
    if (err != cudaSuccess) {
      out->cuda_error = static_cast<int32_t>(err);
      return -1;
    }
  }

  const auto &qkv = stages[0];
  const auto &attention_output = stages[1];
  const auto &gate_up = stages[2];
  const auto &down = stages[3];
  out->reason = kProjectionBatchPlanReady;
  out->exact = 1;
  out->status = 0;
  out->qkv_rows = qkv.rows;
  out->attention_output_rows = attention_output.rows;
  out->gate_up_rows = gate_up.rows;
  out->down_rows = down.rows;
  out->hidden_cols = qkv.cols;
  out->attention_output_cols = attention_output.cols;
  out->down_cols = down.cols;
  out->input_bytes = qkv.input_bytes + attention_output.input_bytes +
                     gate_up.input_bytes + down.input_bytes;
  out->output_bytes = qkv.output_bytes + attention_output.output_bytes +
                      gate_up.output_bytes + down.output_bytes;
  out->qkv_elapsed_ns = qkv.elapsed_ns;
  out->attention_output_elapsed_ns = attention_output.elapsed_ns;
  out->gate_up_elapsed_ns = gate_up.elapsed_ns;
  out->down_elapsed_ns = down.elapsed_ns;
  out->elapsed_ns = qkv.elapsed_ns + attention_output.elapsed_ns +
                    gate_up.elapsed_ns + down.elapsed_ns;
  out->pack_kernel_launches = qkv.pack_kernel_launches +
                              attention_output.pack_kernel_launches +
                              gate_up.pack_kernel_launches +
                              down.pack_kernel_launches;
  out->projection_kernel_launches =
      qkv.projection_kernel_launches +
      attention_output.projection_kernel_launches +
      gate_up.projection_kernel_launches + down.projection_kernel_launches;
  out->scatter_kernel_launches = qkv.scatter_kernel_launches +
                                 attention_output.scatter_kernel_launches +
                                 gate_up.scatter_kernel_launches +
                                 down.scatter_kernel_launches;
  out->sync_calls += qkv.sync_calls + attention_output.sync_calls +
                     gate_up.sync_calls + down.sync_calls;
  out->hot_path_allocations = qkv.hot_path_allocations +
                              attention_output.hot_path_allocations +
                              gate_up.hot_path_allocations +
                              down.hot_path_allocations;
  return 0;
}

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
  session->experimental_rt_decode_enabled = parent->experimental_rt_decode_enabled;
  session->experimental_rt_page_tokens = parent->experimental_rt_page_tokens;
  session->experimental_rt_pages = parent->experimental_rt_pages;
  session->experimental_rt_local_window_tokens =
      parent->experimental_rt_local_window_tokens;
  session->experimental_rt_sink_tokens = parent->experimental_rt_sink_tokens;
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
    const uint32_t blocks =
        (session->kv_block_count + kDecodeThreads - 1u) / kDecodeThreads;
    hf_init_identity_kv_block_table_kernel<<<blocks, kDecodeThreads, 0,
                                             session->stream>>>(
        session->device_kv_block_table, session->kv_block_count);
    err = cudaGetLastError();
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
