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

extern "C" int nerva_cuda_rt_candidate_selector_create(
    uint32_t pages, uint32_t page_tokens, uint32_t query_count,
    uint32_t candidates_per_query, uint32_t *candidate_pages, void *stream,
    void **selector_out, int32_t *cuda_error_out);
extern "C" int nerva_cuda_rt_candidate_selector_create_with_queries(
    uint32_t pages, uint32_t page_tokens, uint32_t query_count,
    uint32_t candidates_per_query, uint32_t *candidate_pages,
    const float *queries, uint32_t query_dims, const uint32_t *step_cursor,
    void *stream, void **selector_out, int32_t *cuda_error_out);
extern "C" int
nerva_cuda_rt_candidate_selector_create_with_query_page_descriptors(
    uint32_t pages, uint32_t page_tokens, uint32_t layer_count,
    uint32_t query_count, uint32_t candidates_per_query,
    uint32_t *candidate_pages, const float *queries, uint32_t query_dims,
    const float *page_descriptors, uint32_t page_descriptor_dims,
    const uint32_t *step_cursor, void *stream, void **selector_out,
    int32_t *cuda_error_out);
extern "C" int nerva_cuda_rt_candidate_selector_launch(
    void *selector, void *stream, uint32_t active_pages, uint32_t current_page,
    uint32_t local_pages, uint32_t sink_pages, uint32_t layer_index,
    int32_t *cuda_error_out);
extern "C" void nerva_cuda_rt_candidate_selector_destroy(void *selector);

#if NERVA_HAVE_CUDNN_FRONTEND
#include <cudnn.h>
#include <cudnn_frontend.h>
#endif

namespace {
#include "hf_decode_sequence/session_prelude.inc.cu"

uint32_t checked_u32_product(uint32_t lhs, uint32_t rhs) {
  if (lhs != 0 && rhs > UINT32_MAX / lhs) {
    return 0;
  }
  return lhs * rhs;
}

#include "hf_decode_sequence/deepseek/layout_plan.inc.cu"
}  // namespace

#include "hf_decode_sequence/session_state.cuh"

extern "C" int nerva_cuda_hf_decode_sequence_plan_layout(
    const NervaCudaHfDecodeSequenceLayoutPlanRequest *request,
    NervaCudaHfDecodeSequenceLayoutPlanResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->layers == nullptr ||
      request->layer_count == 0 || request->layer_index >= request->layer_count ||
      request->hidden == 0 || request->heads == 0 || request->kv_heads == 0 ||
      request->head_dim == 0 || request->intermediate == 0 ||
      request->vocab_size == 0 || request->kv_heads > request->heads ||
      request->heads % request->kv_heads != 0) {
    return -1;
  }
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    if (!valid_layer(request->layers[index], false)) {
      return -1;
    }
  }

  const uint64_t hidden = request->hidden;
  const uint64_t attention_hidden =
      static_cast<uint64_t>(request->heads) * request->head_dim;
  const uint64_t kv_hidden =
      static_cast<uint64_t>(request->kv_heads) * request->head_dim;
  const uint64_t intermediate = request->intermediate;
  const uint64_t vocab_size = request->vocab_size;
  SequenceArenaLayout arena_layout{};
  std::vector<SequenceLayerLayout> layouts(request->layer_count);
  uint64_t elements = 0;
  arena_layout.embeddings = push(elements, vocab_size * hidden);
  arena_layout.input = push(elements, hidden);
  arena_layout.scratch = push(elements, hidden);
  arena_layout.deepseek_hc_head_base = kMissingOffset;
  arena_layout.deepseek_hc_head_fn = kMissingOffset;
  arena_layout.deepseek_hc_head_scale = kMissingOffset;
  pack_deepseek_static(arena_layout, elements, request->layers,
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
  const uint32_t final_norm_weight_dtype =
      final_norm_weight_dtype_for_layers(request->layers, request->layer_count,
                                         kDTypeBF16);
  arena_layout.final_norm = push(elements,
                                 dtype_slots(hidden, final_norm_weight_dtype));
  arena_layout.lm_head = push(elements, vocab_size * hidden);

  const SequenceLayerLayout &layout = layouts[request->layer_index];
  out->status = 0;
  out->hidden = request->hidden;
  out->heads = request->heads;
  out->kv_heads = request->kv_heads;
  out->head_dim = request->head_dim;
  out->intermediate = request->intermediate;
  out->vocab_size = request->vocab_size;
  out->layer_count = request->layer_count;
  out->layer_index = request->layer_index;
  out->attention_kind = layout.attention_kind;
  fill_deepseek_layout_plan_result(
      request->layers[request->layer_index], arena_layout, layout, out);
  out->resident_weight_bytes = elements * sizeof(uint16_t) -
                               hidden * 2u * sizeof(uint16_t);
  out->layout_bytes = layouts.size() * sizeof(SequenceLayerLayout);
  out->final_norm = arena_layout.final_norm;
  out->lm_head = arena_layout.lm_head;
  out->rms_attn = layout.rms_attn;
  out->rms_mlp = layout.rms_mlp;
  out->w_q = layout.w_q;
  out->q_norm = layout.q_norm;
  out->w_k = layout.w_k;
  out->k_norm = layout.k_norm;
  out->w_v = layout.w_v;
  out->w_o = layout.w_o;
  out->w_router = layout.w_router;
  out->w_expert_gate_up = layout.w_expert_gate_up;
  out->w_expert_down = layout.w_expert_down;
  out->w_shared_expert_gate = layout.w_shared_expert_gate;
  out->w_shared_expert_up = layout.w_shared_expert_up;
  out->w_shared_expert_down = layout.w_shared_expert_down;
  return 0;
}

namespace {
#include "hf_decode_sequence/session_common.inc.cu"
#include "hf_decode_sequence/session_cudnn.inc.cu"
#include "hf_decode_sequence/session_profile.inc.cu"
#include "hf_decode_sequence/session_decode_prefill.inc.cu"
#include "hf_decode_sequence/session_graph_result.inc.cu"
#include "hf_decode_sequence/deepseek/session_api_state.inc.cu"
}  // namespace

#include "hf_decode_sequence/session_api_oneshot.inc.cu"
#include "hf_decode_sequence/session_api_lifecycle.inc.cu"
#include "hf_decode_sequence/session_api_projection_batch.inc.cu"
#include "hf_decode_sequence/session_api_layer_projection_batch.inc.cu"
#include "hf_decode_sequence/session_api_batch_fork_destroy.inc.cu"
