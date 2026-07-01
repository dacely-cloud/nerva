__device__ __forceinline__ uint32_t *deepseek_moe_route_ids(
    const LayerScratch &s) {
  return reinterpret_cast<uint32_t *>(s.gate);
}

__device__ __forceinline__ float *deepseek_moe_route_weights(
    const LayerScratch &s) {
  return s.up;
}

__device__ __forceinline__ uint64_t deepseek_moe_extra_scratch_base(
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate) {
  return static_cast<uint64_t>(hidden) * 5u +
         static_cast<uint64_t>(attention_hidden) * 3u +
         static_cast<uint64_t>(kv_hidden) * 2u +
         static_cast<uint64_t>(intermediate) * 3u;
}

__device__ __forceinline__ float *deepseek_moe_rank_ff(
    float *scratch, const LayerScratch &s, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t rank) {
  if (rank == 0) {
    return s.ff;
  }
  return scratch +
         deepseek_moe_extra_scratch_base(hidden, attention_hidden, kv_hidden,
                                         intermediate) +
         static_cast<uint64_t>(rank - 1u) * intermediate;
}

__device__ __forceinline__ float *deepseek_moe_rank_down(
    float *scratch, const LayerScratch &s, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t top_k, uint32_t rank) {
  if (rank == 0) {
    return s.down;
  }
  const uint64_t base =
      deepseek_moe_extra_scratch_base(hidden, attention_hidden, kv_hidden,
                                      intermediate);
  const uint64_t extra_ff =
      static_cast<uint64_t>(top_k - 1u) * intermediate;
  return scratch + base + extra_ff + static_cast<uint64_t>(rank - 1u) * hidden;
}

__global__ void hf_deepseek_v3_sparse_moe_router_logits_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch, const uint16_t *projection_input) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  const uint32_t expert = blockIdx.x;
  const uint32_t num_experts = layout.num_experts;
  if (layout.w_router == kMissingOffset || projection_input == nullptr ||
      num_experts == 0 || num_experts > kSparseMoeExpertsMax ||
      expert >= num_experts || num_experts > intermediate) {
    return;
  }

  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint64_t row = layout.w_router + static_cast<uint64_t>(expert) * hidden;
  float sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    sum += encoded_to_f32(arena[row + col], kDTypeBF16) *
           encoded_to_f32(projection_input[col], dtype);
  }
  sum = block_sum(sum);
  if (threadIdx.x == 0) {
    s.ff[expert] = sum;
  }
}

__global__ void hf_deepseek_v3_sparse_moe_route_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch, uint16_t *projection_input,
    uint64_t *deepseek_runtime_counters) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  __shared__ float correction_bias[kSparseMoeExpertsMax];
  __shared__ uint32_t selected_experts[kSparseMoeTopKMax];
  __shared__ float selected_weights[kSparseMoeTopKMax];
  __shared__ int route_status;

  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  uint32_t *route_ids = deepseek_moe_route_ids(s);
  float *route_weights = deepseek_moe_route_weights(s);
  if (threadIdx.x == 0) {
    route_ids[0] = 1u;
  }
  encoded_slice_to_f32(projection_input, hidden, dtype, s.mlp_norm);
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    s.down[index] = 0.0f;
  }
  __syncthreads();

  const uint32_t num_experts = layout.num_experts;
  const uint32_t top_k = layout.experts_per_token;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  if (layout.w_router == kMissingOffset ||
      layout.w_expert_gate_up == kMissingOffset ||
      layout.w_expert_down == kMissingOffset ||
      num_experts == 0 || num_experts > kSparseMoeExpertsMax ||
      top_k == 0 || top_k > kSparseMoeTopKMax || top_k > num_experts ||
      moe_intermediate == 0 || moe_intermediate > intermediate ||
      top_k + 1u > intermediate || num_experts > intermediate) {
    return;
  }

  const bool has_router_bias =
      (layout.deepseek_flags & kDeepSeekFlagRouterBias) != 0;
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  const uint64_t router_bias_offset =
      layout.w_router + static_cast<uint64_t>(num_experts) * hidden;
  for (uint32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    correction_bias[expert] =
        !has_router_bias
            ? 0.0f
            : (bf16_storage
                   ? encoded_to_f32(arena[router_bias_offset + expert],
                                    kDTypeBF16)
                   : f32_from_u16_slots(arena + router_bias_offset, expert));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const uint32_t router_groups =
        layout.deepseek_router_num_groups == 0 ? 1u
                                               : layout.deepseek_router_num_groups;
    const uint32_t router_topk_groups =
        layout.deepseek_router_topk_groups == 0
            ? 1u
            : layout.deepseek_router_topk_groups;
    const float routed_scale =
        isfinite(layout.deepseek_routed_scaling_factor) &&
                layout.deepseek_routed_scaling_factor != 0.0f
            ? layout.deepseek_routed_scaling_factor
            : 1.0f;
    route_status = nerva::deepseek::router::route_v3_grouped_sigmoid(
        s.ff, has_router_bias ? correction_bias : nullptr,
        num_experts, router_groups, router_topk_groups, top_k,
        layout.norm_topk_prob, routed_scale, selected_experts,
        selected_weights);
    route_ids[0] = route_status == 0 ? 0u : 1u;
    for (uint32_t rank = 0; rank < top_k; ++rank) {
      route_ids[rank + 1u] = selected_experts[rank];
      route_weights[rank] = selected_weights[rank];
    }
  }
  __syncthreads();
  if (route_status != 0) {
    return;
  }
  if (threadIdx.x == 0 && deepseek_runtime_counters != nullptr) {
    atomicAdd(
        reinterpret_cast<unsigned long long *>(
            deepseek_runtime_counters +
            kDeepSeekRuntimeCounterV3GroupedRouterSelections),
        1ull);
  }
}

__global__ void hf_deepseek_v4_sparse_moe_router_logits_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, float *scratch) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  const uint32_t expert = blockIdx.x;
  const uint32_t num_experts = layout.num_experts;
  if (layout.w_router == kMissingOffset || num_experts == 0 ||
      num_experts > kSparseMoeExpertsMax || expert >= num_experts ||
      num_experts > intermediate) {
    return;
  }

  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint64_t row = layout.w_router + static_cast<uint64_t>(expert) * hidden;
  float sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    sum += encoded_to_f32(arena[row + col], kDTypeBF16) * s.mlp_norm[col];
  }
  sum = block_sum(sum);
  if (threadIdx.x == 0) {
    s.ff[expert] = sum;
  }
}

__global__ void hf_deepseek_v4_sparse_moe_route_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t *step_cursor, uint32_t max_steps, uint32_t vocab_size,
    const uint32_t *prompt_tokens, uint32_t prompt_token_count,
    const NervaCudaSyntheticTokenSlot *slots, float *scratch,
    uint64_t *deepseek_runtime_counters) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  __shared__ float correction_bias[kSparseMoeExpertsMax];
  __shared__ uint32_t selected_experts[kSparseMoeTopKMax];
  __shared__ float selected_weights[kSparseMoeTopKMax];
  __shared__ int route_status;

  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  uint32_t *route_ids = deepseek_moe_route_ids(s);
  float *route_weights = deepseek_moe_route_weights(s);
  if (threadIdx.x == 0) {
    route_ids[0] = 1u;
    route_status = -1;
  }
  for (uint32_t index = threadIdx.x; index < hidden; index += blockDim.x) {
    s.down[index] = 0.0f;
  }
  __syncthreads();

  const uint32_t num_experts = layout.num_experts;
  const uint32_t top_k = layout.experts_per_token;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  if (layout.w_router == kMissingOffset ||
      layout.w_expert_gate_up == kMissingOffset ||
      layout.w_expert_down == kMissingOffset ||
      num_experts == 0 || num_experts > kSparseMoeExpertsMax ||
      top_k == 0 || top_k > kSparseMoeTopKMax || top_k > num_experts ||
      moe_intermediate == 0 || moe_intermediate > intermediate ||
      num_experts > intermediate || (hidden & 1u) != 0 ||
      (moe_intermediate & 1u) != 0) {
    return;
  }

  const uint64_t router_metadata_offset =
      layout.w_router + static_cast<uint64_t>(num_experts) * hidden;
  const bool hash_router =
      (layout.deepseek_flags & kDeepSeekFlagHashRouter) != 0;
  for (uint32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    correction_bias[expert] =
        hash_router ? 0.0f
                    : f32_from_u16_slots(arena + router_metadata_offset,
                                         expert);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const float routed_scale =
        isfinite(layout.deepseek_routed_scaling_factor) &&
                layout.deepseek_routed_scaling_factor != 0.0f
            ? layout.deepseek_routed_scaling_factor
            : 1.0f;
    if (hash_router) {
      const uint32_t position = step_cursor == nullptr ? 0 : *step_cursor;
      uint32_t current_token = UINT32_MAX;
      if (position < prompt_token_count) {
        current_token = prompt_tokens[position];
      } else if (position != 0 && slots != nullptr) {
        current_token = slots[position - 1u].token;
      }
      if (current_token < vocab_size) {
        float weight_sum = 0.0f;
        route_status = 0;
        for (uint32_t rank = 0; rank < top_k; ++rank) {
          const uint64_t table_index =
              static_cast<uint64_t>(current_token) * top_k + rank;
          const uint64_t expert64 =
              deepseek_u64_from_u16_slots(arena + router_metadata_offset,
                                          table_index);
          if (expert64 >= num_experts) {
            route_status = -2;
            break;
          }
          const uint32_t expert = static_cast<uint32_t>(expert64);
          selected_experts[rank] = expert;
          selected_weights[rank] =
              nerva::deepseek::router::sqrtsoftplus_score(s.ff[expert]);
          weight_sum += selected_weights[rank];
        }
        if (route_status == 0) {
          const float scale = nerva::deepseek::router::route_scale(
              weight_sum, layout.norm_topk_prob, routed_scale);
          for (uint32_t rank = 0; rank < top_k; ++rank) {
            selected_weights[rank] *= scale;
          }
        }
      } else {
        route_status = -3;
      }
    } else {
      route_status = nerva::deepseek::router::route_v4_sqrtsoftplus(
          s.ff, correction_bias, num_experts, top_k,
          layout.norm_topk_prob, routed_scale, selected_experts,
          selected_weights);
    }

    if (route_status == 0) {
      route_ids[0] = 0u;
      for (uint32_t rank = 0; rank < top_k; ++rank) {
        route_ids[rank + 1u] = selected_experts[rank];
        route_weights[rank] = selected_weights[rank];
      }
      if (deepseek_runtime_counters != nullptr) {
        const uint32_t counter =
            hash_router ? kDeepSeekRuntimeCounterV4HashRouterSelections
                        : kDeepSeekRuntimeCounterV4BiasRouterSelections;
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                deepseek_runtime_counters + counter),
            1ull);
      }
    }
  }
}

__global__ void hf_deepseek_v3_sparse_moe_expert_gate_up_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t rank, uint32_t *step_cursor,
    uint32_t max_steps, float *scratch) {
  (void)dtype;
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint32_t *route_ids = deepseek_moe_route_ids(s);
  if (route_ids[0] != 0u || rank >= layout.experts_per_token) {
    return;
  }
  const uint32_t row = blockIdx.x;
  const uint32_t num_experts = layout.num_experts;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  float *rank_ff = deepseek_moe_rank_ff(
      scratch, s, hidden, attention_hidden, kv_hidden, intermediate, rank);
  if (row >= moe_intermediate || num_experts == 0 ||
      moe_intermediate == 0 || moe_intermediate > intermediate) {
    return;
  }
  const uint32_t expert = route_ids[rank + 1u];
  const uint64_t expert_gate = layout.w_expert_gate_up;
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  const uint64_t expert_gate_data_slots =
      bf16_storage
          ? static_cast<uint64_t>(num_experts) * moe_intermediate * hidden
          : deepseek_device_fp8_slots(
                static_cast<uint64_t>(num_experts) * moe_intermediate,
                hidden);
  const uint64_t expert_gate_scale =
      bf16_storage ? kMissingOffset : expert_gate + expert_gate_data_slots;
  const uint64_t expert_up =
      bf16_storage
          ? expert_gate + expert_gate_data_slots
          : expert_gate_scale +
                static_cast<uint64_t>(num_experts) *
                    deepseek_device_scale_f32_slots(moe_intermediate, hidden);
  const uint64_t expert_up_scale =
      bf16_storage
          ? kMissingOffset
          : expert_up +
                deepseek_device_fp8_slots(
                    static_cast<uint64_t>(num_experts) * moe_intermediate,
                    hidden);
  float gate_sum = 0.0f;
  float up_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float gate_weight =
        bf16_storage
            ? deepseek_bf16_rank3_weight(arena, expert_gate, moe_intermediate,
                                         hidden, expert, row, col)
            : deepseek_fp8_rank3_scaled_weight(
                  arena, expert_gate, expert_gate_scale, moe_intermediate,
                  hidden, expert, row, col);
    const float up_weight =
        bf16_storage
            ? deepseek_bf16_rank3_weight(arena, expert_up, moe_intermediate,
                                         hidden, expert, row, col)
            : deepseek_fp8_rank3_scaled_weight(
                  arena, expert_up, expert_up_scale, moe_intermediate, hidden,
                  expert, row, col);
    gate_sum += gate_weight * s.mlp_norm[col];
    up_sum += up_weight * s.mlp_norm[col];
  }
  gate_sum = block_sum(gate_sum);
  up_sum = block_sum(up_sum);
  if (threadIdx.x == 0) {
    rank_ff[row] =
        deepseek_swiglu(gate_sum, up_sum, layout.deepseek_swiglu_limit);
  }
}

__global__ void hf_deepseek_v4_sparse_moe_expert_gate_up_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t rank, uint32_t *step_cursor, uint32_t max_steps, float *scratch) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint32_t *route_ids = deepseek_moe_route_ids(s);
  if (route_ids[0] != 0u || rank >= layout.experts_per_token) {
    return;
  }
  const uint32_t row = blockIdx.x;
  const uint32_t num_experts = layout.num_experts;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  float *rank_ff = deepseek_moe_rank_ff(
      scratch, s, hidden, attention_hidden, kv_hidden, intermediate, rank);
  if (row >= moe_intermediate || num_experts == 0 ||
      moe_intermediate == 0 || moe_intermediate > intermediate ||
      (hidden & 1u) != 0) {
    return;
  }
  const uint32_t expert = route_ids[rank + 1u];
  const uint32_t half_hidden = hidden >> 1u;
  const uint64_t expert_gate = layout.w_expert_gate_up;
  const uint64_t expert_gate_scale =
      expert_gate + deepseek_device_rank3_slots(num_experts, moe_intermediate,
                                                half_hidden);
  const uint32_t gate_scale_cols = (half_hidden + 15u) / 16u;
  const uint64_t expert_up =
      expert_gate_scale + deepseek_device_rank3_slots(
                              num_experts, moe_intermediate, gate_scale_cols);
  const uint64_t expert_up_scale =
      expert_up + deepseek_device_rank3_slots(num_experts, moe_intermediate,
                                              half_hidden);
  float gate_sum = 0.0f;
  float up_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    gate_sum += deepseek_mxfp4_rank3_scaled_weight(
                    arena, expert_gate, expert_gate_scale,
                    moe_intermediate, half_hidden, expert, row, col) *
                s.mlp_norm[col];
    up_sum += deepseek_mxfp4_rank3_scaled_weight(
                  arena, expert_up, expert_up_scale, moe_intermediate,
                  half_hidden, expert, row, col) *
              s.mlp_norm[col];
  }
  gate_sum = block_sum(gate_sum);
  up_sum = block_sum(up_sum);
  if (threadIdx.x == 0) {
    rank_ff[row] =
        deepseek_swiglu(gate_sum, up_sum, layout.deepseek_swiglu_limit);
  }
}

__global__ void hf_deepseek_v4_sparse_moe_expert_down_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t hidden,
    uint32_t attention_hidden, uint32_t kv_hidden, uint32_t intermediate,
    uint32_t rank, uint32_t *step_cursor, uint32_t max_steps, float *scratch) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint32_t *route_ids = deepseek_moe_route_ids(s);
  const float *route_weights = deepseek_moe_route_weights(s);
  if (route_ids[0] != 0u || rank >= layout.experts_per_token) {
    return;
  }
  const uint32_t row = blockIdx.x;
  const uint32_t num_experts = layout.num_experts;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  const uint32_t top_k = layout.experts_per_token;
  if (top_k == 0) {
    return;
  }
  const float *rank_ff = deepseek_moe_rank_ff(
      scratch, s, hidden, attention_hidden, kv_hidden, intermediate, rank);
  float *rank_down = deepseek_moe_rank_down(
      scratch, s, hidden, attention_hidden, kv_hidden, intermediate, top_k,
      rank);
  if (row >= hidden || num_experts == 0 || moe_intermediate == 0 ||
      moe_intermediate > intermediate || (moe_intermediate & 1u) != 0) {
    return;
  }
  const uint32_t expert = route_ids[rank + 1u];
  const float expert_weight = route_weights[rank];
  const uint32_t half_intermediate = moe_intermediate >> 1u;
  const uint64_t expert_down = layout.w_expert_down;
  const uint64_t expert_down_scale =
      expert_down + deepseek_device_rank3_slots(num_experts, hidden,
                                                half_intermediate);
  float down_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < moe_intermediate; col += blockDim.x) {
    down_sum += deepseek_mxfp4_rank3_scaled_weight(
                    arena, expert_down, expert_down_scale, hidden,
                    half_intermediate, expert, row, col) *
                rank_ff[col];
  }
  down_sum = block_sum(down_sum);
  if (threadIdx.x == 0) {
    rank_down[row] = expert_weight * down_sum;
  }
}

__global__ void hf_deepseek_sparse_moe_reduce_down_kernel(
    SequenceLayerLayout layout, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate, uint32_t *step_cursor,
    uint32_t max_steps, float *scratch) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint32_t *route_ids = deepseek_moe_route_ids(s);
  if (route_ids[0] != 0u) {
    return;
  }
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t top_k = layout.experts_per_token;
  if (row >= hidden || top_k <= 1u) {
    return;
  }
  float sum = 0.0f;
  for (uint32_t rank = 0; rank < top_k; ++rank) {
    const float *rank_down = deepseek_moe_rank_down(
        scratch, s, hidden, attention_hidden, kv_hidden, intermediate, top_k,
        rank);
    sum += rank_down[row];
  }
  s.down[row] = sum;
}

constexpr uint32_t kDeepSeekPrefillSparseMoeRowTile = 8;
constexpr uint32_t kDeepSeekPrefillSparseMoeTokenTile = 2;
constexpr uint32_t kDeepSeekPrefillSparseMoeSlots =
    kDeepSeekPrefillSparseMoeRowTile * kDeepSeekPrefillSparseMoeTokenTile;

__device__ __forceinline__ uint32_t *deepseek_prefill_moe_route_ids(
    uint16_t *route_scratch) {
  return reinterpret_cast<uint32_t *>(route_scratch);
}

__device__ __forceinline__ float *deepseek_prefill_moe_route_weights(
    uint16_t *route_scratch, uint32_t chunk_tokens, uint32_t top_k) {
  return reinterpret_cast<float *>(
      deepseek_prefill_moe_route_ids(route_scratch) +
      static_cast<uint64_t>(chunk_tokens) * top_k);
}

__device__ __forceinline__ float *deepseek_prefill_moe_rank_ff(
    float *gate_up_tmp, uint32_t token, uint32_t rank_ff_stride,
    uint32_t moe_intermediate, uint32_t rank) {
  return gate_up_tmp +
         static_cast<uint64_t>(token) * rank_ff_stride +
         static_cast<uint64_t>(rank) * moe_intermediate;
}

__device__ __forceinline__ float *deepseek_prefill_moe_shared_ff(
    float *gate_up_tmp, uint32_t token, uint32_t rank_ff_stride, uint32_t top_k,
    uint32_t moe_intermediate) {
  return gate_up_tmp + static_cast<uint64_t>(token) * rank_ff_stride +
         static_cast<uint64_t>(top_k) * moe_intermediate;
}

template <uint32_t Slots>
__device__ __forceinline__ void deepseek_prefill_moe_reduce_slots(
    float (&values)[Slots], float *partial) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5u;
  const uint32_t warp_count = (blockDim.x + 31u) >> 5u;

#pragma unroll
  for (uint32_t slot = 0; slot < Slots; ++slot) {
    float value = values[slot];
#pragma unroll
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
      value += __shfl_down_sync(0xffffffffu, value, static_cast<int>(offset));
    }
    if (lane == 0u) {
      partial[slot * blockDim.x + warp] = value;
    }
  }
  __syncthreads();

  if (warp == 0u) {
#pragma unroll
    for (uint32_t slot = 0; slot < Slots; ++slot) {
      float value =
          lane < warp_count ? partial[slot * blockDim.x + lane] : 0.0f;
#pragma unroll
      for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        value += __shfl_down_sync(0xffffffffu, value, static_cast<int>(offset));
      }
      if (lane == 0u) {
        partial[slot * blockDim.x] = value;
      }
    }
  }
  __syncthreads();
}

__global__ void hf_deepseek_prefill_sparse_moe_route_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t intermediate, uint32_t chunk_tokens,
    const uint16_t *norm_in, uint16_t *route_scratch,
    uint64_t *deepseek_runtime_counters) {
  const uint32_t token = blockIdx.x;
  if (token >= chunk_tokens || arena == nullptr || norm_in == nullptr ||
      route_scratch == nullptr) {
    return;
  }

  __shared__ float router_logits[kSparseMoeExpertsMax];
  __shared__ float correction_bias[kSparseMoeExpertsMax];
  __shared__ uint32_t selected_experts[kSparseMoeTopKMax];
  __shared__ float selected_weights[kSparseMoeTopKMax];
  __shared__ int route_status;

  const uint32_t num_experts = layout.num_experts;
  const uint32_t top_k = layout.experts_per_token;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  uint32_t *route_ids = deepseek_prefill_moe_route_ids(route_scratch);
  float *route_weights =
      deepseek_prefill_moe_route_weights(route_scratch, chunk_tokens, top_k);
  uint32_t *token_route_ids = route_ids + static_cast<uint64_t>(token) * top_k;
  float *token_route_weights =
      route_weights + static_cast<uint64_t>(token) * top_k;

  if (layout.w_router == kMissingOffset || num_experts == 0 ||
      num_experts > kSparseMoeExpertsMax || top_k == 0 ||
      top_k > kSparseMoeTopKMax || top_k > num_experts ||
      moe_intermediate == 0 || moe_intermediate > intermediate) {
    if (threadIdx.x == 0 && top_k != 0) {
      for (uint32_t rank = 0; rank < top_k; ++rank) {
        token_route_ids[rank] = 0xffffffffu;
        token_route_weights[rank] = 0.0f;
      }
    }
    return;
  }

  const uint16_t *token_norm =
      norm_in + static_cast<uint64_t>(token) * hidden;
  for (uint32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    const uint64_t router_row =
        layout.w_router + static_cast<uint64_t>(expert) * hidden;
    float sum = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      sum += encoded_to_f32(arena[router_row + col], kDTypeBF16) *
             encoded_to_f32(token_norm[col], dtype);
    }
    router_logits[expert] = sum;
  }

  const bool has_router_bias =
      (layout.deepseek_flags & kDeepSeekFlagRouterBias) != 0;
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  const uint64_t router_bias_offset =
      layout.w_router + static_cast<uint64_t>(num_experts) * hidden;
  for (uint32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    correction_bias[expert] =
        !has_router_bias
            ? 0.0f
            : (bf16_storage
                   ? encoded_to_f32(arena[router_bias_offset + expert],
                                    kDTypeBF16)
                   : f32_from_u16_slots(arena + router_bias_offset, expert));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const uint32_t router_groups =
        layout.deepseek_router_num_groups == 0 ? 1u
                                               : layout.deepseek_router_num_groups;
    const uint32_t router_topk_groups =
        layout.deepseek_router_topk_groups == 0
            ? 1u
            : layout.deepseek_router_topk_groups;
    const float routed_scale =
        isfinite(layout.deepseek_routed_scaling_factor) &&
                layout.deepseek_routed_scaling_factor != 0.0f
            ? layout.deepseek_routed_scaling_factor
            : 1.0f;
    route_status = nerva::deepseek::router::route_v3_grouped_sigmoid(
        router_logits, has_router_bias ? correction_bias : nullptr,
        num_experts, router_groups, router_topk_groups, top_k,
        layout.norm_topk_prob, routed_scale, selected_experts,
        selected_weights);
    if (route_status == 0 && deepseek_runtime_counters != nullptr) {
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterV3GroupedRouterSelections),
          1ull);
    }
    for (uint32_t rank = 0; rank < top_k; ++rank) {
      token_route_ids[rank] =
          route_status == 0 ? selected_experts[rank] : 0xffffffffu;
      token_route_weights[rank] =
          route_status == 0 ? selected_weights[rank] : 0.0f;
    }
  }
}

__global__ void hf_deepseek_prefill_sparse_moe_gate_up_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t intermediate, uint32_t chunk_tokens,
    const uint16_t *norm_in, uint16_t *route_scratch, float *gate_up_tmp) {
  const uint32_t row_start = blockIdx.x * kDeepSeekPrefillSparseMoeRowTile;
  const uint32_t token_start = blockIdx.y * kDeepSeekPrefillSparseMoeTokenTile;
  const uint32_t rank = blockIdx.z;
  const uint32_t num_experts = layout.num_experts;
  const uint32_t top_k = layout.experts_per_token;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  const uint32_t rank_ff_stride =
      top_k * moe_intermediate + layout.shared_expert_intermediate;
  if (arena == nullptr || norm_in == nullptr || route_scratch == nullptr ||
      gate_up_tmp == nullptr || row_start >= moe_intermediate ||
      token_start >= chunk_tokens || rank >= top_k ||
      layout.w_expert_gate_up == kMissingOffset || num_experts == 0 ||
      moe_intermediate == 0 || moe_intermediate > intermediate) {
    return;
  }

  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  const uint64_t expert_gate = layout.w_expert_gate_up;
  const uint64_t expert_gate_data_slots =
      bf16_storage
          ? static_cast<uint64_t>(num_experts) * moe_intermediate * hidden
          : deepseek_device_fp8_slots(
                static_cast<uint64_t>(num_experts) * moe_intermediate,
                hidden);
  const uint64_t expert_gate_scale =
      bf16_storage ? kMissingOffset : expert_gate + expert_gate_data_slots;
  const uint64_t expert_up =
      bf16_storage
          ? expert_gate + expert_gate_data_slots
          : expert_gate_scale +
                static_cast<uint64_t>(num_experts) *
                    deepseek_device_scale_f32_slots(moe_intermediate, hidden);
  const uint64_t expert_up_scale =
      bf16_storage
          ? kMissingOffset
          : expert_up +
                deepseek_device_fp8_slots(
                    static_cast<uint64_t>(num_experts) * moe_intermediate,
                    hidden);

  const uint32_t *route_ids = deepseek_prefill_moe_route_ids(route_scratch);
  extern __shared__ float partial[];
  float gate_sum[kDeepSeekPrefillSparseMoeSlots] = {};
  float up_sum[kDeepSeekPrefillSparseMoeSlots] = {};

  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    float input_value[kDeepSeekPrefillSparseMoeTokenTile] = {};
#pragma unroll
    for (uint32_t token_tile = 0;
         token_tile < kDeepSeekPrefillSparseMoeTokenTile; ++token_tile) {
      const uint32_t token = token_start + token_tile;
      if (token < chunk_tokens) {
        const uint16_t *token_norm =
            norm_in + static_cast<uint64_t>(token) * hidden;
        input_value[token_tile] = encoded_to_f32(token_norm[col], dtype);
      }
    }
#pragma unroll
    for (uint32_t row_tile = 0; row_tile < kDeepSeekPrefillSparseMoeRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= moe_intermediate) continue;
#pragma unroll
      for (uint32_t token_tile = 0;
           token_tile < kDeepSeekPrefillSparseMoeTokenTile; ++token_tile) {
        const uint32_t token = token_start + token_tile;
        if (token >= chunk_tokens) continue;
        const uint32_t expert =
            route_ids[static_cast<uint64_t>(token) * top_k + rank];
        if (expert >= num_experts) continue;
        const float gate_weight =
            bf16_storage
                ? deepseek_bf16_rank3_weight(arena, expert_gate,
                                             moe_intermediate, hidden, expert,
                                             row, col)
                : deepseek_fp8_rank3_scaled_weight(
                      arena, expert_gate, expert_gate_scale, moe_intermediate,
                      hidden, expert, row, col);
        const float up_weight =
            bf16_storage
                ? deepseek_bf16_rank3_weight(arena, expert_up,
                                             moe_intermediate, hidden, expert,
                                             row, col)
                : deepseek_fp8_rank3_scaled_weight(
                      arena, expert_up, expert_up_scale, moe_intermediate,
                      hidden, expert, row, col);
        const uint32_t slot =
            row_tile * kDeepSeekPrefillSparseMoeTokenTile + token_tile;
        gate_sum[slot] += gate_weight * input_value[token_tile];
        up_sum[slot] += up_weight * input_value[token_tile];
      }
    }
  }
  deepseek_prefill_moe_reduce_slots(gate_sum, partial);
  float *up_partial = partial + kDeepSeekPrefillSparseMoeSlots * blockDim.x;
  deepseek_prefill_moe_reduce_slots(up_sum, up_partial);
  if (threadIdx.x == 0) {
#pragma unroll
    for (uint32_t row_tile = 0; row_tile < kDeepSeekPrefillSparseMoeRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= moe_intermediate) continue;
#pragma unroll
      for (uint32_t token_tile = 0;
           token_tile < kDeepSeekPrefillSparseMoeTokenTile; ++token_tile) {
        const uint32_t token = token_start + token_tile;
        if (token >= chunk_tokens) continue;
        const uint32_t slot =
            row_tile * kDeepSeekPrefillSparseMoeTokenTile + token_tile;
        float *rank_ff = deepseek_prefill_moe_rank_ff(
            gate_up_tmp, token, rank_ff_stride, moe_intermediate, rank);
        rank_ff[row] = deepseek_swiglu(partial[slot * blockDim.x],
                                       up_partial[slot * blockDim.x],
                                       layout.deepseek_swiglu_limit);
      }
    }
  }
}

__global__ void hf_deepseek_prefill_sparse_moe_shared_gate_up_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t intermediate, uint32_t chunk_tokens,
    const uint16_t *norm_in, float *gate_up_tmp) {
  const uint32_t row_start = blockIdx.x * kDeepSeekPrefillSparseMoeRowTile;
  const uint32_t token_start = blockIdx.y * kDeepSeekPrefillSparseMoeTokenTile;
  const uint32_t top_k = layout.experts_per_token;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  const uint32_t shared_intermediate = layout.shared_expert_intermediate;
  const uint32_t rank_ff_stride =
      top_k * moe_intermediate + shared_intermediate;
  if (arena == nullptr || norm_in == nullptr || gate_up_tmp == nullptr ||
      shared_intermediate == 0 || row_start >= shared_intermediate ||
      token_start >= chunk_tokens ||
      layout.w_shared_expert_gate == kMissingOffset ||
      layout.w_shared_expert_up == kMissingOffset ||
      layout.w_shared_expert_down == kMissingOffset ||
      shared_intermediate > intermediate) {
    return;
  }

  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  const uint64_t shared_gate_scale =
      bf16_storage ? kMissingOffset
                   : layout.w_shared_expert_gate +
                         deepseek_device_fp8_slots(shared_intermediate,
                                                   hidden);
  const uint64_t shared_up_scale =
      bf16_storage ? kMissingOffset
                   : layout.w_shared_expert_up +
                         deepseek_device_fp8_slots(shared_intermediate,
                                                   hidden);

  extern __shared__ float partial[];
  float gate_sum[kDeepSeekPrefillSparseMoeSlots] = {};
  float up_sum[kDeepSeekPrefillSparseMoeSlots] = {};
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    float input_value[kDeepSeekPrefillSparseMoeTokenTile] = {};
#pragma unroll
    for (uint32_t token_tile = 0;
         token_tile < kDeepSeekPrefillSparseMoeTokenTile; ++token_tile) {
      const uint32_t token = token_start + token_tile;
      if (token < chunk_tokens) {
        const uint16_t *token_norm =
            norm_in + static_cast<uint64_t>(token) * hidden;
        input_value[token_tile] = encoded_to_f32(token_norm[col], dtype);
      }
    }
#pragma unroll
    for (uint32_t row_tile = 0; row_tile < kDeepSeekPrefillSparseMoeRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= shared_intermediate) continue;
#pragma unroll
      for (uint32_t token_tile = 0;
           token_tile < kDeepSeekPrefillSparseMoeTokenTile; ++token_tile) {
        const uint32_t token = token_start + token_tile;
        if (token >= chunk_tokens) continue;
        const float gate_weight =
            bf16_storage
                ? deepseek_bf16_weight(arena, layout.w_shared_expert_gate,
                                       shared_intermediate, hidden, row, col)
                : deepseek_fp8_scaled_weight(
                      arena, layout.w_shared_expert_gate, shared_gate_scale,
                      shared_intermediate, hidden, row, col);
        const float up_weight =
            bf16_storage
                ? deepseek_bf16_weight(arena, layout.w_shared_expert_up,
                                       shared_intermediate, hidden, row, col)
                : deepseek_fp8_scaled_weight(
                      arena, layout.w_shared_expert_up, shared_up_scale,
                      shared_intermediate, hidden, row, col);
        const uint32_t slot =
            row_tile * kDeepSeekPrefillSparseMoeTokenTile + token_tile;
        gate_sum[slot] += gate_weight * input_value[token_tile];
        up_sum[slot] += up_weight * input_value[token_tile];
      }
    }
  }
  deepseek_prefill_moe_reduce_slots(gate_sum, partial);
  float *up_partial = partial + kDeepSeekPrefillSparseMoeSlots * blockDim.x;
  deepseek_prefill_moe_reduce_slots(up_sum, up_partial);
  if (threadIdx.x == 0) {
#pragma unroll
    for (uint32_t row_tile = 0; row_tile < kDeepSeekPrefillSparseMoeRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= shared_intermediate) continue;
#pragma unroll
      for (uint32_t token_tile = 0;
           token_tile < kDeepSeekPrefillSparseMoeTokenTile; ++token_tile) {
        const uint32_t token = token_start + token_tile;
        if (token >= chunk_tokens) continue;
        float *shared_ff = deepseek_prefill_moe_shared_ff(
            gate_up_tmp, token, rank_ff_stride, top_k, moe_intermediate);
        const uint32_t slot =
            row_tile * kDeepSeekPrefillSparseMoeTokenTile + token_tile;
        shared_ff[row] = deepseek_swiglu(partial[slot * blockDim.x],
                                         up_partial[slot * blockDim.x],
                                         layout.deepseek_swiglu_limit);
      }
    }
  }
}

__global__ void hf_deepseek_prefill_sparse_moe_down_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t hidden,
    uint32_t intermediate, uint32_t chunk_tokens, uint16_t *route_scratch,
    float *gate_up_tmp, float *down_out) {
  const uint32_t row_start = blockIdx.x * kDeepSeekPrefillSparseMoeRowTile;
  const uint32_t token_start = blockIdx.y * kDeepSeekPrefillSparseMoeTokenTile;
  const uint32_t num_experts = layout.num_experts;
  const uint32_t top_k = layout.experts_per_token;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  const uint32_t rank_ff_stride =
      top_k * moe_intermediate + layout.shared_expert_intermediate;
  if (arena == nullptr || route_scratch == nullptr || gate_up_tmp == nullptr ||
      down_out == nullptr || row_start >= hidden || token_start >= chunk_tokens ||
      layout.w_expert_down == kMissingOffset || top_k == 0 ||
      num_experts == 0 || moe_intermediate == 0 ||
      moe_intermediate > intermediate) {
    return;
  }

  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  const uint64_t expert_down = layout.w_expert_down;
  const uint64_t expert_down_scale =
      bf16_storage
          ? kMissingOffset
          : expert_down +
                deepseek_device_fp8_slots(
                    static_cast<uint64_t>(num_experts) * hidden,
                    moe_intermediate);
  const uint32_t *route_ids = deepseek_prefill_moe_route_ids(route_scratch);
  const float *route_weights =
      deepseek_prefill_moe_route_weights(route_scratch, chunk_tokens, top_k);

  extern __shared__ float partial[];
  float sum[kDeepSeekPrefillSparseMoeSlots] = {};
  for (uint32_t rank = 0; rank < top_k; ++rank) {
    for (uint32_t col = threadIdx.x; col < moe_intermediate;
         col += blockDim.x) {
#pragma unroll
      for (uint32_t row_tile = 0; row_tile < kDeepSeekPrefillSparseMoeRowTile;
           ++row_tile) {
        const uint32_t row = row_start + row_tile;
        if (row >= hidden) continue;
#pragma unroll
        for (uint32_t token_tile = 0;
             token_tile < kDeepSeekPrefillSparseMoeTokenTile; ++token_tile) {
          const uint32_t token = token_start + token_tile;
          if (token >= chunk_tokens) continue;
          const uint64_t route_offset =
              static_cast<uint64_t>(token) * top_k + rank;
          const uint32_t expert = route_ids[route_offset];
          if (expert >= num_experts) continue;
          const float *rank_ff =
              deepseek_prefill_moe_rank_ff(gate_up_tmp, token, rank_ff_stride,
                                           moe_intermediate, rank);
          const float weight =
              bf16_storage
                  ? deepseek_bf16_rank3_weight(arena, expert_down, hidden,
                                               moe_intermediate, expert, row,
                                               col)
                  : deepseek_fp8_rank3_scaled_weight(
                        arena, expert_down, expert_down_scale, hidden,
                        moe_intermediate, expert, row, col);
          const uint32_t slot =
              row_tile * kDeepSeekPrefillSparseMoeTokenTile + token_tile;
          sum[slot] += route_weights[route_offset] * weight * rank_ff[col];
        }
      }
    }
  }

  const uint32_t shared_intermediate = layout.shared_expert_intermediate;
  if (shared_intermediate != 0 &&
      layout.w_shared_expert_down != kMissingOffset &&
      shared_intermediate <= intermediate) {
    const uint64_t shared_down_scale =
        bf16_storage ? kMissingOffset
                     : layout.w_shared_expert_down +
                           deepseek_device_fp8_slots(hidden,
                                                     shared_intermediate);
    for (uint32_t col = threadIdx.x; col < shared_intermediate;
         col += blockDim.x) {
#pragma unroll
      for (uint32_t row_tile = 0; row_tile < kDeepSeekPrefillSparseMoeRowTile;
           ++row_tile) {
        const uint32_t row = row_start + row_tile;
        if (row >= hidden) continue;
#pragma unroll
        for (uint32_t token_tile = 0;
             token_tile < kDeepSeekPrefillSparseMoeTokenTile; ++token_tile) {
          const uint32_t token = token_start + token_tile;
          if (token >= chunk_tokens) continue;
          const float *shared_ff =
              deepseek_prefill_moe_shared_ff(gate_up_tmp, token, rank_ff_stride,
                                             top_k,
                                             moe_intermediate);
          const float weight =
              bf16_storage
                  ? deepseek_bf16_weight(arena, layout.w_shared_expert_down,
                                         hidden, shared_intermediate, row, col)
                  : deepseek_fp8_scaled_weight(
                        arena, layout.w_shared_expert_down, shared_down_scale,
                        hidden, shared_intermediate, row, col);
          const uint32_t slot =
              row_tile * kDeepSeekPrefillSparseMoeTokenTile + token_tile;
          sum[slot] += weight * shared_ff[col];
        }
      }
    }
  }

  deepseek_prefill_moe_reduce_slots(sum, partial);
  if (threadIdx.x == 0) {
#pragma unroll
    for (uint32_t row_tile = 0; row_tile < kDeepSeekPrefillSparseMoeRowTile;
         ++row_tile) {
      const uint32_t row = row_start + row_tile;
      if (row >= hidden) continue;
#pragma unroll
      for (uint32_t token_tile = 0;
           token_tile < kDeepSeekPrefillSparseMoeTokenTile; ++token_tile) {
        const uint32_t token = token_start + token_tile;
        if (token >= chunk_tokens) continue;
        const uint32_t slot =
            row_tile * kDeepSeekPrefillSparseMoeTokenTile + token_tile;
        down_out[static_cast<uint64_t>(token) * hidden + row] =
            partial[slot * blockDim.x];
      }
    }
  }
}

__global__ void hf_deepseek_prefill_sparse_moe_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t intermediate, uint32_t chunk_tokens,
    const uint16_t *norm_in, float *gate_up_tmp, float *down_out,
    uint64_t *deepseek_runtime_counters) {
  const uint32_t token = blockIdx.x;
  if (token >= chunk_tokens || arena == nullptr || norm_in == nullptr ||
      gate_up_tmp == nullptr || down_out == nullptr) {
    return;
  }

  __shared__ float router_logits[kSparseMoeExpertsMax];
  __shared__ float correction_bias[kSparseMoeExpertsMax];
  __shared__ uint32_t selected_experts[kSparseMoeTopKMax];
  __shared__ float selected_weights[kSparseMoeTopKMax];
  __shared__ int route_status;

  const uint32_t num_experts = layout.num_experts;
  const uint32_t top_k = layout.experts_per_token;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  if (layout.w_router == kMissingOffset ||
      layout.w_expert_gate_up == kMissingOffset ||
      layout.w_expert_down == kMissingOffset ||
      num_experts == 0 || num_experts > kSparseMoeExpertsMax ||
      top_k == 0 || top_k > kSparseMoeTopKMax || top_k > num_experts ||
      moe_intermediate == 0 || moe_intermediate > intermediate) {
    return;
  }

  const uint16_t *token_norm =
      norm_in + static_cast<uint64_t>(token) * hidden;
  float *token_gate =
      gate_up_tmp + static_cast<uint64_t>(token) * intermediate * 2u;
  float *token_up = token_gate + intermediate;
  float *token_down = down_out + static_cast<uint64_t>(token) * hidden;
  for (uint32_t row = threadIdx.x; row < hidden; row += blockDim.x) {
    token_down[row] = 0.0f;
  }

  for (uint32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    const uint64_t router_row =
        layout.w_router + static_cast<uint64_t>(expert) * hidden;
    float sum = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      sum += encoded_to_f32(arena[router_row + col], kDTypeBF16) *
             encoded_to_f32(token_norm[col], dtype);
    }
    router_logits[expert] = sum;
  }

  const bool has_router_bias =
      (layout.deepseek_flags & kDeepSeekFlagRouterBias) != 0;
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  const uint64_t router_bias_offset =
      layout.w_router + static_cast<uint64_t>(num_experts) * hidden;
  for (uint32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    correction_bias[expert] =
        !has_router_bias
            ? 0.0f
            : (bf16_storage
                   ? encoded_to_f32(arena[router_bias_offset + expert],
                                    kDTypeBF16)
                   : f32_from_u16_slots(arena + router_bias_offset, expert));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const uint32_t router_groups =
        layout.deepseek_router_num_groups == 0 ? 1u
                                               : layout.deepseek_router_num_groups;
    const uint32_t router_topk_groups =
        layout.deepseek_router_topk_groups == 0
            ? 1u
            : layout.deepseek_router_topk_groups;
    const float routed_scale =
        isfinite(layout.deepseek_routed_scaling_factor) &&
                layout.deepseek_routed_scaling_factor != 0.0f
            ? layout.deepseek_routed_scaling_factor
            : 1.0f;
    route_status = nerva::deepseek::router::route_v3_grouped_sigmoid(
        router_logits, has_router_bias ? correction_bias : nullptr,
        num_experts, router_groups, router_topk_groups, top_k,
        layout.norm_topk_prob, routed_scale, selected_experts,
        selected_weights);
    if (route_status == 0 && deepseek_runtime_counters != nullptr) {
      atomicAdd(
          reinterpret_cast<unsigned long long *>(
              deepseek_runtime_counters +
              kDeepSeekRuntimeCounterV3GroupedRouterSelections),
          1ull);
    }
  }
  __syncthreads();
  if (route_status != 0) {
    return;
  }

  const uint64_t expert_gate = layout.w_expert_gate_up;
  const uint64_t expert_gate_data_slots =
      bf16_storage
          ? static_cast<uint64_t>(num_experts) * moe_intermediate * hidden
          : deepseek_device_fp8_slots(
                static_cast<uint64_t>(num_experts) * moe_intermediate,
                hidden);
  const uint64_t expert_gate_scale =
      bf16_storage ? kMissingOffset : expert_gate + expert_gate_data_slots;
  const uint64_t expert_up =
      bf16_storage
          ? expert_gate + expert_gate_data_slots
          : expert_gate_scale +
                static_cast<uint64_t>(num_experts) *
                    deepseek_device_scale_f32_slots(moe_intermediate, hidden);
  const uint64_t expert_up_scale =
      bf16_storage
          ? kMissingOffset
          : expert_up +
                deepseek_device_fp8_slots(
                    static_cast<uint64_t>(num_experts) * moe_intermediate,
                    hidden);
  const uint64_t expert_down = layout.w_expert_down;
  const uint64_t expert_down_scale =
      bf16_storage
          ? kMissingOffset
          : expert_down +
                deepseek_device_fp8_slots(
                    static_cast<uint64_t>(num_experts) * hidden,
                    moe_intermediate);

  for (uint32_t rank = 0; rank < top_k; ++rank) {
    const uint32_t expert = selected_experts[rank];
    const float expert_weight = selected_weights[rank];
    for (uint32_t row = threadIdx.x; row < moe_intermediate;
         row += blockDim.x) {
      float gate_sum = 0.0f;
      float up_sum = 0.0f;
      for (uint32_t col = 0; col < hidden; ++col) {
        const float value = encoded_to_f32(token_norm[col], dtype);
        const float gate_weight =
            bf16_storage
                ? deepseek_bf16_rank3_weight(arena, expert_gate,
                                             moe_intermediate, hidden, expert,
                                             row, col)
                : deepseek_fp8_rank3_scaled_weight(
                      arena, expert_gate, expert_gate_scale, moe_intermediate,
                      hidden, expert, row, col);
        const float up_weight =
            bf16_storage
                ? deepseek_bf16_rank3_weight(arena, expert_up,
                                             moe_intermediate, hidden, expert,
                                             row, col)
                : deepseek_fp8_rank3_scaled_weight(
                      arena, expert_up, expert_up_scale, moe_intermediate,
                      hidden, expert, row, col);
        gate_sum += gate_weight * value;
        up_sum += up_weight * value;
      }
      token_gate[row] =
          deepseek_swiglu(gate_sum, up_sum, layout.deepseek_swiglu_limit);
    }
    __syncthreads();

    for (uint32_t row = threadIdx.x; row < hidden; row += blockDim.x) {
      float down_sum = 0.0f;
      for (uint32_t col = 0; col < moe_intermediate; ++col) {
        const float weight =
            bf16_storage
                ? deepseek_bf16_rank3_weight(arena, expert_down, hidden,
                                             moe_intermediate, expert, row, col)
                : deepseek_fp8_rank3_scaled_weight(
                      arena, expert_down, expert_down_scale, hidden,
                      moe_intermediate, expert, row, col);
        down_sum += weight * token_gate[col];
      }
      token_down[row] += expert_weight * down_sum;
    }
    __syncthreads();
  }

  const uint32_t shared_intermediate = layout.shared_expert_intermediate;
  if (shared_intermediate == 0 ||
      layout.w_shared_expert_gate == kMissingOffset ||
      layout.w_shared_expert_up == kMissingOffset ||
      layout.w_shared_expert_down == kMissingOffset ||
      shared_intermediate > intermediate) {
    return;
  }
  const uint64_t shared_gate_scale =
      bf16_storage ? kMissingOffset
                   : layout.w_shared_expert_gate +
                         deepseek_device_fp8_slots(shared_intermediate,
                                                   hidden);
  const uint64_t shared_up_scale =
      bf16_storage ? kMissingOffset
                   : layout.w_shared_expert_up +
                         deepseek_device_fp8_slots(shared_intermediate,
                                                   hidden);
  const uint64_t shared_down_scale =
      bf16_storage ? kMissingOffset
                   : layout.w_shared_expert_down +
                         deepseek_device_fp8_slots(hidden,
                                                   shared_intermediate);
  for (uint32_t row = threadIdx.x; row < shared_intermediate;
       row += blockDim.x) {
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint32_t col = 0; col < hidden; ++col) {
      const float value = encoded_to_f32(token_norm[col], dtype);
      const float gate_weight =
          bf16_storage
              ? deepseek_bf16_weight(arena, layout.w_shared_expert_gate,
                                     shared_intermediate, hidden, row, col)
              : deepseek_fp8_scaled_weight(
                    arena, layout.w_shared_expert_gate, shared_gate_scale,
                    shared_intermediate, hidden, row, col);
      const float up_weight =
          bf16_storage
              ? deepseek_bf16_weight(arena, layout.w_shared_expert_up,
                                     shared_intermediate, hidden, row, col)
              : deepseek_fp8_scaled_weight(
                    arena, layout.w_shared_expert_up, shared_up_scale,
                    shared_intermediate, hidden, row, col);
      gate_sum += gate_weight * value;
      up_sum += up_weight * value;
    }
    token_gate[row] =
        deepseek_swiglu(gate_sum, up_sum, layout.deepseek_swiglu_limit);
  }
  __syncthreads();

  for (uint32_t row = threadIdx.x; row < hidden; row += blockDim.x) {
    float down_sum = 0.0f;
    for (uint32_t col = 0; col < shared_intermediate; ++col) {
      const float weight =
          bf16_storage
              ? deepseek_bf16_weight(arena, layout.w_shared_expert_down,
                                     hidden, shared_intermediate, row, col)
              : deepseek_fp8_scaled_weight(
                    arena, layout.w_shared_expert_down, shared_down_scale,
                    hidden, shared_intermediate, row, col);
      down_sum += weight * token_gate[col];
    }
    token_down[row] += down_sum;
  }
}

__global__ void hf_deepseek_v3_sparse_moe_expert_down_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t rank, uint32_t *step_cursor,
    uint32_t max_steps, float *scratch) {
  (void)dtype;
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint32_t *route_ids = deepseek_moe_route_ids(s);
  const float *route_weights = deepseek_moe_route_weights(s);
  if (route_ids[0] != 0u || rank >= layout.experts_per_token) {
    return;
  }
  const uint32_t row = blockIdx.x;
  const uint32_t num_experts = layout.num_experts;
  const uint32_t moe_intermediate = layout.moe_intermediate;
  const uint32_t top_k = layout.experts_per_token;
  if (top_k == 0) {
    return;
  }
  const float *rank_ff = deepseek_moe_rank_ff(
      scratch, s, hidden, attention_hidden, kv_hidden, intermediate, rank);
  float *rank_down = deepseek_moe_rank_down(
      scratch, s, hidden, attention_hidden, kv_hidden, intermediate, top_k,
      rank);
  if (row >= hidden || num_experts == 0 ||
      moe_intermediate == 0 || moe_intermediate > intermediate) {
    return;
  }
  const uint32_t expert = route_ids[rank + 1u];
  const float expert_weight = route_weights[rank];
  const uint64_t expert_down = layout.w_expert_down;
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  const uint64_t expert_down_scale =
      bf16_storage
          ? kMissingOffset
          : expert_down +
                deepseek_device_fp8_slots(
                    static_cast<uint64_t>(num_experts) * hidden,
                    moe_intermediate);
  float down_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < moe_intermediate; col += blockDim.x) {
    const float weight =
        bf16_storage
            ? deepseek_bf16_rank3_weight(arena, expert_down, hidden,
                                         moe_intermediate, expert, row, col)
            : deepseek_fp8_rank3_scaled_weight(
                  arena, expert_down, expert_down_scale, hidden,
                  moe_intermediate, expert, row, col);
    down_sum += weight * rank_ff[col];
  }
  down_sum = block_sum(down_sum);
  if (threadIdx.x == 0) {
    rank_down[row] = expert_weight * down_sum;
  }
}

__global__ void hf_deepseek_v3_sparse_moe_shared_gate_up_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch) {
  (void)dtype;
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint32_t row = blockIdx.x;
  const uint32_t shared_intermediate = layout.shared_expert_intermediate;
  if (shared_intermediate == 0 || row >= shared_intermediate ||
      layout.w_shared_expert_gate == kMissingOffset ||
      layout.w_shared_expert_up == kMissingOffset ||
      layout.w_shared_expert_down == kMissingOffset ||
      shared_intermediate > intermediate) {
    return;
  }
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  const uint64_t shared_gate_scale =
      bf16_storage ? kMissingOffset
                   : layout.w_shared_expert_gate +
                         deepseek_device_fp8_slots(shared_intermediate,
                                                   hidden);
  const uint64_t shared_up_scale =
      bf16_storage ? kMissingOffset
                   : layout.w_shared_expert_up +
                         deepseek_device_fp8_slots(shared_intermediate,
                                                   hidden);
  float gate_sum = 0.0f;
  float up_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float gate_weight =
        bf16_storage
            ? deepseek_bf16_weight(arena, layout.w_shared_expert_gate,
                                   shared_intermediate, hidden, row, col)
            : deepseek_fp8_scaled_weight(
                  arena, layout.w_shared_expert_gate, shared_gate_scale,
                  shared_intermediate, hidden, row, col);
    const float up_weight =
        bf16_storage
            ? deepseek_bf16_weight(arena, layout.w_shared_expert_up,
                                   shared_intermediate, hidden, row, col)
            : deepseek_fp8_scaled_weight(
                  arena, layout.w_shared_expert_up, shared_up_scale,
                  shared_intermediate, hidden, row, col);
    gate_sum += gate_weight * s.mlp_norm[col];
    up_sum += up_weight * s.mlp_norm[col];
  }
  gate_sum = block_sum(gate_sum);
  up_sum = block_sum(up_sum);
  if (threadIdx.x == 0) {
    s.ff[row] =
        deepseek_swiglu(gate_sum, up_sum, layout.deepseek_swiglu_limit);
  }
}

__global__ void hf_deepseek_v3_sparse_moe_shared_down_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch) {
  (void)dtype;
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
  const uint32_t row = blockIdx.x;
  const uint32_t shared_intermediate = layout.shared_expert_intermediate;
  if (shared_intermediate == 0 || row >= hidden ||
      layout.w_shared_expert_down == kMissingOffset ||
      shared_intermediate > intermediate) {
    return;
  }
  const bool bf16_storage = layout.deepseek_storage == kDeepSeekStorageBf16;
  const uint64_t shared_down_scale =
      bf16_storage ? kMissingOffset
                   : layout.w_shared_expert_down +
                         deepseek_device_fp8_slots(hidden,
                                                   shared_intermediate);
  float down_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < shared_intermediate;
       col += blockDim.x) {
    const float weight =
        bf16_storage
            ? deepseek_bf16_weight(arena, layout.w_shared_expert_down, hidden,
                                   shared_intermediate, row, col)
            : deepseek_fp8_scaled_weight(
                  arena, layout.w_shared_expert_down, shared_down_scale,
                  hidden, shared_intermediate, row, col);
    down_sum += weight * s.ff[col];
  }
  down_sum = block_sum(down_sum);
  if (threadIdx.x == 0) {
    s.down[row] += down_sum;
  }
}
