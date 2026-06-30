__global__ void hf_deepseek_v3_sparse_moe_encode_kernel(
    uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t attention_hidden, uint32_t kv_hidden,
    uint32_t intermediate, uint32_t *step_cursor, uint32_t max_steps,
    float *scratch, uint16_t *projection_input,
    uint64_t *deepseek_runtime_counters) {
  if (step_cursor != nullptr && *step_cursor >= max_steps) {
    return;
  }
  __shared__ float router_logits[kSparseMoeExpertsMax];
  __shared__ float correction_bias[kSparseMoeExpertsMax];
  __shared__ uint32_t selected_experts[kSparseMoeTopKMax];
  __shared__ float selected_weights[kSparseMoeTopKMax];
  __shared__ int route_status;

  LayerScratch s =
      layer_scratch_ptrs(scratch, hidden, attention_hidden, kv_hidden, intermediate);
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
      moe_intermediate == 0 || moe_intermediate > intermediate) {
    return;
  }

  for (uint32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    float sum = 0.0f;
    const uint64_t row = layout.w_router +
                         static_cast<uint64_t>(expert) * hidden;
    for (uint32_t col = 0; col < hidden; ++col) {
      sum += encoded_to_f32(arena[row + col], kDTypeBF16) * s.mlp_norm[col];
    }
    router_logits[expert] = sum;
  }
  const bool has_router_bias =
      (layout.deepseek_flags & kDeepSeekFlagRouterBias) != 0;
  const uint64_t router_bias_offset =
      layout.w_router + static_cast<uint64_t>(num_experts) * hidden;
  for (uint32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    correction_bias[expert] =
        has_router_bias ? f32_from_u16_slots(arena + router_bias_offset, expert)
                        : 0.0f;
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
  __syncthreads();

  const uint64_t expert_gate = layout.w_expert_gate_up;
  const uint64_t expert_gate_scale =
      expert_gate +
      deepseek_device_fp8_slots(
          static_cast<uint64_t>(num_experts) * moe_intermediate, hidden);
  const uint64_t expert_up =
      expert_gate_scale +
      static_cast<uint64_t>(num_experts) *
          deepseek_device_scale_f32_slots(moe_intermediate, hidden);
  const uint64_t expert_up_scale =
      expert_up +
      deepseek_device_fp8_slots(
          static_cast<uint64_t>(num_experts) * moe_intermediate, hidden);
  const uint64_t expert_down = layout.w_expert_down;
  const uint64_t expert_down_scale =
      expert_down +
      deepseek_device_fp8_slots(
          static_cast<uint64_t>(num_experts) * hidden, moe_intermediate);

  for (uint32_t rank = 0; rank < top_k; ++rank) {
    const uint32_t expert = selected_experts[rank];
    const float expert_weight = selected_weights[rank];
    for (uint32_t row = threadIdx.x; row < moe_intermediate;
         row += blockDim.x) {
      float gate_sum = 0.0f;
      float up_sum = 0.0f;
      for (uint32_t col = 0; col < hidden; ++col) {
        gate_sum += deepseek_fp8_rank3_scaled_weight(
                        arena, expert_gate, expert_gate_scale,
                        moe_intermediate, hidden, expert, row, col) *
                    s.mlp_norm[col];
        up_sum += deepseek_fp8_rank3_scaled_weight(
                      arena, expert_up, expert_up_scale, moe_intermediate,
                      hidden, expert, row, col) *
                  s.mlp_norm[col];
      }
      s.gate[row] = gate_sum;
      s.up[row] = up_sum;
      s.ff[row] = silu(gate_sum) * up_sum;
    }
    __syncthreads();
    for (uint32_t row = threadIdx.x; row < hidden; row += blockDim.x) {
      float down_sum = 0.0f;
      for (uint32_t col = 0; col < moe_intermediate; ++col) {
        down_sum += deepseek_fp8_rank3_scaled_weight(
                        arena, expert_down, expert_down_scale, hidden,
                        moe_intermediate, expert, row, col) *
                    s.ff[col];
      }
      s.down[row] += expert_weight * down_sum;
    }
    __syncthreads();
  }

  const uint32_t shared_intermediate = layout.shared_expert_intermediate;
  if (shared_intermediate != 0) {
    if (layout.w_shared_expert_gate == kMissingOffset ||
        layout.w_shared_expert_up == kMissingOffset ||
        layout.w_shared_expert_down == kMissingOffset ||
        shared_intermediate > intermediate) {
      return;
    }
    const uint64_t shared_gate_scale =
        layout.w_shared_expert_gate +
        deepseek_device_fp8_slots(shared_intermediate, hidden);
    const uint64_t shared_up_scale =
        layout.w_shared_expert_up +
        deepseek_device_fp8_slots(shared_intermediate, hidden);
    const uint64_t shared_down_scale =
        layout.w_shared_expert_down +
        deepseek_device_fp8_slots(hidden, shared_intermediate);
    for (uint32_t row = threadIdx.x; row < shared_intermediate;
         row += blockDim.x) {
      float gate_sum = 0.0f;
      float up_sum = 0.0f;
      for (uint32_t col = 0; col < hidden; ++col) {
        gate_sum += deepseek_fp8_scaled_weight(
                        arena, layout.w_shared_expert_gate,
                        shared_gate_scale, shared_intermediate, hidden, row,
                        col) *
                    s.mlp_norm[col];
        up_sum += deepseek_fp8_scaled_weight(
                      arena, layout.w_shared_expert_up, shared_up_scale,
                      shared_intermediate, hidden, row, col) *
                  s.mlp_norm[col];
      }
      s.ff[row] = silu(gate_sum) * up_sum;
    }
    __syncthreads();
    for (uint32_t row = threadIdx.x; row < hidden; row += blockDim.x) {
      float down_sum = 0.0f;
      for (uint32_t col = 0; col < shared_intermediate; ++col) {
        down_sum += deepseek_fp8_scaled_weight(
                        arena, layout.w_shared_expert_down,
                        shared_down_scale, hidden, shared_intermediate, row,
                        col) *
                    s.ff[col];
      }
      s.down[row] += down_sum;
    }
    __syncthreads();
  }
}
