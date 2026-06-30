uint64_t deepseek_norm_slots(const NervaCudaHfDecodeChainLayer &layer,
                             uint64_t rows) {
  return layer.deepseek_mode == kDeepSeekModeV32MlaIndexer
             ? f32_slots(rows, 1)
             : bf16_slots(rows, 1);
}

struct DeepSeekV4CompressorOffsets {
  uint64_t ape;
  uint64_t wkv;
  uint64_t wgate;
  uint64_t norm;
};

void initialize_deepseek_layout_metadata(
    SequenceLayerLayout &layout, const NervaCudaHfDecodeChainLayer &layer) {
  layout.deepseek_mode = layer.deepseek_mode;
  layout.deepseek_flags = layer.deepseek_flags;
  layout.deepseek_hc_mult = layer.deepseek_hc_mult;
  layout.deepseek_hc_sinkhorn_iters = layer.deepseek_hc_sinkhorn_iters;
  layout.deepseek_q_lora_rank = layer.deepseek_q_lora_rank;
  layout.deepseek_kv_lora_rank = layer.deepseek_kv_lora_rank;
  layout.deepseek_o_lora_rank = layer.deepseek_o_lora_rank;
  layout.deepseek_o_groups = layer.deepseek_o_groups;
  layout.deepseek_qk_nope_head_dim = layer.deepseek_qk_nope_head_dim;
  layout.deepseek_qk_rope_head_dim = layer.deepseek_qk_rope_head_dim;
  layout.deepseek_v_head_dim = layer.deepseek_v_head_dim;
  layout.deepseek_compress_ratio = layer.deepseek_compress_ratio;
  layout.deepseek_index_topk = layer.deepseek_index_topk;
  layout.deepseek_index_n_heads = layer.deepseek_index_n_heads;
  layout.deepseek_index_head_dim = layer.deepseek_index_head_dim;
  layout.deepseek_router_num_groups = layer.deepseek_router_num_groups;
  layout.deepseek_router_topk_groups = layer.deepseek_router_topk_groups;
  layout.deepseek_routed_scaling_factor =
      layer.deepseek_routed_scaling_factor;
  layout.deepseek_hc_eps = layer.deepseek_hc_eps;
  layout.deepseek_hc_post_alpha = layer.deepseek_hc_post_alpha;
}

void initialize_deepseek_layout_offsets(SequenceLayerLayout &layout) {
  layout.deepseek_q_a_scale = kMissingOffset;
  layout.deepseek_q_b = kMissingOffset;
  layout.deepseek_q_b_scale = kMissingOffset;
  layout.deepseek_kv_a_scale = kMissingOffset;
  layout.deepseek_kv_b_scale = kMissingOffset;
  layout.deepseek_o_a_scale = kMissingOffset;
  layout.deepseek_o_b = kMissingOffset;
  layout.deepseek_o_b_scale = kMissingOffset;
  layout.deepseek_hc_attn_base = kMissingOffset;
  layout.deepseek_hc_attn_fn = kMissingOffset;
  layout.deepseek_hc_attn_scale = kMissingOffset;
  layout.deepseek_hc_ffn_base = kMissingOffset;
  layout.deepseek_hc_ffn_fn = kMissingOffset;
  layout.deepseek_hc_ffn_scale = kMissingOffset;
  layout.deepseek_attention_sink = kMissingOffset;
  layout.deepseek_indexer_q = kMissingOffset;
  layout.deepseek_indexer_q_scale = kMissingOffset;
  layout.deepseek_indexer_k = kMissingOffset;
  layout.deepseek_indexer_k_scale = kMissingOffset;
  layout.deepseek_indexer_k_norm = kMissingOffset;
  layout.deepseek_indexer_k_norm_bias = kMissingOffset;
  layout.deepseek_indexer_weights = kMissingOffset;
  layout.deepseek_compressor_ape = kMissingOffset;
  layout.deepseek_compressor_wkv = kMissingOffset;
  layout.deepseek_compressor_wgate = kMissingOffset;
  layout.deepseek_compressor_norm = kMissingOffset;
  layout.deepseek_indexer_compressor_ape = kMissingOffset;
  layout.deepseek_indexer_compressor_wkv = kMissingOffset;
  layout.deepseek_indexer_compressor_wgate = kMissingOffset;
  layout.deepseek_indexer_compressor_norm = kMissingOffset;
}

DeepSeekV4CompressorOffsets push_deepseek_v4_compressor(
    uint64_t &cursor, uint64_t compress_ratio, uint64_t hidden,
    uint64_t head_dim) {
  const uint64_t coff = compress_ratio == 4 ? 2u : 1u;
  const uint64_t rows = head_dim * coff;
  DeepSeekV4CompressorOffsets offsets{};
  offsets.ape = push(cursor, f32_slots(compress_ratio, rows));
  offsets.wkv = push(cursor, bf16_slots(rows, hidden));
  offsets.wgate = push(cursor, bf16_slots(rows, hidden));
  offsets.norm = push(cursor, bf16_slots(head_dim, 1));
  return offsets;
}

void pack_deepseek_static(SequenceArenaLayout &arena_layout, uint64_t &cursor,
                          const NervaCudaHfDecodeChainLayer *layers,
                          uint32_t layer_count, uint64_t hidden) {
  if (layers == nullptr || layer_count == 0) {
    return;
  }
  const NervaCudaHfDecodeChainLayer &layer = layers[0];
  if (layer.attention_kind != kAttentionKindDeepSeekMla ||
      !deepseek_is_v4(layer.deepseek_mode) || layer.deepseek_hc_mult == 0) {
    return;
  }
  const uint64_t hc_mult = layer.deepseek_hc_mult;
  const uint64_t hc_dim = hidden * hc_mult;
  arena_layout.deepseek_hc_head_base = push(cursor, f32_slots(hc_mult, 1));
  arena_layout.deepseek_hc_head_fn = push(cursor, f32_slots(hc_mult, hc_dim));
  arena_layout.deepseek_hc_head_scale = push(cursor, f32_slots(1, 1));
}

void pack_deepseek_v3_attention(SequenceLayerLayout &layout, uint64_t &cursor,
                                const NervaCudaHfDecodeChainLayer &layer,
                                uint64_t hidden, uint64_t attention_hidden,
                                uint64_t head_dim) {
  const uint64_t heads = attention_hidden / head_dim;
  const uint64_t q_lora_rank = layer.deepseek_q_lora_rank;
  const uint64_t kv_lora_rank = layer.deepseek_kv_lora_rank;
  const uint64_t qk_nope = layer.deepseek_qk_nope_head_dim;
  const uint64_t qk_rope = layer.deepseek_qk_rope_head_dim;
  const uint64_t v_head = layer.deepseek_v_head_dim;
  const uint64_t q_rows = heads * (qk_nope + qk_rope);
  const uint64_t kv_a_rows = kv_lora_rank + qk_rope;
  const uint64_t kv_b_rows = heads * (qk_nope + v_head);
  const uint64_t value_hidden = heads * v_head;

  layout.w_q = push(cursor, fp8_slots(q_lora_rank, hidden));
  layout.deepseek_q_a_scale = push(cursor, scale_f32_slots(q_lora_rank, hidden));
  layout.q_norm = push(cursor, deepseek_norm_slots(layer, q_lora_rank));
  layout.deepseek_q_b = push(cursor, fp8_slots(q_rows, q_lora_rank));
  layout.deepseek_q_b_scale =
      push(cursor, scale_f32_slots(q_rows, q_lora_rank));
  layout.w_k = push(cursor, fp8_slots(kv_a_rows, hidden));
  layout.deepseek_kv_a_scale =
      push(cursor, scale_f32_slots(kv_a_rows, hidden));
  layout.k_norm = push(cursor, deepseek_norm_slots(layer, kv_lora_rank));
  layout.w_v = push(cursor, fp8_slots(kv_b_rows, kv_lora_rank));
  layout.deepseek_kv_b_scale =
      push(cursor, scale_f32_slots(kv_b_rows, kv_lora_rank));
  layout.w_o = push(cursor, fp8_slots(hidden, value_hidden));
  layout.deepseek_o_a_scale =
      push(cursor, scale_f32_slots(hidden, value_hidden));

  if ((layer.deepseek_flags & kDeepSeekFlagSparseIndexer) != 0) {
    const uint64_t index_rows = static_cast<uint64_t>(layer.deepseek_index_n_heads) *
                                layer.deepseek_index_head_dim;
    layout.deepseek_indexer_q =
        push(cursor, fp8_slots(index_rows, q_lora_rank));
    layout.deepseek_indexer_q_scale =
        push(cursor, scale_f32_slots(index_rows, q_lora_rank));
    layout.deepseek_indexer_k =
        push(cursor, fp8_slots(layer.deepseek_index_head_dim, hidden));
    layout.deepseek_indexer_k_scale =
        push(cursor, scale_f32_slots(layer.deepseek_index_head_dim, hidden));
    layout.deepseek_indexer_k_norm =
        push(cursor, f32_slots(layer.deepseek_index_head_dim, 1));
    layout.deepseek_indexer_k_norm_bias =
        push(cursor, f32_slots(layer.deepseek_index_head_dim, 1));
    layout.deepseek_indexer_weights =
        push(cursor, bf16_slots(layer.deepseek_index_n_heads, hidden));
  }
}

void pack_deepseek_v4_attention(SequenceLayerLayout &layout, uint64_t &cursor,
                                const NervaCudaHfDecodeChainLayer &layer,
                                uint64_t hidden, uint64_t attention_hidden,
                                uint64_t head_dim) {
  const uint64_t heads = attention_hidden / head_dim;
  const uint64_t hc_mult = layer.deepseek_hc_mult;
  const uint64_t hc_dim = hidden * hc_mult;
  const uint64_t mix_hc = hc_mult * (hc_mult + 2u);
  const uint64_t q_lora_rank = layer.deepseek_q_lora_rank;
  const uint64_t q_rows = attention_hidden;
  const uint64_t wo_a_rows =
      static_cast<uint64_t>(layer.deepseek_o_groups) * layer.deepseek_o_lora_rank;
  const uint64_t wo_a_cols = q_rows / layer.deepseek_o_groups;

  layout.deepseek_hc_attn_base = push(cursor, f32_slots(mix_hc, 1));
  layout.deepseek_hc_attn_fn = push(cursor, f32_slots(mix_hc, hc_dim));
  layout.deepseek_hc_attn_scale = push(cursor, f32_slots(3, 1));
  layout.deepseek_hc_ffn_base = push(cursor, f32_slots(mix_hc, 1));
  layout.deepseek_hc_ffn_fn = push(cursor, f32_slots(mix_hc, hc_dim));
  layout.deepseek_hc_ffn_scale = push(cursor, f32_slots(3, 1));
  layout.deepseek_attention_sink = push(cursor, f32_slots(heads, 1));
  layout.w_q = push(cursor, fp8_slots(q_lora_rank, hidden));
  layout.deepseek_q_a_scale =
      push(cursor, scale_e8m0_slots(q_lora_rank, hidden));
  layout.deepseek_q_b = push(cursor, fp8_slots(q_rows, q_lora_rank));
  layout.deepseek_q_b_scale =
      push(cursor, scale_e8m0_slots(q_rows, q_lora_rank));
  layout.q_norm = push(cursor, bf16_slots(q_lora_rank, 1));
  layout.w_k = push(cursor, fp8_slots(head_dim, hidden));
  layout.deepseek_kv_a_scale =
      push(cursor, scale_e8m0_slots(head_dim, hidden));
  layout.k_norm = push(cursor, bf16_slots(head_dim, 1));
  layout.w_o = push(cursor, fp8_slots(wo_a_rows, wo_a_cols));
  layout.deepseek_o_a_scale =
      push(cursor, scale_e8m0_slots(wo_a_rows, wo_a_cols));
  layout.deepseek_o_b = push(cursor, fp8_slots(hidden, wo_a_rows));
  layout.deepseek_o_b_scale =
      push(cursor, scale_e8m0_slots(hidden, wo_a_rows));

  if ((layer.deepseek_flags & kDeepSeekFlagCompressor) != 0 &&
      layer.deepseek_compress_ratio > 1) {
    const DeepSeekV4CompressorOffsets offsets =
        push_deepseek_v4_compressor(cursor, layer.deepseek_compress_ratio,
                                    hidden, head_dim);
    layout.deepseek_compressor_ape = offsets.ape;
    layout.deepseek_compressor_wkv = offsets.wkv;
    layout.deepseek_compressor_wgate = offsets.wgate;
    layout.deepseek_compressor_norm = offsets.norm;
  }
  if (layer.deepseek_compress_ratio == 4) {
    const uint64_t index_rows = static_cast<uint64_t>(layer.deepseek_index_n_heads) *
                                layer.deepseek_index_head_dim;
    layout.deepseek_indexer_q =
        push(cursor, fp8_slots(index_rows, q_lora_rank));
    layout.deepseek_indexer_q_scale =
        push(cursor, scale_e8m0_slots(index_rows, q_lora_rank));
    const DeepSeekV4CompressorOffsets offsets =
        push_deepseek_v4_compressor(cursor, 4, hidden,
                                    layer.deepseek_index_head_dim);
    layout.deepseek_indexer_compressor_ape = offsets.ape;
    layout.deepseek_indexer_compressor_wkv = offsets.wkv;
    layout.deepseek_indexer_compressor_wgate = offsets.wgate;
    layout.deepseek_indexer_compressor_norm = offsets.norm;
    layout.deepseek_indexer_weights =
        push(cursor, bf16_slots(layer.deepseek_index_n_heads, hidden));
  }
}

void pack_deepseek_dense_mlp(SequenceLayerLayout &layout, uint64_t &cursor,
                             uint64_t hidden, uint64_t intermediate) {
  layout.w_gate = push(cursor, fp8_slots(intermediate, hidden));
  push(cursor, scale_f32_slots(intermediate, hidden));
  layout.w_up = push(cursor, fp8_slots(intermediate, hidden));
  push(cursor, scale_f32_slots(intermediate, hidden));
  layout.w_down = push(cursor, fp8_slots(hidden, intermediate));
  push(cursor, scale_f32_slots(hidden, intermediate));
}

void pack_deepseek_v3_moe(SequenceLayerLayout &layout, uint64_t &cursor,
                          const NervaCudaHfDecodeChainLayer &layer,
                          uint64_t hidden) {
  const uint64_t num_experts = layer.num_experts;
  const uint64_t moe_intermediate = layer.moe_intermediate;
  const uint64_t shared_intermediate = layer.shared_expert_intermediate;
  layout.w_router = push(cursor, bf16_slots(num_experts, hidden));
  if ((layer.deepseek_flags & kDeepSeekFlagRouterBias) != 0) {
    push(cursor, f32_slots(num_experts, 1));
  }
  if (shared_intermediate != 0) {
    layout.w_shared_expert_gate =
        push(cursor, fp8_slots(shared_intermediate, hidden));
    push(cursor, scale_f32_slots(shared_intermediate, hidden));
    layout.w_shared_expert_up =
        push(cursor, fp8_slots(shared_intermediate, hidden));
    push(cursor, scale_f32_slots(shared_intermediate, hidden));
    layout.w_shared_expert_down =
        push(cursor, fp8_slots(hidden, shared_intermediate));
    push(cursor, scale_f32_slots(hidden, shared_intermediate));
  }
  layout.w_expert_gate_up = cursor;
  push(cursor, rank3_slots(num_experts, moe_intermediate, hidden, 1));
  push(cursor, rank3_f32_slots(num_experts, scale_dim(moe_intermediate),
                               scale_dim(hidden)));
  push(cursor, rank3_slots(num_experts, moe_intermediate, hidden, 1));
  push(cursor, rank3_f32_slots(num_experts, scale_dim(moe_intermediate),
                               scale_dim(hidden)));
  layout.w_expert_down =
      push(cursor, rank3_slots(num_experts, hidden, moe_intermediate, 1));
  push(cursor, rank3_f32_slots(num_experts, scale_dim(hidden),
                               scale_dim(moe_intermediate)));
}

void pack_deepseek_v4_moe(SequenceLayerLayout &layout, uint64_t &cursor,
                          const NervaCudaHfDecodeChainLayer &layer,
                          uint64_t hidden, uint64_t vocab_size) {
  const uint64_t num_experts = layer.num_experts;
  const uint64_t top_k = layer.experts_per_token;
  const uint64_t moe_intermediate = layer.moe_intermediate;
  const uint64_t shared_intermediate = layer.shared_expert_intermediate;
  layout.w_router = push(cursor, bf16_slots(num_experts, hidden));
  if ((layer.deepseek_flags & kDeepSeekFlagHashRouter) != 0) {
    push(cursor, i64_slots(vocab_size, top_k));
  } else {
    push(cursor, f32_slots(num_experts, 1));
  }
  if (shared_intermediate != 0) {
    layout.w_shared_expert_gate =
        push(cursor, fp8_slots(shared_intermediate, hidden));
    push(cursor, scale_e8m0_slots(shared_intermediate, hidden));
    layout.w_shared_expert_up =
        push(cursor, fp8_slots(shared_intermediate, hidden));
    push(cursor, scale_e8m0_slots(shared_intermediate, hidden));
    layout.w_shared_expert_down =
        push(cursor, fp8_slots(hidden, shared_intermediate));
    push(cursor, scale_e8m0_slots(hidden, shared_intermediate));
  }
  const uint64_t half_hidden = hidden / 2u;
  const uint64_t half_intermediate = moe_intermediate / 2u;
  layout.w_expert_gate_up = cursor;
  push(cursor, rank3_slots(num_experts, moe_intermediate, half_hidden, 1));
  push(cursor, rank3_slots(num_experts, moe_intermediate,
                           ceil_div_u64_local(half_hidden, 16), 1));
  push(cursor, rank3_slots(num_experts, moe_intermediate, half_hidden, 1));
  push(cursor, rank3_slots(num_experts, moe_intermediate,
                           ceil_div_u64_local(half_hidden, 16), 1));
  layout.w_expert_down =
      push(cursor, rank3_slots(num_experts, hidden, half_intermediate, 1));
  push(cursor, rank3_slots(num_experts, hidden,
                           ceil_div_u64_local(half_intermediate, 16), 1));
}

void pack_deepseek_layer(SequenceLayerLayout &layout, uint64_t &cursor,
                         const NervaCudaHfDecodeChainLayer &layer,
                         uint64_t hidden, uint64_t attention_hidden,
                         uint64_t head_dim, uint64_t intermediate,
                         uint64_t vocab_size) {
  layout.rms_attn = push(cursor, deepseek_norm_slots(layer, hidden));
  if (deepseek_is_v4(layer.deepseek_mode)) {
    pack_deepseek_v4_attention(layout, cursor, layer, hidden, attention_hidden,
                               head_dim);
  } else {
    pack_deepseek_v3_attention(layout, cursor, layer, hidden, attention_hidden,
                               head_dim);
  }
  layout.rms_mlp = push(cursor, deepseek_norm_slots(layer, hidden));
  if (layer.mlp_kind != kMlpKindSparseMoe) {
    pack_deepseek_dense_mlp(layout, cursor, hidden, intermediate);
  } else if (deepseek_is_v4(layer.deepseek_mode)) {
    pack_deepseek_v4_moe(layout, cursor, layer, hidden, vocab_size);
  } else {
    pack_deepseek_v3_moe(layout, cursor, layer, hidden);
  }
}
