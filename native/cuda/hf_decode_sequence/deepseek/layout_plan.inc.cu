void fill_deepseek_v3_mla_shape(
    const NervaCudaHfDecodeChainLayer &source_layer,
    NervaCudaHfDecodeSequenceLayoutPlanResult *out) {
  if (out == nullptr ||
      source_layer.attention_kind != kAttentionKindDeepSeekMla ||
      (source_layer.deepseek_mode != kDeepSeekModeV3Mla &&
       source_layer.deepseek_mode != kDeepSeekModeV32MlaIndexer)) {
    return;
  }
  const uint32_t qk_head_dim =
      source_layer.deepseek_qk_nope_head_dim +
      source_layer.deepseek_qk_rope_head_dim;
  if (qk_head_dim < source_layer.deepseek_qk_nope_head_dim) {
    return;
  }
  const uint32_t kv_cache_width =
      source_layer.deepseek_kv_lora_rank +
      source_layer.deepseek_qk_rope_head_dim;
  if (kv_cache_width < source_layer.deepseek_kv_lora_rank) {
    return;
  }
  const uint32_t kv_b_head_rows =
      source_layer.deepseek_qk_nope_head_dim + source_layer.deepseek_v_head_dim;
  if (kv_b_head_rows < source_layer.deepseek_qk_nope_head_dim) {
    return;
  }
  out->deepseek_qk_head_dim = qk_head_dim;
  out->deepseek_q_rows = checked_u32_product(out->heads, qk_head_dim);
  out->deepseek_kv_cache_width = kv_cache_width;
  out->deepseek_kv_b_rows = checked_u32_product(out->heads, kv_b_head_rows);
  out->deepseek_value_rows =
      checked_u32_product(out->heads, source_layer.deepseek_v_head_dim);
}

void fill_deepseek_layout_plan_result(
    const NervaCudaHfDecodeChainLayer &source_layer,
    const SequenceArenaLayout &arena_layout, const SequenceLayerLayout &layout,
    NervaCudaHfDecodeSequenceLayoutPlanResult *out) {
  if (out == nullptr) {
    return;
  }
  out->deepseek_mode = layout.deepseek_mode;
  out->deepseek_flags = layout.deepseek_flags;
  out->deepseek_hc_mult = layout.deepseek_hc_mult;
  out->deepseek_hc_sinkhorn_iters = layout.deepseek_hc_sinkhorn_iters;
  out->deepseek_hc_eps = layout.deepseek_hc_eps;
  out->deepseek_hc_post_alpha = layout.deepseek_hc_post_alpha;
  out->deepseek_swiglu_limit = layout.deepseek_swiglu_limit;
  fill_deepseek_v3_mla_shape(source_layer, out);
  out->deepseek_q_a_scale = layout.deepseek_q_a_scale;
  out->deepseek_q_b = layout.deepseek_q_b;
  out->deepseek_q_b_scale = layout.deepseek_q_b_scale;
  out->deepseek_kv_a_scale = layout.deepseek_kv_a_scale;
  out->deepseek_kv_b_scale = layout.deepseek_kv_b_scale;
  out->deepseek_o_a_scale = layout.deepseek_o_a_scale;
  out->deepseek_o_b = layout.deepseek_o_b;
  out->deepseek_o_b_scale = layout.deepseek_o_b_scale;
  out->deepseek_hc_head_base = arena_layout.deepseek_hc_head_base;
  out->deepseek_hc_head_fn = arena_layout.deepseek_hc_head_fn;
  out->deepseek_hc_head_scale = arena_layout.deepseek_hc_head_scale;
  out->deepseek_hc_attn_base = layout.deepseek_hc_attn_base;
  out->deepseek_hc_attn_fn = layout.deepseek_hc_attn_fn;
  out->deepseek_hc_attn_scale = layout.deepseek_hc_attn_scale;
  out->deepseek_hc_ffn_base = layout.deepseek_hc_ffn_base;
  out->deepseek_hc_ffn_fn = layout.deepseek_hc_ffn_fn;
  out->deepseek_hc_ffn_scale = layout.deepseek_hc_ffn_scale;
  out->deepseek_attention_sink = layout.deepseek_attention_sink;
  out->deepseek_indexer_q = layout.deepseek_indexer_q;
  out->deepseek_indexer_q_scale = layout.deepseek_indexer_q_scale;
  out->deepseek_indexer_k = layout.deepseek_indexer_k;
  out->deepseek_indexer_k_scale = layout.deepseek_indexer_k_scale;
  out->deepseek_indexer_k_norm = layout.deepseek_indexer_k_norm;
  out->deepseek_indexer_k_norm_bias = layout.deepseek_indexer_k_norm_bias;
  out->deepseek_indexer_weights = layout.deepseek_indexer_weights;
  out->deepseek_compressor_ape = layout.deepseek_compressor_ape;
  out->deepseek_compressor_wkv = layout.deepseek_compressor_wkv;
  out->deepseek_compressor_wgate = layout.deepseek_compressor_wgate;
  out->deepseek_compressor_norm = layout.deepseek_compressor_norm;
  out->deepseek_indexer_compressor_ape =
      layout.deepseek_indexer_compressor_ape;
  out->deepseek_indexer_compressor_wkv =
      layout.deepseek_indexer_compressor_wkv;
  out->deepseek_indexer_compressor_wgate =
      layout.deepseek_indexer_compressor_wgate;
  out->deepseek_indexer_compressor_norm =
      layout.deepseek_indexer_compressor_norm;
}
