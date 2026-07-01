bool layout_is_deepseek_v3_mla(const SequenceLayerLayout &layout) {
  return layout.attention_kind == kAttentionKindDeepSeekMla &&
         (layout.deepseek_mode == kDeepSeekModeV3Mla ||
          layout.deepseek_mode == kDeepSeekModeV32MlaIndexer);
}

bool layout_is_deepseek_v32_mla_packed(const SequenceLayerLayout &layout) {
  return layout.attention_kind == kAttentionKindDeepSeekMla &&
         layout.deepseek_mode == kDeepSeekModeV32MlaIndexer &&
         layout.deepseek_kv_lora_rank == kDeepSeekV32PackedKvNopeBytes &&
         layout.deepseek_qk_rope_head_dim == kDeepSeekV32PackedKvRopeValues;
}

bool layout_is_deepseek_v32_indexer_native(
    const SequenceLayerLayout &layout) {
  return layout.attention_kind == kAttentionKindDeepSeekMla &&
         layout.deepseek_mode == kDeepSeekModeV32MlaIndexer &&
         (layout.deepseek_flags & kDeepSeekFlagSparseIndexer) != 0 &&
         layout.deepseek_index_head_dim != 0 &&
         layout.deepseek_indexer_k != kMissingOffset &&
         layout.deepseek_indexer_k_scale != kMissingOffset &&
         layout.deepseek_indexer_k_norm != kMissingOffset &&
         layout.deepseek_indexer_k_norm_bias != kMissingOffset;
}

bool layout_is_deepseek_v32_indexer_query_native(
    const SequenceLayerLayout &layout) {
  return layout_is_deepseek_v32_indexer_native(layout) &&
         layout.deepseek_q_lora_rank != 0 &&
         layout.deepseek_index_n_heads != 0 &&
         layout.deepseek_index_head_dim != 0 &&
         layout.deepseek_indexer_q != kMissingOffset &&
         layout.deepseek_indexer_q_scale != kMissingOffset &&
         layout.deepseek_indexer_weights != kMissingOffset;
}

bool layout_deepseek_v4_mlp_supported(const SequenceLayerLayout &layout) {
  return layout.mlp_kind == kMlpKindDense ||
         layout.mlp_kind == kMlpKindSparseMoe;
}

uint64_t deepseek_v32_mla_kv_page_bytes(const SequenceLayerLayout &layout) {
  return layout_is_deepseek_v32_mla_packed(layout)
             ? sat_mul_u64(kDeepSeekV32PackedKvBlockTokens,
                           kDeepSeekV32PackedKvTokenBytes)
             : 0;
}

uint64_t deepseek_v32_mla_kv_layer_bytes(
    const SequenceLayerLayout &layout, uint32_t max_context_tokens) {
  const uint64_t page_bytes = deepseek_v32_mla_kv_page_bytes(layout);
  if (page_bytes == 0 || max_context_tokens == 0) {
    return 0;
  }
  const uint64_t blocks =
      (static_cast<uint64_t>(max_context_tokens) +
       kDeepSeekV32PackedKvBlockTokens - 1u) /
      kDeepSeekV32PackedKvBlockTokens;
  return sat_mul_u64(blocks, page_bytes);
}

uint64_t deepseek_v32_mla_kv_layer_offset_bytes(
    const NervaCudaHfDecodeSequenceSession *session,
    uint32_t target_layer_index) {
  if (session == nullptr) {
    return 0;
  }
  uint64_t offset = 0;
  const uint32_t capped_layer =
      std::min(target_layer_index,
               static_cast<uint32_t>(session->host_layouts.size()));
  for (uint32_t layer_index = 0; layer_index < capped_layer; ++layer_index) {
    offset = sat_add_u64(
        offset, deepseek_v32_mla_kv_layer_bytes(
                    session->host_layouts[layer_index],
                    session->max_context_tokens));
  }
  return offset;
}

uint32_t deepseek_v32_mla_kv_block_count(
    const NervaCudaHfDecodeSequenceSession *session,
    const SequenceLayerLayout &layout) {
  if (session == nullptr || !layout_is_deepseek_v32_mla_packed(layout)) {
    return 0;
  }
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(session->max_context_tokens) +
       kDeepSeekV32PackedKvBlockTokens - 1u) /
      kDeepSeekV32PackedKvBlockTokens);
}

uint64_t deepseek_v32_indexer_kv_token_bytes(
    const SequenceLayerLayout &layout) {
  if (!layout_is_deepseek_v32_indexer_native(layout)) {
    return 0;
  }
  const uint64_t scale_bytes =
      ((static_cast<uint64_t>(layout.deepseek_index_head_dim) + 127u) / 128u) *
      sizeof(float);
  return static_cast<uint64_t>(layout.deepseek_index_head_dim) + scale_bytes;
}

uint64_t deepseek_v32_indexer_kv_page_bytes(
    const SequenceLayerLayout &layout) {
  const uint64_t token_bytes = deepseek_v32_indexer_kv_token_bytes(layout);
  return token_bytes == 0
             ? 0
             : sat_mul_u64(kDeepSeekV32IndexerKvBlockTokens, token_bytes);
}

uint64_t deepseek_v32_indexer_kv_layer_bytes(
    const SequenceLayerLayout &layout, uint32_t max_context_tokens) {
  const uint64_t page_bytes = deepseek_v32_indexer_kv_page_bytes(layout);
  if (page_bytes == 0 || max_context_tokens == 0) {
    return 0;
  }
  const uint64_t blocks =
      (static_cast<uint64_t>(max_context_tokens) +
       kDeepSeekV32IndexerKvBlockTokens - 1u) /
      kDeepSeekV32IndexerKvBlockTokens;
  return sat_mul_u64(blocks, page_bytes);
}

uint64_t deepseek_v32_indexer_kv_layer_offset_bytes(
    const NervaCudaHfDecodeSequenceSession *session,
    uint32_t target_layer_index) {
  if (session == nullptr) {
    return 0;
  }
  uint64_t offset = 0;
  const uint32_t capped_layer =
      std::min(target_layer_index,
               static_cast<uint32_t>(session->host_layouts.size()));
  for (uint32_t layer_index = 0; layer_index < capped_layer; ++layer_index) {
    offset = sat_add_u64(
        offset, deepseek_v32_indexer_kv_layer_bytes(
                    session->host_layouts[layer_index],
                    session->max_context_tokens));
  }
  return offset;
}

uint32_t deepseek_v32_indexer_kv_block_count(
    const NervaCudaHfDecodeSequenceSession *session,
    const SequenceLayerLayout &layout) {
  if (session == nullptr || !layout_is_deepseek_v32_indexer_native(layout)) {
    return 0;
  }
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(session->max_context_tokens) +
       kDeepSeekV32IndexerKvBlockTokens - 1u) /
      kDeepSeekV32IndexerKvBlockTokens);
}

uint64_t deepseek_v32_indexer_query_state_q_scale_offset_bytes(
    const SequenceLayerLayout &layout) {
  const uint64_t query_bytes =
      static_cast<uint64_t>(layout.deepseek_index_n_heads) *
      layout.deepseek_index_head_dim;
  return deepseek_v4_round_up_u64(query_bytes, sizeof(float));
}

uint64_t deepseek_v32_indexer_query_state_weights_offset_bytes(
    const SequenceLayerLayout &layout) {
  return deepseek_v32_indexer_query_state_q_scale_offset_bytes(layout) +
         static_cast<uint64_t>(layout.deepseek_index_n_heads) * sizeof(float);
}

uint64_t deepseek_v32_indexer_query_state_token_bytes(
    const SequenceLayerLayout &layout) {
  if (!layout_is_deepseek_v32_indexer_query_native(layout)) {
    return 0;
  }
  return deepseek_v32_indexer_query_state_weights_offset_bytes(layout) +
         static_cast<uint64_t>(layout.deepseek_index_n_heads) * sizeof(float);
}

uint64_t deepseek_v32_indexer_query_state_layer_bytes(
    const SequenceLayerLayout &layout, uint32_t max_context_tokens) {
  const uint64_t token_bytes =
      deepseek_v32_indexer_query_state_token_bytes(layout);
  if (token_bytes == 0 || max_context_tokens == 0) {
    return 0;
  }
  return sat_mul_u64(max_context_tokens, token_bytes);
}

uint64_t deepseek_v32_indexer_query_state_layer_offset_bytes(
    const NervaCudaHfDecodeSequenceSession *session,
    uint32_t target_layer_index) {
  if (session == nullptr) {
    return 0;
  }
  uint64_t offset = 0;
  const uint32_t capped_layer =
      std::min(target_layer_index,
               static_cast<uint32_t>(session->host_layouts.size()));
  for (uint32_t layer_index = 0; layer_index < capped_layer; ++layer_index) {
    offset = sat_add_u64(
        offset, deepseek_v32_indexer_query_state_layer_bytes(
                    session->host_layouts[layer_index],
                    session->max_context_tokens));
  }
  return offset;
}

bool layout_is_deepseek_v4_swa_native(const SequenceLayerLayout &layout) {
  if (layout.attention_kind != kAttentionKindDeepSeekMla ||
      layout.deepseek_mode != kDeepSeekModeV4Swa) {
    return false;
  }
  return layout_deepseek_v4_mlp_supported(layout);
}

bool layout_is_deepseek_v4_compressed_native(const SequenceLayerLayout &layout) {
  if (layout.attention_kind != kAttentionKindDeepSeekMla ||
      (layout.deepseek_mode != kDeepSeekModeV4Compressed &&
       layout.deepseek_mode != kDeepSeekModeV4CompressedIndexer)) {
    return false;
  }
  return layout.deepseek_compress_ratio > 1 &&
         layout_deepseek_v4_mlp_supported(layout);
}

uint32_t layout_deepseek_v4_compressor_coff(
    const SequenceLayerLayout &layout) {
  return layout.deepseek_compress_ratio == 4 ? 2u : 1u;
}

uint64_t deepseek_v4_compressed_token_capacity(uint32_t max_context_tokens,
                                               uint32_t compress_ratio) {
  if (max_context_tokens == 0 || compress_ratio == 0) {
    return 0;
  }
  const uint64_t compressed_tokens =
      (static_cast<uint64_t>(max_context_tokens) + compress_ratio - 1u) /
      compress_ratio;
  const uint32_t block_tokens =
      deepseek_v4_packed_kv_block_tokens(compress_ratio);
  const uint64_t blocks =
      (compressed_tokens + block_tokens - 1u) / block_tokens;
  return blocks * block_tokens;
}

uint64_t deepseek_v4_main_compressed_kv_token_bytes(
    const SequenceLayerLayout &layout) {
  const uint64_t nope = layout.deepseek_qk_nope_head_dim;
  const uint64_t rope = layout.deepseek_qk_rope_head_dim;
  if (nope == 0 && rope == 0) {
    return 0;
  }
  const uint64_t token_stride = nope + rope * 2u;
  const uint64_t scale_dim = nope / 64u + 1u;
  return token_stride + scale_dim;
}

uint64_t deepseek_v4_main_compressed_kv_page_bytes(
    const SequenceLayerLayout &layout) {
  const uint64_t token_bytes =
      deepseek_v4_main_compressed_kv_token_bytes(layout);
  if (token_bytes == 0) {
    return 0;
  }
  const uint64_t real_page_bytes = sat_mul_u64(
      deepseek_v4_packed_kv_block_tokens(layout.deepseek_compress_ratio),
      token_bytes);
  return deepseek_v4_round_up_u64(real_page_bytes,
                                  kDeepSeekV4PackedKvAlignmentBytes);
}

uint64_t deepseek_v4_swa_kv_page_bytes(const SequenceLayerLayout &layout) {
  const uint64_t token_bytes =
      deepseek_v4_main_compressed_kv_token_bytes(layout);
  if (token_bytes == 0) {
    return 0;
  }
  const uint64_t real_page_bytes =
      sat_mul_u64(kDeepSeekV4PackedKvDefaultBlockTokens, token_bytes);
  return deepseek_v4_round_up_u64(real_page_bytes,
                                  kDeepSeekV4PackedKvAlignmentBytes);
}

uint64_t deepseek_v4_indexer_kv_token_bytes(
    const SequenceLayerLayout &layout) {
  if (layout.deepseek_index_head_dim == 0) {
    return 0;
  }
  const uint64_t scale_dim =
      ((layout.deepseek_index_head_dim + 127u) / 128u) * sizeof(float);
  return layout.deepseek_index_head_dim + scale_dim;
}

uint64_t deepseek_v4_indexer_kv_page_bytes(
    const SequenceLayerLayout &layout) {
  const uint64_t token_bytes = deepseek_v4_indexer_kv_token_bytes(layout);
  if (token_bytes == 0) {
    return 0;
  }
  const uint64_t real_page_bytes = sat_mul_u64(
      deepseek_v4_packed_kv_block_tokens(layout.deepseek_compress_ratio),
      token_bytes);
  return deepseek_v4_round_up_u64(real_page_bytes,
                                  kDeepSeekV4PackedKvAlignmentBytes);
}

uint64_t deepseek_v4_main_compressed_kv_layer_bytes(
    const SequenceLayerLayout &layout, uint32_t max_context_tokens) {
  if (!layout_is_deepseek_v4_compressed_native(layout)) {
    return 0;
  }
  const uint64_t compressed_capacity =
      deepseek_v4_compressed_token_capacity(max_context_tokens,
                                            layout.deepseek_compress_ratio);
  const uint64_t block_tokens =
      deepseek_v4_packed_kv_block_tokens(layout.deepseek_compress_ratio);
  const uint64_t blocks = block_tokens == 0 ? 0 : compressed_capacity / block_tokens;
  return sat_mul_u64(blocks, deepseek_v4_main_compressed_kv_page_bytes(layout));
}

uint64_t deepseek_v4_swa_kv_layer_bytes(
    const SequenceLayerLayout &layout, uint32_t max_context_tokens) {
  if (!layout_is_deepseek_v4_swa_native(layout) &&
      !layout_is_deepseek_v4_compressed_native(layout)) {
    return 0;
  }
  const uint64_t blocks =
      (static_cast<uint64_t>(max_context_tokens) +
       kDeepSeekV4PackedKvDefaultBlockTokens - 1u) /
      kDeepSeekV4PackedKvDefaultBlockTokens;
  return sat_mul_u64(blocks, deepseek_v4_swa_kv_page_bytes(layout));
}

uint64_t deepseek_v4_swa_kv_layer_offset_bytes(
    const NervaCudaHfDecodeSequenceSession *session,
    uint32_t target_layer_index) {
  if (session == nullptr) {
    return 0;
  }
  uint64_t offset = 0;
  const uint32_t capped_layer =
      std::min(target_layer_index,
               static_cast<uint32_t>(session->host_layouts.size()));
  for (uint32_t layer_index = 0; layer_index < capped_layer; ++layer_index) {
    offset = sat_add_u64(
        offset, deepseek_v4_swa_kv_layer_bytes(session->host_layouts[layer_index],
                                               session->max_context_tokens));
  }
  return offset;
}

uint32_t deepseek_v4_swa_kv_block_count(
    const NervaCudaHfDecodeSequenceSession *session,
    const SequenceLayerLayout &layout) {
  if (session == nullptr ||
      (!layout_is_deepseek_v4_swa_native(layout) &&
       !layout_is_deepseek_v4_compressed_native(layout))) {
    return 0;
  }
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(session->max_context_tokens) +
       kDeepSeekV4PackedKvDefaultBlockTokens - 1u) /
      kDeepSeekV4PackedKvDefaultBlockTokens);
}

uint64_t deepseek_v4_main_compressed_kv_layer_offset_bytes(
    const NervaCudaHfDecodeSequenceSession *session,
    uint32_t target_layer_index) {
  if (session == nullptr) {
    return 0;
  }
  uint64_t offset = 0;
  const uint32_t capped_layer =
      std::min(target_layer_index,
               static_cast<uint32_t>(session->host_layouts.size()));
  for (uint32_t layer_index = 0; layer_index < capped_layer; ++layer_index) {
    offset = sat_add_u64(
        offset, deepseek_v4_main_compressed_kv_layer_bytes(
                    session->host_layouts[layer_index],
                    session->max_context_tokens));
  }
  return offset;
}

uint32_t deepseek_v4_compressed_kv_block_count(
    const NervaCudaHfDecodeSequenceSession *session,
    const SequenceLayerLayout &layout) {
  if (session == nullptr || layout.deepseek_compress_ratio == 0) {
    return 0;
  }
  const uint64_t compressed_capacity =
      deepseek_v4_compressed_token_capacity(session->max_context_tokens,
                                            layout.deepseek_compress_ratio);
  const uint32_t block_tokens =
      deepseek_v4_packed_kv_block_tokens(layout.deepseek_compress_ratio);
  return block_tokens == 0
             ? 0
             : static_cast<uint32_t>(compressed_capacity / block_tokens);
}

uint64_t deepseek_v4_indexer_kv_layer_bytes(
    const SequenceLayerLayout &layout, uint32_t max_context_tokens) {
  if (layout.deepseek_mode != kDeepSeekModeV4CompressedIndexer) {
    return 0;
  }
  const uint64_t compressed_capacity =
      deepseek_v4_compressed_token_capacity(max_context_tokens,
                                            layout.deepseek_compress_ratio);
  const uint64_t block_tokens =
      deepseek_v4_packed_kv_block_tokens(layout.deepseek_compress_ratio);
  const uint64_t blocks = block_tokens == 0 ? 0 : compressed_capacity / block_tokens;
  return sat_mul_u64(blocks, deepseek_v4_indexer_kv_page_bytes(layout));
}

uint64_t deepseek_v4_indexer_kv_layer_offset_bytes(
    const NervaCudaHfDecodeSequenceSession *session,
    uint32_t target_layer_index) {
  if (session == nullptr) {
    return 0;
  }
  uint64_t offset = 0;
  const uint32_t capped_layer =
      std::min(target_layer_index,
               static_cast<uint32_t>(session->host_layouts.size()));
  for (uint32_t layer_index = 0; layer_index < capped_layer; ++layer_index) {
    offset = sat_add_u64(
        offset, deepseek_v4_indexer_kv_layer_bytes(
                    session->host_layouts[layer_index],
                    session->max_context_tokens));
  }
  return offset;
}

uint64_t deepseek_v4_compressor_state_layer_bytes(
    const SequenceLayerLayout &layout, uint32_t kv_token_capacity) {
  if (!layout_is_deepseek_v4_compressed_native(layout)) {
    return 0;
  }
  const uint64_t coff = layout_deepseek_v4_compressor_coff(layout);
  const uint64_t state_width =
      coff * (layout.deepseek_qk_nope_head_dim +
              layout.deepseek_qk_rope_head_dim);
  return sat_mul_u64(sat_mul_u64(kv_token_capacity, state_width * 2u),
                     sizeof(float));
}

uint64_t deepseek_v4_compressor_state_layer_offset_bytes(
    const NervaCudaHfDecodeSequenceSession *session,
    uint32_t target_layer_index) {
  if (session == nullptr) {
    return 0;
  }
  uint64_t offset = 0;
  const uint32_t capped_layer =
      std::min(target_layer_index,
               static_cast<uint32_t>(session->host_layouts.size()));
  for (uint32_t layer_index = 0; layer_index < capped_layer; ++layer_index) {
    offset = sat_add_u64(
        offset, deepseek_v4_compressor_state_layer_bytes(
                    session->host_layouts[layer_index],
                    session->kv_token_capacity));
  }
  return offset;
}

uint64_t deepseek_v4_indexer_state_layer_bytes(
    const SequenceLayerLayout &layout, uint32_t kv_token_capacity) {
  if (layout.deepseek_mode != kDeepSeekModeV4CompressedIndexer ||
      layout.deepseek_index_head_dim == 0) {
    return 0;
  }
  const uint64_t coff = layout_deepseek_v4_compressor_coff(layout);
  const uint64_t state_width = coff * layout.deepseek_index_head_dim;
  return sat_mul_u64(sat_mul_u64(kv_token_capacity, state_width * 2u),
                     sizeof(float));
}

uint64_t deepseek_v4_indexer_state_layer_offset_bytes(
    const NervaCudaHfDecodeSequenceSession *session,
    uint32_t target_layer_index) {
  if (session == nullptr) {
    return 0;
  }
  uint64_t offset = 0;
  const uint32_t capped_layer =
      std::min(target_layer_index,
               static_cast<uint32_t>(session->host_layouts.size()));
  for (uint32_t layer_index = 0; layer_index < capped_layer; ++layer_index) {
    offset = sat_add_u64(
        offset, deepseek_v4_indexer_state_layer_bytes(
                    session->host_layouts[layer_index],
                    session->kv_token_capacity));
  }
  return offset;
}

void accumulate_deepseek_v4_compressed_runtime_bytes(
    const std::vector<SequenceLayerLayout> &layouts, uint32_t max_context_tokens,
    uint32_t kv_token_capacity, uint64_t *swa_kv_bytes,
    uint64_t *compressor_state_bytes, uint64_t *compressed_kv_bytes,
    uint64_t *indexer_state_bytes, uint64_t *indexer_kv_bytes) {
  uint64_t swa_kv = 0;
  uint64_t main_state = 0;
  uint64_t main_kv = 0;
  uint64_t idx_state = 0;
  uint64_t idx_kv = 0;
  for (const SequenceLayerLayout &layout : layouts) {
    swa_kv = sat_add_u64(
        swa_kv, deepseek_v4_swa_kv_layer_bytes(layout, max_context_tokens));
    idx_state = sat_add_u64(
        idx_state, deepseek_v32_indexer_query_state_layer_bytes(
                       layout, max_context_tokens));
    idx_kv = sat_add_u64(
        idx_kv, deepseek_v32_indexer_kv_layer_bytes(layout, max_context_tokens));
    if (!layout_is_deepseek_v4_compressed_native(layout)) {
      continue;
    }
    main_state = sat_add_u64(
        main_state,
        deepseek_v4_compressor_state_layer_bytes(layout, kv_token_capacity));
    main_kv = sat_add_u64(
        main_kv,
        deepseek_v4_main_compressed_kv_layer_bytes(layout,
                                                   max_context_tokens));

    if (layout.deepseek_mode == kDeepSeekModeV4CompressedIndexer) {
      idx_state = sat_add_u64(
          idx_state,
          deepseek_v4_indexer_state_layer_bytes(layout, kv_token_capacity));
      idx_kv = sat_add_u64(
          idx_kv,
          deepseek_v4_indexer_kv_layer_bytes(layout, max_context_tokens));
    }
  }
  if (swa_kv_bytes != nullptr) *swa_kv_bytes = swa_kv;
  if (compressor_state_bytes != nullptr) *compressor_state_bytes = main_state;
  if (compressed_kv_bytes != nullptr) *compressed_kv_bytes = main_kv;
  if (indexer_state_bytes != nullptr) *indexer_state_bytes = idx_state;
  if (indexer_kv_bytes != nullptr) *indexer_kv_bytes = idx_kv;
}

uint64_t accumulate_deepseek_v32_mla_kv_bytes(
    const std::vector<SequenceLayerLayout> &layouts,
    uint32_t max_context_tokens) {
  uint64_t bytes = 0;
  for (const SequenceLayerLayout &layout : layouts) {
    bytes = sat_add_u64(
        bytes, deepseek_v32_mla_kv_layer_bytes(layout, max_context_tokens));
  }
  return bytes;
}

bool layout_is_deepseek_v4_native(const SequenceLayerLayout &layout) {
  return layout_is_deepseek_v4_swa_native(layout) ||
         layout_is_deepseek_v4_compressed_native(layout);
}

void accumulate_deepseek_v4_mhc_runtime_bytes(
    const std::vector<SequenceLayerLayout> &layouts, uint32_t max_context_tokens,
    uint32_t hidden, uint64_t *residual_bytes, uint64_t *post_mix_bytes,
    uint64_t *comb_mix_bytes) {
  uint32_t hc_mult = 0;
  for (const SequenceLayerLayout &layout : layouts) {
    if (!layout_is_deepseek_v4_native(layout)) {
      continue;
    }
    hc_mult = std::max(hc_mult, layout.deepseek_hc_mult);
  }

  uint64_t residual = 0;
  uint64_t post_mix = 0;
  uint64_t comb_mix = 0;
  if (hc_mult != 0 && hidden != 0 && max_context_tokens != 0) {
    const uint64_t tokens = max_context_tokens;
    const uint64_t hc = hc_mult;
    const uint64_t hc_hidden =
        sat_mul_u64(static_cast<uint64_t>(hidden), hc);
    residual = sat_mul_u64(sat_mul_u64(tokens, hc_hidden), sizeof(float));
    post_mix = sat_mul_u64(sat_mul_u64(tokens, hc), sizeof(float));
    comb_mix =
        sat_mul_u64(sat_mul_u64(tokens, sat_mul_u64(hc, hc)), sizeof(float));
  }
  if (residual_bytes != nullptr) *residual_bytes = residual;
  if (post_mix_bytes != nullptr) *post_mix_bytes = post_mix;
  if (comb_mix_bytes != nullptr) *comb_mix_bytes = comb_mix;
}

bool layout_is_native_deepseek_session(const SequenceLayerLayout &layout) {
  return layout_is_deepseek_v3_mla(layout) ||
         layout_is_deepseek_v4_native(layout);
}

uint32_t session_deepseek_v4_compressed_context_limit(
    const NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr ||
      session->host_layouts.size() != session->layer_count) {
    return 0;
  }
  uint32_t limit = UINT32_MAX;
  for (const SequenceLayerLayout &layout : session->host_layouts) {
    if (!layout_is_deepseek_v4_compressed_native(layout)) {
      continue;
    }
    limit = std::min(limit, layout.deepseek_compress_ratio);
  }
  return limit == UINT32_MAX ? 0 : limit;
}

cudaError_t validate_deepseek_v4_compressed_context(
    const NervaCudaHfDecodeSequenceSession *session, uint32_t context_steps) {
  (void)session;
  (void)context_steps;
  return cudaSuccess;
}

uint64_t layout_deepseek_v3_qk_head_dim(const SequenceLayerLayout &layout) {
  return sat_add_u64(layout.deepseek_qk_nope_head_dim,
                     layout.deepseek_qk_rope_head_dim);
}

uint64_t layout_deepseek_v3_q_rows(const SequenceLayerLayout &layout,
                                   uint64_t fallback_attention_hidden) {
  if (!layout_is_deepseek_v3_mla(layout)) return fallback_attention_hidden;
  const uint64_t qk_head_dim = layout_deepseek_v3_qk_head_dim(layout);
  if (qk_head_dim == 0) return fallback_attention_hidden;
  const uint64_t heads = fallback_attention_hidden / qk_head_dim;
  const uint64_t rows = sat_mul_u64(heads, qk_head_dim);
  return rows == 0 ? fallback_attention_hidden : rows;
}

uint64_t layout_deepseek_v3_kv_cache_width(const SequenceLayerLayout &layout,
                                           uint64_t fallback_kv_hidden) {
  if (!layout_is_deepseek_v3_mla(layout)) return fallback_kv_hidden;
  const uint64_t width = sat_add_u64(layout.deepseek_kv_lora_rank,
                                     layout.deepseek_qk_rope_head_dim);
  return width == 0 ? fallback_kv_hidden : width;
}

uint64_t layout_deepseek_kv_cache_width(const SequenceLayerLayout &layout,
                                        uint64_t fallback_kv_hidden) {
  if (layout_is_deepseek_v4_native(layout)) {
    return layout.deepseek_qk_nope_head_dim + layout.deepseek_qk_rope_head_dim;
  }
  return layout_deepseek_v3_kv_cache_width(layout, fallback_kv_hidden);
}

uint64_t layout_deepseek_v3_kv_b_rows(const SequenceLayerLayout &layout,
                                      uint64_t fallback_attention_hidden) {
  if (!layout_is_deepseek_v3_mla(layout)) return fallback_attention_hidden;
  const uint64_t qk_head_dim = layout_deepseek_v3_qk_head_dim(layout);
  if (qk_head_dim == 0) return fallback_attention_hidden;
  const uint64_t heads = fallback_attention_hidden / qk_head_dim;
  const uint64_t per_head = sat_add_u64(layout.deepseek_qk_nope_head_dim,
                                        layout.deepseek_v_head_dim);
  const uint64_t rows = sat_mul_u64(heads, per_head);
  return rows == 0 ? fallback_attention_hidden : rows;
}

uint64_t layout_deepseek_v3_value_rows(const SequenceLayerLayout &layout,
                                       uint64_t fallback_attention_hidden) {
  if (!layout_is_deepseek_v3_mla(layout)) return fallback_attention_hidden;
  const uint64_t qk_head_dim = layout_deepseek_v3_qk_head_dim(layout);
  if (qk_head_dim == 0) return fallback_attention_hidden;
  const uint64_t heads = fallback_attention_hidden / qk_head_dim;
  const uint64_t rows = sat_mul_u64(heads, layout.deepseek_v_head_dim);
  return rows == 0 ? fallback_attention_hidden : rows;
}

uint64_t layer_attention_workspace_rows(const SequenceLayerLayout &layout,
                                        uint64_t attention_hidden) {
  if (layout_is_deepseek_v4_native(layout)) {
    return std::max<uint64_t>(attention_hidden, layout.deepseek_q_lora_rank);
  }
  if (!layout_is_deepseek_v3_mla(layout)) return attention_hidden;
  uint64_t rows = layout_deepseek_v3_q_rows(layout, attention_hidden);
  rows = std::max(rows, layout_deepseek_v3_kv_b_rows(layout, attention_hidden));
  rows = std::max(rows, layout_deepseek_v3_value_rows(layout, attention_hidden));
  return std::max(rows, attention_hidden);
}

uint64_t max_attention_workspace_rows(
    const std::vector<SequenceLayerLayout> &layouts, uint64_t attention_hidden) {
  uint64_t rows = attention_hidden;
  for (const SequenceLayerLayout &layout : layouts) {
    rows = std::max(rows, layer_attention_workspace_rows(layout, attention_hidden));
  }
  return rows;
}

uint64_t max_kv_cache_width(const std::vector<SequenceLayerLayout> &layouts,
                            uint64_t kv_hidden) {
  uint64_t width = kv_hidden;
  for (const SequenceLayerLayout &layout : layouts) {
    width = std::max(width, layout_deepseek_kv_cache_width(layout, kv_hidden));
  }
  return width;
}

uint64_t layer_scratch_elements(const SequenceLayerLayout &layout,
                                uint64_t hidden,
                                uint64_t attention_hidden,
                                uint64_t kv_hidden,
                                uint64_t intermediate) {
  if (layout_is_native_deepseek_session(layout)) {
    uint64_t total = full_attention_scratch_elements(
        hidden, layer_attention_workspace_rows(layout, attention_hidden),
        layout_deepseek_kv_cache_width(layout, kv_hidden), intermediate);
    if (layout.mlp_kind == kMlpKindSparseMoe &&
        layout.experts_per_token > 1u) {
      const uint64_t extra_ranks = layout.experts_per_token - 1u;
      total = sat_add_u64(total, sat_mul_u64(extra_ranks, intermediate));
      total = sat_add_u64(total, sat_mul_u64(extra_ranks, hidden));
    }
    return total;
  }
  if (layout.attention_kind != kAttentionKindLinearGdn) {
    return full_attention_scratch_elements(hidden, attention_hidden, kv_hidden,
                                           intermediate);
  }
  const uint64_t conv_dim = layout_linear_gdn_conv_dim(layout);
  const uint64_t value_dim = layout_linear_gdn_value_dim(layout);
  uint64_t total = 0;
  total = sat_add_u64(total, sat_mul_u64(hidden, 5));
  total = sat_add_u64(total, sat_mul_u64(conv_dim, 2));
  total = sat_add_u64(total, sat_mul_u64(value_dim, 3));
  total = sat_add_u64(
      total, sat_mul_u64(static_cast<uint64_t>(layout.linear_value_heads), 2));
  total = sat_add_u64(total, sat_mul_u64(intermediate, 3));
  return total;
}

uint64_t max_layer_scratch_elements(
    const std::vector<SequenceLayerLayout> &layouts, uint64_t hidden,
    uint64_t attention_hidden, uint64_t kv_hidden, uint64_t intermediate) {
  uint64_t max_scratch = full_attention_scratch_elements(
      hidden, attention_hidden, kv_hidden, intermediate);
  for (const SequenceLayerLayout &layout : layouts) {
    max_scratch = std::max(
        max_scratch, layer_scratch_elements(layout, hidden, attention_hidden,
                                            kv_hidden, intermediate));
  }
  return max_scratch;
}
