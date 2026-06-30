bool deepseek_is_v4(uint32_t mode) {
  return mode == kDeepSeekModeV4Swa || mode == kDeepSeekModeV4Compressed ||
         mode == kDeepSeekModeV4CompressedIndexer;
}

bool deepseek_mode_valid(uint32_t mode) {
  return mode == kDeepSeekModeV3Mla || mode == kDeepSeekModeV32MlaIndexer ||
         deepseek_is_v4(mode);
}

bool deepseek_dims_valid(const NervaCudaHfDecodeChainLayer &layer) {
  if (!deepseek_mode_valid(layer.deepseek_mode) ||
      layer.deepseek_q_lora_rank == 0 ||
      layer.deepseek_qk_rope_head_dim == 0 ||
      layer.deepseek_v_head_dim == 0 ||
      layer.deepseek_compress_ratio == 0) {
    return false;
  }
  if (deepseek_is_v4(layer.deepseek_mode)) {
    if (layer.deepseek_hc_mult == 0 || layer.deepseek_o_lora_rank == 0 ||
        layer.deepseek_o_groups == 0 ||
        layer.deepseek_qk_nope_head_dim == 0) {
      return false;
    }
  } else if (layer.deepseek_kv_lora_rank == 0 ||
             layer.deepseek_qk_nope_head_dim == 0) {
    return false;
  }
  if ((layer.deepseek_flags & kDeepSeekFlagSparseIndexer) != 0 &&
      (layer.deepseek_index_n_heads == 0 ||
       layer.deepseek_index_head_dim == 0)) {
    return false;
  }
  return true;
}

bool has_deepseek_layers(const NervaCudaHfDecodeChainLayer *layers,
                         uint32_t layer_count) {
  if (layers == nullptr) {
    return false;
  }
  for (uint32_t index = 0; index < layer_count; ++index) {
    if (layers[index].attention_kind == kAttentionKindDeepSeekMla) {
      return true;
    }
  }
  return false;
}

bool has_unsupported_deepseek_layers(const NervaCudaHfDecodeChainLayer *layers,
                                     uint32_t layer_count) {
  if (layers == nullptr) {
    return false;
  }
  for (uint32_t index = 0; index < layer_count; ++index) {
    const NervaCudaHfDecodeChainLayer &layer = layers[index];
    if (layer.attention_kind == kAttentionKindDeepSeekMla) {
      const bool supported_v3 =
          layer.deepseek_mode == kDeepSeekModeV3Mla ||
          layer.deepseek_mode == kDeepSeekModeV32MlaIndexer;
      const bool supported_v4_swa =
          layer.deepseek_mode == kDeepSeekModeV4Swa &&
          (layer.mlp_kind == kMlpKindDense ||
           layer.mlp_kind == kMlpKindSparseMoe);
      const bool supported_v4_compressed =
          (layer.deepseek_mode == kDeepSeekModeV4Compressed ||
           layer.deepseek_mode == kDeepSeekModeV4CompressedIndexer) &&
          layer.deepseek_compress_ratio > 1 &&
          (layer.mlp_kind == kMlpKindDense ||
           layer.mlp_kind == kMlpKindSparseMoe);
      if (!supported_v3 && !supported_v4_swa && !supported_v4_compressed) {
        return true;
      }
    }
  }
  return false;
}
