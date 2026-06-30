extern "C" int nerva_cuda_hf_decode_sequence_deepseek_v4_swa_kv_snapshot(
    const NervaCudaHfDecodeSequenceDeepSeekV4SwaKvSnapshotRequest *request,
    NervaCudaHfDecodeSequenceDeepSeekV4SwaKvSnapshotResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->output_bytes == nullptr || request->output_byte_capacity == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  out->layer_index = request->layer_index;
  if (request->layer_index >= session->host_layouts.size() ||
      session->device_deepseek_swa_kv == nullptr ||
      session->deepseek_swa_kv_bytes == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const SequenceLayerLayout &layout =
      session->host_layouts[request->layer_index];
  const uint64_t layer_offset =
      deepseek_v4_swa_kv_layer_offset_bytes(session, request->layer_index);
  const uint64_t layer_bytes =
      deepseek_v4_swa_kv_layer_bytes(layout, session->max_context_tokens);
  const uint32_t block_count = deepseek_v4_swa_kv_block_count(session, layout);
  const uint64_t page_bytes = deepseek_v4_swa_kv_page_bytes(layout);
  if (layer_bytes == 0 || block_count == 0 ||
      layer_offset > session->deepseek_swa_kv_bytes ||
      layer_bytes > session->deepseek_swa_kv_bytes - layer_offset) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const uint64_t copy_bytes =
      std::min(layer_bytes, request->output_byte_capacity);
  cudaError_t err =
      cudaMemcpy(request->output_bytes,
                 session->device_deepseek_swa_kv + layer_offset, copy_bytes,
                 cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }
  uint64_t hash = kFnvOffset;
  for (uint64_t index = 0; index < copy_bytes; ++index) {
    hash ^= static_cast<uint64_t>(request->output_bytes[index]);
    hash *= kFnvPrime;
  }
  out->status = 0;
  out->layer_index = request->layer_index;
  out->block_count = block_count;
  out->layer_offset_bytes = layer_offset;
  out->layer_bytes = layer_bytes;
  out->page_bytes = page_bytes;
  out->copied_bytes = copy_bytes;
  out->output_hash = hash;
  return 0;
}

extern "C" int nerva_cuda_hf_decode_sequence_deepseek_v3_mla_kv_snapshot(
    const NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotRequest *request,
    NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->output_bytes == nullptr || request->output_byte_capacity == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  out->layer_index = request->layer_index;
  if (request->layer_index >= session->host_layouts.size() ||
      session->device_kv_keys == nullptr || session->kv_block_count == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const SequenceLayerLayout &layout =
      session->host_layouts[request->layer_index];
  if (!layout_is_deepseek_v3_mla(layout)) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const uint64_t kv_cache_width = layout_deepseek_v3_kv_cache_width(
      layout, static_cast<uint64_t>(session->kv_heads) * session->head_dim);
  if (kv_cache_width == 0 || kv_cache_width > UINT32_MAX) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const uint64_t layer_offset_elements =
      kv_cache_page_offset(request->layer_index, session->kv_block_count, 0, 0,
                           static_cast<uint32_t>(kv_cache_width), 0);
  const uint64_t layer_offset = layer_offset_elements * sizeof(uint16_t);
  const uint64_t page_bytes =
      static_cast<uint64_t>(kKvCacheBlockTokens) * kv_cache_width *
      sizeof(uint16_t);
  const uint64_t layer_bytes =
      static_cast<uint64_t>(session->kv_block_count) * page_bytes;
  const uint64_t kv_keys_bytes = session->kv_bytes / 2u;
  if (layer_bytes == 0 || layer_offset > kv_keys_bytes ||
      layer_bytes > kv_keys_bytes - layer_offset) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const uint64_t copy_bytes =
      std::min(layer_bytes, request->output_byte_capacity);
  cudaError_t err =
      cudaMemcpy(request->output_bytes,
                 reinterpret_cast<const uint8_t *>(session->device_kv_keys) +
                     layer_offset,
                 copy_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }
  uint64_t hash = kFnvOffset;
  for (uint64_t index = 0; index < copy_bytes; ++index) {
    hash ^= static_cast<uint64_t>(request->output_bytes[index]);
    hash *= kFnvPrime;
  }
  out->status = 0;
  out->layer_index = request->layer_index;
  out->block_count = session->kv_block_count;
  out->layer_offset_bytes = layer_offset;
  out->layer_bytes = layer_bytes;
  out->page_bytes = page_bytes;
  out->copied_bytes = copy_bytes;
  out->output_hash = hash;
  return 0;
}

extern "C" int
nerva_cuda_hf_decode_sequence_deepseek_v32_mla_packed_kv_snapshot(
    const NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotRequest *request,
    NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->output_bytes == nullptr || request->output_byte_capacity == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  out->layer_index = request->layer_index;
  if (request->layer_index >= session->host_layouts.size() ||
      session->device_deepseek_v32_mla_kv == nullptr ||
      session->deepseek_v32_mla_kv_bytes == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const SequenceLayerLayout &layout =
      session->host_layouts[request->layer_index];
  if (!layout_is_deepseek_v32_mla_packed(layout)) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const uint64_t layer_offset =
      deepseek_v32_mla_kv_layer_offset_bytes(session, request->layer_index);
  const uint64_t layer_bytes =
      deepseek_v32_mla_kv_layer_bytes(layout, session->max_context_tokens);
  const uint32_t block_count =
      deepseek_v32_mla_kv_block_count(session, layout);
  const uint64_t page_bytes = deepseek_v32_mla_kv_page_bytes(layout);
  if (layer_bytes == 0 || block_count == 0 ||
      layer_offset > session->deepseek_v32_mla_kv_bytes ||
      layer_bytes > session->deepseek_v32_mla_kv_bytes - layer_offset) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const uint64_t copy_bytes =
      std::min(layer_bytes, request->output_byte_capacity);
  cudaError_t err =
      cudaMemcpy(request->output_bytes,
                 session->device_deepseek_v32_mla_kv + layer_offset,
                 copy_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }
  uint64_t hash = kFnvOffset;
  for (uint64_t index = 0; index < copy_bytes; ++index) {
    hash ^= static_cast<uint64_t>(request->output_bytes[index]);
    hash *= kFnvPrime;
  }
  out->status = 0;
  out->layer_index = request->layer_index;
  out->block_count = block_count;
  out->layer_offset_bytes = layer_offset;
  out->layer_bytes = layer_bytes;
  out->page_bytes = page_bytes;
  out->copied_bytes = copy_bytes;
  out->output_hash = hash;
  return 0;
}

extern "C" int nerva_cuda_hf_decode_sequence_deepseek_v32_indexer_kv_snapshot(
    const NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotRequest *request,
    NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->output_bytes == nullptr || request->output_byte_capacity == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  out->layer_index = request->layer_index;
  if (request->layer_index >= session->host_layouts.size() ||
      session->device_deepseek_indexer_kv == nullptr ||
      session->deepseek_indexer_kv_bytes == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const SequenceLayerLayout &layout =
      session->host_layouts[request->layer_index];
  if (!layout_is_deepseek_v32_indexer_native(layout)) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const uint64_t layer_offset =
      deepseek_v32_indexer_kv_layer_offset_bytes(session,
                                                request->layer_index);
  const uint64_t layer_bytes =
      deepseek_v32_indexer_kv_layer_bytes(layout, session->max_context_tokens);
  const uint32_t block_count =
      deepseek_v32_indexer_kv_block_count(session, layout);
  const uint64_t page_bytes = deepseek_v32_indexer_kv_page_bytes(layout);
  if (layer_bytes == 0 || block_count == 0 ||
      layer_offset > session->deepseek_indexer_kv_bytes ||
      layer_bytes > session->deepseek_indexer_kv_bytes - layer_offset) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const uint64_t copy_bytes =
      std::min(layer_bytes, request->output_byte_capacity);
  cudaError_t err =
      cudaMemcpy(request->output_bytes,
                 session->device_deepseek_indexer_kv + layer_offset,
                 copy_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }
  uint64_t hash = kFnvOffset;
  for (uint64_t index = 0; index < copy_bytes; ++index) {
    hash ^= static_cast<uint64_t>(request->output_bytes[index]);
    hash *= kFnvPrime;
  }
  out->status = 0;
  out->layer_index = request->layer_index;
  out->block_count = block_count;
  out->layer_offset_bytes = layer_offset;
  out->layer_bytes = layer_bytes;
  out->page_bytes = page_bytes;
  out->copied_bytes = copy_bytes;
  out->output_hash = hash;
  return 0;
}

extern "C" int
nerva_cuda_hf_decode_sequence_deepseek_v32_indexer_query_state_snapshot(
    const NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotRequest *request,
    NervaCudaHfDecodeSequenceDeepSeekV3MlaKvSnapshotResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->output_bytes == nullptr || request->output_byte_capacity == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  out->layer_index = request->layer_index;
  if (request->layer_index >= session->host_layouts.size() ||
      session->device_deepseek_indexer_state == nullptr ||
      session->deepseek_indexer_state_bytes == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const SequenceLayerLayout &layout =
      session->host_layouts[request->layer_index];
  if (!layout_is_deepseek_v32_indexer_query_native(layout)) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const uint64_t layer_offset =
      deepseek_v32_indexer_query_state_layer_offset_bytes(
          session, request->layer_index);
  const uint64_t layer_bytes =
      deepseek_v32_indexer_query_state_layer_bytes(
          layout, session->max_context_tokens);
  const uint64_t token_bytes =
      deepseek_v32_indexer_query_state_token_bytes(layout);
  if (layer_bytes == 0 || token_bytes == 0 ||
      layer_offset > session->deepseek_indexer_state_bytes ||
      layer_bytes > session->deepseek_indexer_state_bytes - layer_offset) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const uint64_t copy_bytes =
      std::min(layer_bytes, request->output_byte_capacity);
  cudaError_t err = cudaMemcpy(
      request->output_bytes,
      reinterpret_cast<uint8_t *>(session->device_deepseek_indexer_state) +
          layer_offset,
      copy_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }
  uint64_t hash = kFnvOffset;
  for (uint64_t index = 0; index < copy_bytes; ++index) {
    hash ^= static_cast<uint64_t>(request->output_bytes[index]);
    hash *= kFnvPrime;
  }
  out->status = 0;
  out->layer_index = request->layer_index;
  out->block_count = session->max_context_tokens;
  out->layer_offset_bytes = layer_offset;
  out->layer_bytes = layer_bytes;
  out->page_bytes = token_bytes;
  out->copied_bytes = copy_bytes;
  out->output_hash = hash;
  return 0;
}

extern "C" int
nerva_cuda_hf_decode_sequence_deepseek_v4_compressed_kv_snapshot(
    const NervaCudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshotRequest
        *request,
    NervaCudaHfDecodeSequenceDeepSeekV4CompressedKvSnapshotResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->output_bytes == nullptr || request->output_byte_capacity == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  out->layer_index = request->layer_index;
  if (request->layer_index >= session->host_layouts.size() ||
      session->device_deepseek_compressed_kv == nullptr ||
      session->deepseek_compressed_kv_bytes == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const SequenceLayerLayout &layout =
      session->host_layouts[request->layer_index];
  const uint64_t layer_offset =
      deepseek_v4_main_compressed_kv_layer_offset_bytes(
          session, request->layer_index);
  const uint64_t layer_bytes =
      deepseek_v4_main_compressed_kv_layer_bytes(
          layout, session->max_context_tokens);
  const uint32_t block_count =
      deepseek_v4_compressed_kv_block_count(session, layout);
  const uint64_t page_bytes =
      deepseek_v4_main_compressed_kv_page_bytes(layout);
  if (layer_bytes == 0 || block_count == 0 ||
      layer_offset > session->deepseek_compressed_kv_bytes ||
      layer_bytes > session->deepseek_compressed_kv_bytes - layer_offset) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const uint64_t copy_bytes =
      std::min(layer_bytes, request->output_byte_capacity);
  cudaError_t err =
      cudaMemcpy(request->output_bytes,
                 session->device_deepseek_compressed_kv + layer_offset,
                 copy_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }
  uint64_t hash = kFnvOffset;
  for (uint64_t index = 0; index < copy_bytes; ++index) {
    hash ^= static_cast<uint64_t>(request->output_bytes[index]);
    hash *= kFnvPrime;
  }
  out->status = 0;
  out->layer_index = request->layer_index;
  out->block_count = block_count;
  out->layer_offset_bytes = layer_offset;
  out->layer_bytes = layer_bytes;
  out->page_bytes = page_bytes;
  out->copied_bytes = copy_bytes;
  out->output_hash = hash;
  return 0;
}

extern "C" int nerva_cuda_hf_decode_sequence_deepseek_v4_mhc_snapshot(
    const NervaCudaHfDecodeSequenceDeepSeekV4MhcSnapshotRequest *request,
    NervaCudaHfDecodeSequenceDeepSeekV4MhcSnapshotResult *out) {
  if (out == nullptr) {
    return -1;
  }
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request == nullptr || request->session == nullptr ||
      request->output_bytes == nullptr || request->output_byte_capacity == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  NervaCudaHfDecodeSequenceSession *session = request->session;
  out->state_kind = request->state_kind;
  out->token_index = request->token_index;
  if (request->token_index >= session->max_context_tokens) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }

  uint32_t hc_mult = 0;
  for (const SequenceLayerLayout &layout : session->host_layouts) {
    if (layout_is_deepseek_v4_native(layout)) {
      hc_mult = std::max(hc_mult, layout.deepseek_hc_mult);
    }
  }
  if (hc_mult == 0 || session->hidden == 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }

  const float *source = nullptr;
  uint64_t total_bytes = 0;
  uint64_t token_bytes = 0;
  switch (request->state_kind) {
    case NERVA_CUDA_DEEPSEEK_V4_MHC_STATE_RESIDUAL:
      source = session->device_deepseek_mhc_residual;
      total_bytes = session->deepseek_mhc_residual_bytes;
      token_bytes = static_cast<uint64_t>(hc_mult) * session->hidden *
                    sizeof(float);
      break;
    case NERVA_CUDA_DEEPSEEK_V4_MHC_STATE_POST_MIX:
      source = session->device_deepseek_mhc_post_mix;
      total_bytes = session->deepseek_mhc_post_mix_bytes;
      token_bytes = static_cast<uint64_t>(hc_mult) * sizeof(float);
      break;
    case NERVA_CUDA_DEEPSEEK_V4_MHC_STATE_COMB_MIX:
      source = session->device_deepseek_mhc_comb_mix;
      total_bytes = session->deepseek_mhc_comb_mix_bytes;
      token_bytes = static_cast<uint64_t>(hc_mult) * hc_mult * sizeof(float);
      break;
    default:
      out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
      return -1;
  }

  const uint64_t token_offset =
      static_cast<uint64_t>(request->token_index) * token_bytes;
  if (source == nullptr || total_bytes == 0 || token_bytes == 0 ||
      token_offset > total_bytes || token_bytes > total_bytes - token_offset) {
    out->cuda_error = static_cast<int32_t>(cudaErrorInvalidValue);
    return -1;
  }
  const uint64_t copy_bytes =
      std::min(token_bytes, request->output_byte_capacity);
  cudaError_t err =
      cudaMemcpy(request->output_bytes,
                 reinterpret_cast<const uint8_t *>(source) + token_offset,
                 copy_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    out->cuda_error = static_cast<int32_t>(err);
    return -1;
  }
  uint64_t hash = kFnvOffset;
  for (uint64_t index = 0; index < copy_bytes; ++index) {
    hash ^= static_cast<uint64_t>(request->output_bytes[index]);
    hash *= kFnvPrime;
  }
  out->status = 0;
  out->state_kind = request->state_kind;
  out->token_index = request->token_index;
  out->token_count = session->max_context_tokens;
  out->token_offset_bytes = token_offset;
  out->token_bytes = token_bytes;
  out->total_bytes = total_bytes;
  out->copied_bytes = copy_bytes;
  out->output_hash = hash;
  return 0;
}
