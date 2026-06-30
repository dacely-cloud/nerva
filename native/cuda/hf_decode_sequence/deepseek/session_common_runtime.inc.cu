bool session_has_deepseek_layers(
    const NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr ||
      session->host_layouts.size() != session->layer_count) {
    return false;
  }
  for (const SequenceLayerLayout &layout : session->host_layouts) {
    if (layout.attention_kind == kAttentionKindDeepSeekMla) {
      return true;
    }
  }
  return false;
}

bool session_has_deepseek_v4_native_layers(
    const NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr ||
      session->host_layouts.size() != session->layer_count) {
    return false;
  }
  for (const SequenceLayerLayout &layout : session->host_layouts) {
    if (layout_is_deepseek_v4_native(layout)) {
      return true;
    }
  }
  return false;
}

cudaError_t initialize_deepseek_v4_attention_aux_resources(
    NervaCudaHfDecodeSequenceSession *session) {
  if (!session_has_deepseek_v4_native_layers(session)) {
    return cudaSuccess;
  }
  for (uint32_t index = 0; index < kDeepSeekV4AttentionAuxStreamCount;
       ++index) {
    if (session->deepseek_v4_attention_aux_streams[index] != nullptr) {
      continue;
    }
    cudaError_t err = cudaStreamCreateWithFlags(
        &session->deepseek_v4_attention_aux_streams[index],
        cudaStreamNonBlocking);
    if (err != cudaSuccess) {
      return err;
    }
    session->deepseek_v4_attention_aux_stream_count += 1;
  }
  for (uint32_t index = 0; index < kDeepSeekV4AttentionEventCount; ++index) {
    if (session->deepseek_v4_attention_events[index] != nullptr) {
      continue;
    }
    cudaError_t err = cudaEventCreateWithFlags(
        &session->deepseek_v4_attention_events[index], cudaEventDisableTiming);
    if (err != cudaSuccess) {
      return err;
    }
    session->deepseek_v4_attention_event_count += 1;
  }
  return cudaSuccess;
}

bool session_has_only_native_deepseek_layers(
    const NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr ||
      session->host_layouts.size() != session->layer_count ||
      session->layer_count == 0) {
    return false;
  }
  bool has_deepseek = false;
  for (const SequenceLayerLayout &layout : session->host_layouts) {
    if (layout.attention_kind == kAttentionKindDeepSeekMla) {
      if (!layout_is_native_deepseek_session(layout)) {
        return false;
      }
      has_deepseek = true;
    }
  }
  return has_deepseek;
}
