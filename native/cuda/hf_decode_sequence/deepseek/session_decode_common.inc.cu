uint32_t deepseek_norm_weight_dtype(const SequenceLayerLayout &layout) {
  return layout.deepseek_mode == kDeepSeekModeV32MlaIndexer ? kDTypeF32
                                                            : kDTypeBF16;
}

uint32_t layer_norm_weight_dtype(const SequenceLayerLayout &layout,
                                 uint32_t dtype) {
  if (layout_is_deepseek_v3_mla(layout)) {
    return deepseek_norm_weight_dtype(layout);
  }
  if (layout.attention_kind == kAttentionKindDeepSeekMla) {
    return kDTypeBF16;
  }
  return dtype;
}

const uint8_t *deepseek_fp8_ptr(uint16_t *arena, uint64_t offset) {
  return reinterpret_cast<const uint8_t *>(arena + offset);
}

const float *deepseek_scale_ptr(uint16_t *arena, uint64_t offset) {
  return reinterpret_cast<const float *>(arena + offset);
}

cudaError_t launch_deepseek_fp8_f32_scale_encoded_matvec_from_arena(
    cudaStream_t stream, uint16_t *arena, uint64_t weight_offset,
    uint64_t scale_offset, const uint16_t *input, uint32_t input_dtype,
    uint32_t rows, uint32_t cols, uint32_t block_rows, uint32_t block_cols,
    float *output) {
  if ((scale_offset & 1ull) == 0ull) {
    return launch_deepseek_fp8_f32_scale_encoded_matvec(
        stream, deepseek_fp8_ptr(arena, weight_offset),
        deepseek_scale_ptr(arena, scale_offset), input, input_dtype, rows,
        cols, block_rows, block_cols, output);
  }
  return launch_deepseek_fp8_f32_scale_slots_encoded_matvec(
      stream, deepseek_fp8_ptr(arena, weight_offset), arena + scale_offset,
      input, input_dtype, rows, cols, block_rows, block_cols, output);
}

cudaError_t launch_deepseek_fp8_f32_scale_dual_encoded_matvec_from_arena(
    cudaStream_t stream, uint16_t *arena, uint64_t weight_a_offset,
    uint64_t scale_a_offset, uint64_t weight_b_offset, uint64_t scale_b_offset,
    const uint16_t *input, uint32_t input_dtype, uint32_t rows, uint32_t cols,
    uint32_t block_rows, uint32_t block_cols, float *output_a,
    float *output_b) {
  const bool a_float_scale = (scale_a_offset & 1ull) == 0ull;
  const bool b_float_scale = (scale_b_offset & 1ull) == 0ull;
  if (a_float_scale && b_float_scale) {
    return launch_deepseek_fp8_f32_scale_dual_encoded_matvec(
        stream, deepseek_fp8_ptr(arena, weight_a_offset),
        deepseek_scale_ptr(arena, scale_a_offset),
        deepseek_fp8_ptr(arena, weight_b_offset),
        deepseek_scale_ptr(arena, scale_b_offset), input, input_dtype, rows,
        cols, block_rows, block_cols, output_a, output_b);
  }
  cudaError_t err = launch_deepseek_fp8_f32_scale_encoded_matvec_from_arena(
      stream, arena, weight_a_offset, scale_a_offset, input, input_dtype, rows,
      cols, block_rows, block_cols, output_a);
  if (err != cudaSuccess) {
    return err;
  }
  return launch_deepseek_fp8_f32_scale_encoded_matvec_from_arena(
      stream, arena, weight_b_offset, scale_b_offset, input, input_dtype, rows,
      cols, block_rows, block_cols, output_b);
}

uint64_t deepseek_fp8_slots_u64(uint64_t rows, uint64_t cols) {
  return (rows * cols + 1u) / 2u;
}

uint64_t deepseek_f32_scale_offset(uint64_t matrix_offset, uint64_t rows,
                                   uint64_t cols) {
  return matrix_offset + deepseek_fp8_slots_u64(rows, cols);
}

float deepseek_v4_layer_rope_theta(float session_rope_theta,
                                   const SequenceLayerLayout &layout) {
  if ((layout.deepseek_mode == kDeepSeekModeV4Compressed ||
       layout.deepseek_mode == kDeepSeekModeV4CompressedIndexer) &&
      layout.deepseek_compress_ratio > 1 &&
      isfinite(layout.deepseek_compress_rope_theta) &&
      layout.deepseek_compress_rope_theta > 0.0f) {
    return layout.deepseek_compress_rope_theta;
  }
  return session_rope_theta;
}

struct DeepseekDecodeProfileBuckets {
  uint64_t *qkv_projection_ns;
  uint64_t *attention_output_projection_ns;
  uint64_t *gate_up_projection_ns;
  uint64_t *down_projection_ns;
  uint64_t *attention_ns;
  uint64_t *mlp_ns;
  uint64_t *norm_ns;
};

cudaError_t deepseek_profile_begin_if(
    NervaCudaHfDecodeSequenceSession *session,
    const DeepseekDecodeProfileBuckets *profile) {
  return profile == nullptr ? cudaSuccess : profile_begin(session);
}

cudaError_t deepseek_profile_end_if(
    NervaCudaHfDecodeSequenceSession *session,
    const DeepseekDecodeProfileBuckets *profile,
    uint64_t *bucket) {
  return profile == nullptr ? cudaSuccess : profile_end(session, bucket);
}

bool deepseek_v4_aux_ready(const NervaCudaHfDecodeSequenceSession *session,
                           uint32_t stream_count, uint32_t event_count) {
  if (session == nullptr ||
      session->deepseek_v4_attention_aux_stream_count < stream_count ||
      session->deepseek_v4_attention_event_count < event_count) {
    return false;
  }
  for (uint32_t index = 0; index < stream_count; ++index) {
    if (session->deepseek_v4_attention_aux_streams[index] == nullptr) {
      return false;
    }
  }
  for (uint32_t index = 0; index < event_count; ++index) {
    if (session->deepseek_v4_attention_events[index] == nullptr) {
      return false;
    }
  }
  return true;
}

cudaError_t deepseek_v4_aux_fanout(
    NervaCudaHfDecodeSequenceSession *session, uint32_t stream_count) {
  cudaError_t err = cudaEventRecord(session->deepseek_v4_attention_events[0],
                                    session->stream);
  if (err != cudaSuccess) {
    return err;
  }
  for (uint32_t index = 0; index < stream_count; ++index) {
    err = cudaStreamWaitEvent(session->deepseek_v4_attention_aux_streams[index],
                              session->deepseek_v4_attention_events[0], 0);
    if (err != cudaSuccess) {
      return err;
    }
  }
  return cudaSuccess;
}

cudaError_t deepseek_v4_aux_join(
    NervaCudaHfDecodeSequenceSession *session, uint32_t stream_count) {
  for (uint32_t index = 0; index < stream_count; ++index) {
    cudaError_t err = cudaEventRecord(
        session->deepseek_v4_attention_events[index + 1u],
        session->deepseek_v4_attention_aux_streams[index]);
    if (err != cudaSuccess) {
      return err;
    }
    err = cudaStreamWaitEvent(session->stream,
                              session->deepseek_v4_attention_events[index + 1u],
                              0);
    if (err != cudaSuccess) {
      return err;
    }
  }
  return cudaSuccess;
}
