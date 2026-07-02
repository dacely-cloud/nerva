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

// Launches the unified MLA attention family (query-latent absorption,
// flash-attention over the shared latent KV cache, per-head V projection)
// for `token_count` query positions. Decode calls this with token_count == 1
// and step_cursor == session->device_step; the batched prefill path calls it
// per sub-chunk with step_cursor == nullptr and positions derived from
// chunk_start + blockIdx. The same kernels run in both cases, so the
// per-(position, head) arithmetic is identical by construction.
cudaError_t launch_deepseek_mla_unified_attention(
    NervaCudaHfDecodeSequenceSession *session,
    const SequenceLayerLayout &layout, uint32_t layer_index,
    uint32_t *step_cursor, uint32_t max_steps, uint32_t chunk_start,
    uint32_t token_count, const float *q_tokens, uint32_t q_stride,
    uint16_t *attn_out, uint32_t attn_stride,
    const int32_t *sparse_topk_slots, uint32_t sparse_topk_stride,
    const uint32_t *sparse_topk_count, uint32_t record_sparse_attention) {
  if (token_count == 0 || token_count > kDeepSeekMlaAttentionSubChunkTokens ||
      session->device_deepseek_mla_q_latent == nullptr ||
      session->device_deepseek_mla_attn_latent == nullptr) {
    return cudaErrorInvalidValue;
  }
  const uint32_t kv_lora_rank = layout.deepseek_kv_lora_rank;
  const uint32_t qk_rope = layout.deepseek_qk_rope_head_dim;
  uint16_t *q_latent = session->device_deepseek_mla_q_latent;
  uint16_t *attn_latent = session->device_deepseek_mla_attn_latent;

  const uint32_t token_groups =
      ceil_div_u32(token_count, kDeepSeekMlaQLatentTokensPerBlock);
  const dim3 latent_grid(session->heads, token_groups);
  hf_deepseek_mla_q_latent_tokens_kernel<<<latent_grid, kDecodeThreads, 0,
                                           session->stream>>>(
      session->device_arena, layout, session->dtype, session->heads,
      step_cursor, max_steps, chunk_start, token_count, session->rope_theta,
      q_tokens, q_stride, q_latent);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  bool mma_path = kv_lora_rank == kDeepSeekMlaFaLora &&
                  qk_rope == kDeepSeekMlaFaRope &&
                  session->dtype == kDTypeBF16;
  if (mma_path) {
    static std::once_flag mla_fa_smem_once;
    static cudaError_t mla_fa_smem_status = cudaSuccess;
    std::call_once(mla_fa_smem_once, [] {
      mla_fa_smem_status = cudaFuncSetAttribute(
          hf_deepseek_mla_fa_tile_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(deepseek_mla_fa_smem_bytes()));
    });
    if (mla_fa_smem_status != cudaSuccess) {
      mma_path = false;
    }
  }
  if (mma_path) {
    const dim3 fa_grid(token_count,
                       ceil_div_u32(session->heads, kDeepSeekMlaFaHeadTile));
    hf_deepseek_mla_fa_tile_kernel<<<fa_grid, kDecodeThreads,
                                     deepseek_mla_fa_smem_bytes(),
                                     session->stream>>>(
        layout, layer_index, session->heads, step_cursor, max_steps,
        chunk_start, token_count, q_latent, session->device_kv_keys,
        session->kv_block_count, session->device_kv_block_table, attn_latent,
        sparse_topk_slots, sparse_topk_stride, sparse_topk_count,
        session->device_deepseek_runtime_counters);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  } else {
    const size_t generic_shared_bytes =
        (static_cast<size_t>(kv_lora_rank) * 2u + qk_rope) * sizeof(float);
    const dim3 fa_grid(token_count, session->heads);
    hf_deepseek_mla_fa_generic_kernel<<<fa_grid, kDecodeThreads,
                                        generic_shared_bytes,
                                        session->stream>>>(
        layout, layer_index, session->dtype, session->heads, step_cursor,
        max_steps, chunk_start, token_count, q_latent,
        session->device_kv_keys, session->kv_block_count,
        session->device_kv_block_table, attn_latent, sparse_topk_slots,
        sparse_topk_stride, sparse_topk_count,
        session->device_deepseek_runtime_counters);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  const dim3 vproj_grid(session->heads, token_groups);
  hf_deepseek_mla_v_proj_tokens_kernel<<<vproj_grid, kDecodeThreads, 0,
                                         session->stream>>>(
      session->device_arena, layout, session->dtype, session->heads,
      step_cursor, max_steps, chunk_start, token_count, attn_latent, attn_out,
      attn_stride, session->device_deepseek_runtime_counters,
      record_sparse_attention);
  return cudaGetLastError();
}
