void copy_deepseek_session_byte_fields(
    NervaCudaHfDecodeSequenceSession *session,
    const NervaCudaHfDecodeSequenceSession *source) {
  session->deepseek_v32_mla_kv_bytes = source->deepseek_v32_mla_kv_bytes;
  session->deepseek_swa_kv_bytes = source->deepseek_swa_kv_bytes;
  session->deepseek_compressor_state_bytes =
      source->deepseek_compressor_state_bytes;
  session->deepseek_compressed_kv_bytes =
      source->deepseek_compressed_kv_bytes;
  session->deepseek_indexer_state_bytes =
      source->deepseek_indexer_state_bytes;
  session->deepseek_indexer_kv_bytes = source->deepseek_indexer_kv_bytes;
  session->deepseek_mhc_residual_bytes =
      source->deepseek_mhc_residual_bytes;
  session->deepseek_mhc_post_mix_bytes =
      source->deepseek_mhc_post_mix_bytes;
  session->deepseek_mhc_comb_mix_bytes =
      source->deepseek_mhc_comb_mix_bytes;
  session->deepseek_sparse_topk_slots_bytes =
      source->deepseek_sparse_topk_slots_bytes;
  session->deepseek_sparse_topk_count_bytes =
      source->deepseek_sparse_topk_count_bytes;
  session->deepseek_sparse_topk_scores_bytes =
      source->deepseek_sparse_topk_scores_bytes;
  session->deepseek_runtime_counters_bytes =
      source->deepseek_runtime_counters_bytes;
}

cudaError_t allocate_deepseek_session_device_state(
    NervaCudaHfDecodeSequenceSession *session, int32_t *failure_stage) {
  cudaError_t err = cudaSuccess;
  if (session->deepseek_v32_mla_kv_bytes != 0) {
    if (failure_stage != nullptr) {
      *failure_stage = kCreateStageDeepSeekCompressedKvAlloc;
    }
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_deepseek_v32_mla_kv),
        session->deepseek_v32_mla_kv_bytes);
  }
  if (err == cudaSuccess && session->deepseek_swa_kv_bytes != 0) {
    if (failure_stage != nullptr) {
      *failure_stage = kCreateStageDeepSeekCompressedKvAlloc;
    }
    err = cudaMalloc(reinterpret_cast<void **>(&session->device_deepseek_swa_kv),
                     session->deepseek_swa_kv_bytes);
  }
  if (err == cudaSuccess && session->deepseek_compressor_state_bytes != 0) {
    if (failure_stage != nullptr) {
      *failure_stage = kCreateStageDeepSeekCompressedKvAlloc;
    }
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_deepseek_compressor_state),
        session->deepseek_compressor_state_bytes);
  }
  if (err == cudaSuccess && session->deepseek_compressed_kv_bytes != 0) {
    if (failure_stage != nullptr) {
      *failure_stage = kCreateStageDeepSeekCompressedKvAlloc;
    }
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_deepseek_compressed_kv),
        session->deepseek_compressed_kv_bytes);
  }
  if (err == cudaSuccess && session->deepseek_indexer_state_bytes != 0) {
    if (failure_stage != nullptr) {
      *failure_stage = kCreateStageDeepSeekCompressedKvAlloc;
    }
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_deepseek_indexer_state),
        session->deepseek_indexer_state_bytes);
  }
  if (err == cudaSuccess && session->deepseek_indexer_kv_bytes != 0) {
    if (failure_stage != nullptr) {
      *failure_stage = kCreateStageDeepSeekCompressedKvAlloc;
    }
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_deepseek_indexer_kv),
        session->deepseek_indexer_kv_bytes);
  }
  if (err == cudaSuccess && session->deepseek_mhc_residual_bytes != 0) {
    if (failure_stage != nullptr) {
      *failure_stage = kCreateStageDeepSeekCompressedKvAlloc;
    }
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_deepseek_mhc_residual),
        session->deepseek_mhc_residual_bytes);
  }
  if (err == cudaSuccess && session->deepseek_mhc_post_mix_bytes != 0) {
    if (failure_stage != nullptr) {
      *failure_stage = kCreateStageDeepSeekCompressedKvAlloc;
    }
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_deepseek_mhc_post_mix),
        session->deepseek_mhc_post_mix_bytes);
  }
  if (err == cudaSuccess && session->deepseek_mhc_comb_mix_bytes != 0) {
    if (failure_stage != nullptr) {
      *failure_stage = kCreateStageDeepSeekCompressedKvAlloc;
    }
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_deepseek_mhc_comb_mix),
        session->deepseek_mhc_comb_mix_bytes);
  }
  if (err == cudaSuccess && session->deepseek_sparse_topk_slots_bytes != 0) {
    if (failure_stage != nullptr) {
      *failure_stage = kCreateStageDeepSeekCompressedKvAlloc;
    }
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_deepseek_sparse_topk_slots),
        session->deepseek_sparse_topk_slots_bytes);
  }
  if (err == cudaSuccess && session->deepseek_sparse_topk_count_bytes != 0) {
    if (failure_stage != nullptr) {
      *failure_stage = kCreateStageDeepSeekCompressedKvAlloc;
    }
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_deepseek_sparse_topk_count),
        session->deepseek_sparse_topk_count_bytes);
  }
  if (err == cudaSuccess && session->deepseek_sparse_topk_scores_bytes != 0) {
    if (failure_stage != nullptr) {
      *failure_stage = kCreateStageDeepSeekCompressedKvAlloc;
    }
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_deepseek_sparse_topk_scores),
        session->deepseek_sparse_topk_scores_bytes);
  }
  if (err == cudaSuccess && session->deepseek_runtime_counters_bytes != 0) {
    if (failure_stage != nullptr) {
      *failure_stage = kCreateStageDeepSeekCompressedKvAlloc;
    }
    err = cudaMalloc(
        reinterpret_cast<void **>(&session->device_deepseek_runtime_counters),
        session->deepseek_runtime_counters_bytes);
  }
  return err;
}

cudaError_t reset_deepseek_session_device_state(
    NervaCudaHfDecodeSequenceSession *session) {
  cudaError_t err = cudaSuccess;
  if (session->deepseek_v32_mla_kv_bytes != 0) {
    err = cudaMemsetAsync(session->device_deepseek_v32_mla_kv, 0,
                          session->deepseek_v32_mla_kv_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->deepseek_compressor_state_bytes != 0) {
    err = cudaMemsetAsync(session->device_deepseek_compressor_state, 0,
                          session->deepseek_compressor_state_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->deepseek_swa_kv_bytes != 0) {
    err = cudaMemsetAsync(session->device_deepseek_swa_kv, 0,
                          session->deepseek_swa_kv_bytes, session->stream);
  }
  if (err == cudaSuccess && session->deepseek_compressed_kv_bytes != 0) {
    err = cudaMemsetAsync(session->device_deepseek_compressed_kv, 0,
                          session->deepseek_compressed_kv_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->deepseek_indexer_state_bytes != 0) {
    err = cudaMemsetAsync(session->device_deepseek_indexer_state, 0,
                          session->deepseek_indexer_state_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->deepseek_indexer_kv_bytes != 0) {
    err = cudaMemsetAsync(session->device_deepseek_indexer_kv, 0,
                          session->deepseek_indexer_kv_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->deepseek_mhc_residual_bytes != 0) {
    err = cudaMemsetAsync(session->device_deepseek_mhc_residual, 0,
                          session->deepseek_mhc_residual_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->deepseek_mhc_post_mix_bytes != 0) {
    err = cudaMemsetAsync(session->device_deepseek_mhc_post_mix, 0,
                          session->deepseek_mhc_post_mix_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->deepseek_mhc_comb_mix_bytes != 0) {
    err = cudaMemsetAsync(session->device_deepseek_mhc_comb_mix, 0,
                          session->deepseek_mhc_comb_mix_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->deepseek_sparse_topk_slots_bytes != 0) {
    err = cudaMemsetAsync(session->device_deepseek_sparse_topk_slots, 0,
                          session->deepseek_sparse_topk_slots_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->deepseek_sparse_topk_count_bytes != 0) {
    err = cudaMemsetAsync(session->device_deepseek_sparse_topk_count, 0,
                          session->deepseek_sparse_topk_count_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->deepseek_sparse_topk_scores_bytes != 0) {
    err = cudaMemsetAsync(session->device_deepseek_sparse_topk_scores, 0,
                          session->deepseek_sparse_topk_scores_bytes,
                          session->stream);
  }
  if (err == cudaSuccess && session->deepseek_runtime_counters_bytes != 0) {
    err = cudaMemsetAsync(session->device_deepseek_runtime_counters, 0,
                          session->deepseek_runtime_counters_bytes,
                          session->stream);
  }
  return err;
}

cudaError_t reset_deepseek_runtime_counters(
    NervaCudaHfDecodeSequenceSession *session) {
  if (session == nullptr || session->deepseek_runtime_counters_bytes == 0) {
    return cudaSuccess;
  }
  return cudaMemsetAsync(session->device_deepseek_runtime_counters, 0,
                         session->deepseek_runtime_counters_bytes,
                         session->stream);
}

cudaError_t clone_deepseek_session_device_state(
    NervaCudaHfDecodeSequenceSession *session,
    const NervaCudaHfDecodeSequenceSession *source) {
  cudaError_t err = cudaSuccess;
  if (session->deepseek_v32_mla_kv_bytes != 0) {
    err = cudaMemcpyAsync(session->device_deepseek_v32_mla_kv,
                          source->device_deepseek_v32_mla_kv,
                          session->deepseek_v32_mla_kv_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  if (err == cudaSuccess && session->deepseek_swa_kv_bytes != 0) {
    err = cudaMemcpyAsync(session->device_deepseek_swa_kv,
                          source->device_deepseek_swa_kv,
                          session->deepseek_swa_kv_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  if (err == cudaSuccess && session->deepseek_compressor_state_bytes != 0) {
    err = cudaMemcpyAsync(session->device_deepseek_compressor_state,
                          source->device_deepseek_compressor_state,
                          session->deepseek_compressor_state_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  if (err == cudaSuccess && session->deepseek_compressed_kv_bytes != 0) {
    err = cudaMemcpyAsync(session->device_deepseek_compressed_kv,
                          source->device_deepseek_compressed_kv,
                          session->deepseek_compressed_kv_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  if (err == cudaSuccess && session->deepseek_indexer_state_bytes != 0) {
    err = cudaMemcpyAsync(session->device_deepseek_indexer_state,
                          source->device_deepseek_indexer_state,
                          session->deepseek_indexer_state_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  if (err == cudaSuccess && session->deepseek_indexer_kv_bytes != 0) {
    err = cudaMemcpyAsync(session->device_deepseek_indexer_kv,
                          source->device_deepseek_indexer_kv,
                          session->deepseek_indexer_kv_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  if (err == cudaSuccess && session->deepseek_mhc_residual_bytes != 0) {
    err = cudaMemcpyAsync(session->device_deepseek_mhc_residual,
                          source->device_deepseek_mhc_residual,
                          session->deepseek_mhc_residual_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  if (err == cudaSuccess && session->deepseek_mhc_post_mix_bytes != 0) {
    err = cudaMemcpyAsync(session->device_deepseek_mhc_post_mix,
                          source->device_deepseek_mhc_post_mix,
                          session->deepseek_mhc_post_mix_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  if (err == cudaSuccess && session->deepseek_mhc_comb_mix_bytes != 0) {
    err = cudaMemcpyAsync(session->device_deepseek_mhc_comb_mix,
                          source->device_deepseek_mhc_comb_mix,
                          session->deepseek_mhc_comb_mix_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  if (err == cudaSuccess && session->deepseek_sparse_topk_slots_bytes != 0) {
    err = cudaMemcpyAsync(session->device_deepseek_sparse_topk_slots,
                          source->device_deepseek_sparse_topk_slots,
                          session->deepseek_sparse_topk_slots_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  if (err == cudaSuccess && session->deepseek_sparse_topk_count_bytes != 0) {
    err = cudaMemcpyAsync(session->device_deepseek_sparse_topk_count,
                          source->device_deepseek_sparse_topk_count,
                          session->deepseek_sparse_topk_count_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  if (err == cudaSuccess && session->deepseek_sparse_topk_scores_bytes != 0) {
    err = cudaMemcpyAsync(session->device_deepseek_sparse_topk_scores,
                          source->device_deepseek_sparse_topk_scores,
                          session->deepseek_sparse_topk_scores_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  if (err == cudaSuccess && session->deepseek_runtime_counters_bytes != 0) {
    err = cudaMemcpyAsync(session->device_deepseek_runtime_counters,
                          source->device_deepseek_runtime_counters,
                          session->deepseek_runtime_counters_bytes,
                          cudaMemcpyDeviceToDevice, session->stream);
  }
  return err;
}
