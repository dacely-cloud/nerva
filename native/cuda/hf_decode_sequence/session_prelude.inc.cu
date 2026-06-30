enum CreateFailureStage : int32_t {
  kCreateStageNone = 0,
  kCreateStageInvalidRequest = 1,
  kCreateStageGetDeviceCount = 2,
  kCreateStageSetDevice = 3,
  kCreateStageSessionAlloc = 4,
  kCreateStageHostWeightAlloc = 5,
  kCreateStageHostSlotsAlloc = 6,
  kCreateStageDeviceArenaAlloc = 7,
  kCreateStageDeviceLayoutsAlloc = 8,
  kCreateStageDeviceScratchAlloc = 9,
  kCreateStageProjectionInputAlloc = 10,
  kCreateStagePackedQkvAlloc = 11,
  kCreateStagePackedGateUpAlloc = 12,
  kCreateStageKvKeysAlloc = 13,
  kCreateStageKvValuesAlloc = 14,
  kCreateStagePromptTokensAlloc = 15,
  kCreateStageDeviceSlotsAlloc = 16,
  kCreateStageDeviceStepAlloc = 17,
  kCreateStageCublasWorkspaceAlloc = 18,
  kCreateStageStreamCreate = 19,
  kCreateStageCublasCreate = 20,
  kCreateStageCublasLtCreate = 21,
  kCreateStageCublasConfigure = 22,
  kCreateStageStartEventCreate = 23,
  kCreateStageStopEventCreate = 24,
  kCreateStageDescriptorCopy = 25,
  kCreateStageLayoutCopy = 26,
  kCreateStagePackReplicas = 27,
  kCreateStageWarmCublas = 28,
  kCreateStageSetupSynchronize = 29,
  kCreateStagePrefillHiddenAlloc = 30,
  kCreateStagePrefillChunkAlloc = 31,
  kCreateStageDecodeAttentionAlloc = 32,
  kCreateStageDecodeSdpaAlloc = 33,
  kCreateStageProjectionPlanAutotune = 34,
  kCreateStageExperimentalRtDecodeInit = 35,
  kCreateStageDeepSeekCompressedKvAlloc = 36,
};

struct SessionSharedWeights {
  uint16_t *device_arena = nullptr;
  SequenceLayerLayout *device_layouts = nullptr;
  uint16_t *device_qkv_packed = nullptr;
  uint16_t *device_gate_up_packed = nullptr;

  ~SessionSharedWeights() {
    cudaFree(device_gate_up_packed);
    cudaFree(device_qkv_packed);
    cudaFree(device_layouts);
    cudaFree(device_arena);
  }
};

constexpr uint32_t kProjectionBatchPlanReady = 0;
constexpr uint32_t kProjectionBatchPlanInvalidRequest = 1;
constexpr uint32_t kProjectionBatchPlanNoSessions = 2;
constexpr uint32_t kProjectionBatchPlanNoReadySessions = 3;
constexpr uint32_t kProjectionBatchPlanSharedWeightsUnproven = 4;
constexpr uint32_t kProjectionBatchPlanInsufficientCompatibleReady = 5;
constexpr uint32_t kProjectionBatchPlanUnsupportedProjection = 6;
constexpr uint32_t kProjectionBatchPlanInvalidLayer = 7;
constexpr uint32_t kProjectionBatchPlanInsufficientScratch = 8;
constexpr uint32_t kProjectionBatchKindQkv = 1;
constexpr uint32_t kProjectionBatchKindAttentionOutput = 2;
constexpr uint32_t kProjectionBatchKindGateUp = 3;
constexpr uint32_t kProjectionBatchKindDown = 4;
constexpr uint32_t kProjectionBatchKindLmHead = 5;
constexpr uint32_t kExperimentalRtModeAuto = 1;
constexpr uint32_t kExperimentalRtModeShadow = 2;
constexpr uint32_t kExperimentalRtModeSparse = 3;































int fail(NervaCudaHfDecodeSequenceResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  return -1;
}

#if NERVA_HAVE_CUDNN_FRONTEND
cudaError_t cudnn_to_cuda(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return cudaSuccess;
    case CUDNN_STATUS_ALLOC_FAILED:
      return cudaErrorMemoryAllocation;
    case CUDNN_STATUS_BAD_PARAM:
      return cudaErrorInvalidValue;
    case CUDNN_STATUS_NOT_SUPPORTED:
      return cudaErrorNotSupported;
    case CUDNN_STATUS_EXECUTION_FAILED:
      return cudaErrorLaunchFailure;
    default:
      return cudaErrorUnknown;
  }
}
#endif

cudaError_t final_head_gemv(cublasHandle_t handle, uint16_t *arena,
                            SequenceArenaLayout arena_layout, uint32_t dtype,
                            uint32_t hidden, uint32_t vocab_size,
                            float *device_logits) {
  return encoded_row_major_gemv(handle, arena + arena_layout.lm_head,
                                arena + arena_layout.input, vocab_size, hidden,
                                dtype, device_logits);
}

cudaError_t warm_cublas_gemv(cublasHandle_t handle, uint16_t *arena,
                             SequenceArenaLayout arena_layout, uint32_t dtype,
                             float *scratch, cudaStream_t stream) {
  cudaError_t err = cudaMemsetAsync(arena + arena_layout.input, 0,
                                    sizeof(uint16_t), stream);
  if (err != cudaSuccess) {
    return err;
  }
  return encoded_row_major_gemv(handle, arena + arena_layout.lm_head,
                                arena + arena_layout.input, 1, 1, dtype,
                                scratch);
}

uint32_t observed_count_for(uint32_t steps, uint32_t prompt_token_count,
                            uint32_t has_eos_token, uint32_t eos_token,
                            const NervaCudaSyntheticTokenSlot *slots) {
  uint32_t count = steps;
  if (has_eos_token == 0) {
    return count;
  }
  const uint32_t output_start = prompt_token_count - 1u;
  for (uint32_t index = 0; index < steps; ++index) {
    if (slots[output_start + index].token == eos_token) {
      count = index + 1;
      break;
    }
  }
  return count;
}

uint32_t observed_count(const NervaCudaHfDecodeSequenceRequest *request,
                        const NervaCudaSyntheticTokenSlot *slots) {
  return observed_count_for(request->steps, request->prompt_token_count,
                            request->has_eos_token, request->eos_token, slots);
}

uint64_t saturating_mul_profile_value(uint64_t value, uint64_t scale) {
  if (value != 0 && scale > UINT64_MAX / value) {
    return UINT64_MAX;
  }
  return value * scale;
}

void scale_profile_counters(NervaCudaHfDecodeSequenceResult *out,
                            uint64_t scale) {
  if (out == nullptr || scale <= 1) {
    return;
  }
  out->projection_ns = saturating_mul_profile_value(out->projection_ns, scale);
  out->qkv_projection_ns =
      saturating_mul_profile_value(out->qkv_projection_ns, scale);
  out->attention_output_projection_ns =
      saturating_mul_profile_value(out->attention_output_projection_ns, scale);
  out->gate_up_projection_ns =
      saturating_mul_profile_value(out->gate_up_projection_ns, scale);
  out->down_projection_ns =
      saturating_mul_profile_value(out->down_projection_ns, scale);
  out->lm_head_projection_ns =
      saturating_mul_profile_value(out->lm_head_projection_ns, scale);
  out->attention_ns = saturating_mul_profile_value(out->attention_ns, scale);
  out->mlp_ns = saturating_mul_profile_value(out->mlp_ns, scale);
  out->norm_ns = saturating_mul_profile_value(out->norm_ns, scale);
  out->sampling_ns = saturating_mul_profile_value(out->sampling_ns, scale);
}

#if NERVA_HAVE_CUDNN_FRONTEND
struct CudnnPrefillSdpaPlan {
  std::unique_ptr<cudnn_frontend::graph::Graph> graph;
  uint32_t seq_tokens = 0;
  uint32_t heads = 0;
  uint32_t kv_heads = 0;
  uint32_t head_dim = 0;
  uint64_t rows = 0;
  size_t workspace_bytes = 0;
};

struct CudnnDecodeSdpaPlan {
  std::unique_ptr<cudnn_frontend::graph::Graph> graph;
  uint32_t max_context_tokens = 0;
  uint32_t kv_token_capacity = 0;
  uint32_t heads = 0;
  uint32_t kv_heads = 0;
  uint32_t head_dim = 0;
  size_t workspace_bytes = 0;
};
#endif
