#include "nerva_cuda_api.h"
#include "hf_decode_sequence/device_ops.cuh"
#include "hf_decode_sequence/kernels.cuh"
#include "hf_decode_sequence/projection.cuh"
#include "hf_decode_sequence/sampler.cuh"
#include "hf_decode_sequence/types.cuh"
#include "hf_decode_sequence/weights.cuh"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" int nerva_cuda_rt_candidate_selector_create(
    uint32_t pages, uint32_t page_tokens, uint32_t query_count,
    uint32_t candidates_per_query, uint32_t *candidate_pages, void *stream,
    void **selector_out, int32_t *cuda_error_out);
extern "C" int nerva_cuda_rt_candidate_selector_launch(
    void *selector, void *stream, uint32_t active_pages, uint32_t current_page,
    uint32_t local_pages, uint32_t sink_pages, int32_t *cuda_error_out);
extern "C" void nerva_cuda_rt_candidate_selector_destroy(void *selector);

#if NERVA_HAVE_CUDNN_FRONTEND
#include <cudnn.h>
#include <cudnn_frontend.h>
#endif

namespace {
#include "hf_decode_sequence/session_prelude.inc.cu"
}  // namespace

#include "hf_decode_sequence/session_state.cuh"

namespace {
#include "hf_decode_sequence/session_common.inc.cu"
#include "hf_decode_sequence/session_cudnn.inc.cu"
#include "hf_decode_sequence/session_profile.inc.cu"
#include "hf_decode_sequence/session_decode_prefill.inc.cu"
#include "hf_decode_sequence/session_graph_result.inc.cu"
}  // namespace

#include "hf_decode_sequence/session_api_oneshot.inc.cu"
#include "hf_decode_sequence/session_api_lifecycle.inc.cu"
#include "hf_decode_sequence/session_api_projection_batch.inc.cu"
#include "hf_decode_sequence/session_api_layer_projection_batch.inc.cu"
#include "hf_decode_sequence/session_api_batch_fork_destroy.inc.cu"
