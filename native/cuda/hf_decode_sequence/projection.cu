#include "projection.cuh"

#include <climits>
#include <stdint.h>

namespace {

constexpr uint32_t kDTypeBF16 = 1;
constexpr uint32_t kLtGemvMaxHeuristics = 32;
constexpr uint32_t kLtGemvAutotuneWarmups = 1;
constexpr uint32_t kLtGemvAutotuneIterations = 3;

}  // namespace

cudaError_t cublas_to_cuda(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return cudaSuccess;
    case CUBLAS_STATUS_ALLOC_FAILED:
      return cudaErrorMemoryAllocation;
    case CUBLAS_STATUS_INVALID_VALUE:
      return cudaErrorInvalidValue;
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return cudaErrorInvalidDeviceFunction;
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return cudaErrorLaunchFailure;
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return cudaErrorNotSupported;
    default:
      return cudaErrorUnknown;
  }
}

cudaDataType_t encoded_cuda_type(uint32_t dtype) {
  return dtype == kDTypeBF16 ? CUDA_R_16BF : CUDA_R_16F;
}

cudaError_t configure_cublas(cublasHandle_t handle, cudaStream_t stream,
                             void *workspace, size_t workspace_bytes) {
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_to_cuda(status);
  }
  status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_to_cuda(status);
  }
  status = cublasSetWorkspace(handle, workspace, workspace_bytes);
  return cublas_to_cuda(status);
}

cudaError_t encoded_row_major_gemv_beta(
    cublasHandle_t handle, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t dtype, float beta,
    float *output) {
  if (rows == 0 || cols == 0 || rows > INT32_MAX || cols > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  const float alpha = 1.0f;
  const cudaDataType_t data_type = encoded_cuda_type(dtype);
  cublasStatus_t status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rows), 1,
      static_cast<int>(cols), &alpha, matrix, data_type, static_cast<int>(cols),
      input, data_type, static_cast<int>(cols), &beta, output, CUDA_R_32F,
      static_cast<int>(rows), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return cublas_to_cuda(status);
}

cudaError_t encoded_row_major_gemv(cublasHandle_t handle, const uint16_t *matrix,
                                   const uint16_t *input, uint32_t rows,
                                   uint32_t cols, uint32_t dtype,
                                   float *output) {
  return encoded_row_major_gemv_beta(handle, matrix, input, rows, cols, dtype,
                                     0.0f, output);
}

cudaError_t encoded_row_major_gemv_strided_batched(
    cublasHandle_t handle, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t tokens, uint32_t dtype, float beta,
    float *output) {
  if (rows == 0 || cols == 0 || tokens == 0 || rows > INT32_MAX ||
      cols > INT32_MAX || tokens > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  const float alpha = 1.0f;
  const cudaDataType_t data_type = encoded_cuda_type(dtype);
  const cublasStatus_t status = cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rows), 1,
      static_cast<int>(cols), &alpha, matrix, data_type,
      static_cast<int>(cols), 0, input, data_type, static_cast<int>(cols),
      static_cast<long long>(cols), &beta, output, CUDA_R_32F,
      static_cast<int>(rows), static_cast<long long>(rows),
      static_cast<int>(tokens), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return cublas_to_cuda(status);
}

void destroy_lt_descriptors(cublasLtMatmulDesc_t op_desc,
                            cublasLtMatrixLayout_t a_desc,
                            cublasLtMatrixLayout_t b_desc,
                            cublasLtMatrixLayout_t c_desc,
                            cublasLtMatrixLayout_t d_desc) {
  if (d_desc != nullptr) cublasLtMatrixLayoutDestroy(d_desc);
  if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
  if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
  if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
  if (op_desc != nullptr) cublasLtMatmulDescDestroy(op_desc);
}

void destroy_lt_gemv_plan(LtGemvPlan *plan) {
  if (plan == nullptr) {
    return;
  }
  destroy_lt_descriptors(plan->op_desc, plan->a_desc, plan->b_desc,
                         plan->c_desc, plan->d_desc);
  *plan = LtGemvPlan{};
}

void destroy_lt_gemm_tokens_plan(LtGemmTokensPlan *plan) {
  if (plan == nullptr) {
    return;
  }
  destroy_lt_descriptors(plan->op_desc, plan->a_desc, plan->b_desc,
                         plan->c_desc, plan->d_desc);
  *plan = LtGemmTokensPlan{};
}

cudaError_t create_lt_gemv_descriptors(
    uint32_t rows, uint32_t cols, uint32_t dtype,
    cublasLtMatmulDesc_t *op_desc, cublasLtMatrixLayout_t *a_desc,
    cublasLtMatrixLayout_t *b_desc, cublasLtMatrixLayout_t *c_desc,
    cublasLtMatrixLayout_t *d_desc) {
  if (rows == 0 || cols == 0 || op_desc == nullptr || a_desc == nullptr ||
      b_desc == nullptr || c_desc == nullptr || d_desc == nullptr) {
    return cudaErrorInvalidValue;
  }
  const cudaDataType_t data_type = encoded_cuda_type(dtype);
  cublasStatus_t status =
      cublasLtMatmulDescCreate(op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t op_a = CUBLAS_OP_N;
  cublasOperation_t op_b = CUBLAS_OP_T;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        *op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        *op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(a_desc, data_type, 1, cols, cols);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(b_desc, data_type, rows, cols, cols);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(c_desc, CUDA_R_32F, 1, rows, rows);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(d_desc, CUDA_R_32F, 1, rows, rows);
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        *a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        *b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        *c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        *d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_lt_descriptors(*op_desc, *a_desc, *b_desc, *c_desc, *d_desc);
    *op_desc = nullptr;
    *a_desc = nullptr;
    *b_desc = nullptr;
    *c_desc = nullptr;
    *d_desc = nullptr;
  }
  return cublas_to_cuda(status);
}

cudaError_t create_lt_gemm_tokens_plan(LtGemmTokensPlan *plan, uint32_t rows,
                                       uint32_t cols, uint32_t tokens,
                                       uint32_t dtype) {
  if (plan == nullptr || rows == 0 || cols == 0 || tokens == 0 ||
      rows > INT32_MAX || cols > INT32_MAX || tokens > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  destroy_lt_gemm_tokens_plan(plan);
  const cudaDataType_t data_type = encoded_cuda_type(dtype);
  cublasStatus_t status =
      cublasLtMatmulDescCreate(&plan->op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t op_a = CUBLAS_OP_N;
  cublasOperation_t op_b = CUBLAS_OP_T;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(
        &plan->a_desc, data_type, tokens, cols, cols);
  if (status == CUBLAS_STATUS_SUCCESS)
    status =
        cublasLtMatrixLayoutCreate(&plan->b_desc, data_type, rows, cols, cols);
  if (status == CUBLAS_STATUS_SUCCESS)
    status =
        cublasLtMatrixLayoutCreate(&plan->c_desc, CUDA_R_32F, tokens, rows, rows);
  if (status == CUBLAS_STATUS_SUCCESS)
    status =
        cublasLtMatrixLayoutCreate(&plan->d_desc, CUDA_R_32F, tokens, rows, rows);
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        plan->a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        plan->b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        plan->c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        plan->d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_lt_gemm_tokens_plan(plan);
    return cublas_to_cuda(status);
  }
  plan->rows = rows;
  plan->cols = cols;
  plan->tokens = tokens;
  plan->dtype = dtype;
  plan->ready = true;
  return cudaSuccess;
}

cudaError_t launch_lt_gemm_tokens_plan(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const LtGemmTokensPlan *plan,
    const uint16_t *matrix, const uint16_t *input, float beta, float *output) {
  if (handle == nullptr || plan == nullptr || !plan->ready ||
      matrix == nullptr || input == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  const float alpha = 1.0f;
  const cublasStatus_t status = cublasLtMatmul(
      handle, plan->op_desc, &alpha, input, plan->a_desc, matrix, plan->b_desc,
      &beta, output, plan->c_desc, output, plan->d_desc, nullptr, workspace,
      workspace_bytes, stream);
  return cublas_to_cuda(status);
}

cudaError_t create_lt_gemv_plan(LtGemvPlan *plan, uint32_t rows,
                                uint32_t cols, uint32_t dtype) {
  if (plan == nullptr) {
    return cudaErrorInvalidValue;
  }
  destroy_lt_gemv_plan(plan);
  cudaError_t err = create_lt_gemv_descriptors(
      rows, cols, dtype, &plan->op_desc, &plan->a_desc, &plan->b_desc,
      &plan->c_desc, &plan->d_desc);
  if (err != cudaSuccess) {
    return err;
  }
  plan->rows = rows;
  plan->cols = cols;
  plan->dtype = dtype;
  plan->ready = true;
  return cudaSuccess;
}

const cublasLtMatmulAlgo_t *lt_gemv_algo(const LtGemvPlan *plan) {
  return plan != nullptr && plan->has_algo ? &plan->algo : nullptr;
}

cudaError_t launch_lt_gemv_plan(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const LtGemvPlan *plan, const uint16_t *matrix,
    const uint16_t *input, float beta, float *output,
    const cublasLtMatmulAlgo_t *algo) {
  if (handle == nullptr || plan == nullptr || !plan->ready ||
      matrix == nullptr || input == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  const float alpha = 1.0f;
  const cublasStatus_t status = cublasLtMatmul(
      handle, plan->op_desc, &alpha, input, plan->a_desc, matrix, plan->b_desc,
      &beta, output, plan->c_desc, output, plan->d_desc, algo, workspace,
      workspace_bytes, stream);
  return cublas_to_cuda(status);
}

cudaError_t encoded_row_major_gemv_lt(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t dtype, float beta, float *output) {
  if (handle == nullptr || rows == 0 || cols == 0) {
    return cudaErrorInvalidValue;
  }
  const float alpha = 1.0f;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cudaError_t err = create_lt_gemv_descriptors(
      rows, cols, dtype, &op_desc, &a_desc, &b_desc, &c_desc, &d_desc);
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  if (err == cudaSuccess)
    status = cublasLtMatmul(handle, op_desc, &alpha, input, a_desc, matrix,
                            b_desc, &beta, output, c_desc, output, d_desc,
                            nullptr, workspace, workspace_bytes, stream);
  destroy_lt_descriptors(op_desc, a_desc, b_desc, c_desc, d_desc);
  return err == cudaSuccess ? cublas_to_cuda(status) : err;
}

cudaError_t encoded_row_major_gemv_lt_planned(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const LtGemvPlan *plan, const uint16_t *matrix,
    const uint16_t *input, float beta, float *output) {
  if (plan == nullptr || !plan->ready) {
    return cudaErrorInvalidValue;
  }
  return launch_lt_gemv_plan(handle, stream, workspace, workspace_bytes, plan,
                             matrix, input, beta, output, lt_gemv_algo(plan));
}

cudaError_t encoded_row_major_gemv_planned(
    cublasHandle_t cublas, cublasLtHandle_t cublas_lt, cudaStream_t stream,
    void *workspace, size_t workspace_bytes, const LtGemvPlan *plan,
    const uint16_t *matrix, const uint16_t *input, float beta, float *output) {
  if (plan == nullptr || !plan->ready) {
    return cudaErrorInvalidValue;
  }
  if (plan->backend == kGemvBackendCublas) {
    return encoded_row_major_gemv_beta(cublas, matrix, input, plan->rows,
                                       plan->cols, plan->dtype, beta, output);
  }
  return encoded_row_major_gemv_lt_planned(cublas_lt, stream, workspace,
                                           workspace_bytes, plan, matrix,
                                           input, beta, output);
}

cudaError_t find_lt_gemv_heuristics(
    cublasLtHandle_t handle, const LtGemvPlan *plan,
    size_t workspace_bytes, cublasLtMatmulHeuristicResult_t *heuristics,
    uint32_t *heuristic_count) {
  if (handle == nullptr || plan == nullptr || !plan->ready ||
      heuristics == nullptr || heuristic_count == nullptr) {
    return cudaErrorInvalidValue;
  }
  *heuristic_count = 0;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasStatus_t status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_to_cuda(status);
  }
  status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes));
  int returned_count = 0;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoGetHeuristic(
        handle, plan->op_desc, plan->a_desc, plan->b_desc, plan->c_desc,
        plan->d_desc, preference, kLtGemvMaxHeuristics, heuristics,
        &returned_count);
  }
  cublasLtMatmulPreferenceDestroy(preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_to_cuda(status);
  }
  *heuristic_count = returned_count > 0
                         ? static_cast<uint32_t>(returned_count)
                         : 0;
  return cudaSuccess;
}

uint64_t cuda_event_elapsed_ns(cudaEvent_t start, cudaEvent_t stop) {
  float elapsed_ms = 0.0f;
  cudaError_t err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (err != cudaSuccess || elapsed_ms <= 0.0f) {
    return 0;
  }
  const uint64_t ns = static_cast<uint64_t>(elapsed_ms * 1000000.0f);
  return ns == 0 ? 1 : ns;
}

cudaError_t time_lt_gemv_candidate(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const LtGemvPlan *plan, const uint16_t *matrix,
    const uint16_t *input, float *output, const cublasLtMatmulAlgo_t *algo,
    uint64_t *avg_ns) {
  if (avg_ns == nullptr) {
    return cudaErrorInvalidValue;
  }
  *avg_ns = 0;
  for (uint32_t index = 0; index < kLtGemvAutotuneWarmups; ++index) {
    cudaError_t err = launch_lt_gemv_plan(
        handle, stream, workspace, workspace_bytes, plan, matrix, input, 0.0f,
        output, algo);
    if (err != cudaSuccess) return err;
  }
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return err;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  err = cudaEventCreate(&start);
  if (err == cudaSuccess) err = cudaEventCreate(&stop);
  if (err == cudaSuccess) err = cudaEventRecord(start, stream);
  for (uint32_t index = 0;
       err == cudaSuccess && index < kLtGemvAutotuneIterations; ++index) {
    err = launch_lt_gemv_plan(handle, stream, workspace, workspace_bytes, plan,
                              matrix, input, 0.0f, output, algo);
  }
  if (err == cudaSuccess) err = cudaEventRecord(stop, stream);
  if (err == cudaSuccess) err = cudaEventSynchronize(stop);
  if (err == cudaSuccess) {
    const uint64_t total_ns = cuda_event_elapsed_ns(start, stop);
    *avg_ns = total_ns / kLtGemvAutotuneIterations;
  }
  if (stop != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(stop);
    if (err == cudaSuccess) err = cleanup_err;
  }
  if (start != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(start);
    if (err == cudaSuccess) err = cleanup_err;
  }
  return err;
}

cudaError_t time_cublas_gemv_candidate(
    cublasHandle_t handle, cudaStream_t stream, const LtGemvPlan *plan,
    const uint16_t *matrix, const uint16_t *input, float *output,
    uint64_t *avg_ns) {
  if (avg_ns == nullptr || plan == nullptr || !plan->ready) {
    return cudaErrorInvalidValue;
  }
  *avg_ns = 0;
  for (uint32_t index = 0; index < kLtGemvAutotuneWarmups; ++index) {
    cudaError_t err = encoded_row_major_gemv_beta(
        handle, matrix, input, plan->rows, plan->cols, plan->dtype, 0.0f,
        output);
    if (err != cudaSuccess) return err;
  }
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return err;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  err = cudaEventCreate(&start);
  if (err == cudaSuccess) err = cudaEventCreate(&stop);
  if (err == cudaSuccess) err = cudaEventRecord(start, stream);
  for (uint32_t index = 0;
       err == cudaSuccess && index < kLtGemvAutotuneIterations; ++index) {
    err = encoded_row_major_gemv_beta(handle, matrix, input, plan->rows,
                                      plan->cols, plan->dtype, 0.0f, output);
  }
  if (err == cudaSuccess) err = cudaEventRecord(stop, stream);
  if (err == cudaSuccess) err = cudaEventSynchronize(stop);
  if (err == cudaSuccess) {
    const uint64_t total_ns = cuda_event_elapsed_ns(start, stop);
    *avg_ns = total_ns / kLtGemvAutotuneIterations;
  }
  if (stop != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(stop);
    if (err == cudaSuccess) err = cleanup_err;
  }
  if (start != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(start);
    if (err == cudaSuccess) err = cleanup_err;
  }
  return err;
}

cudaError_t time_lt_gemv_graph_candidate(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const LtGemvPlan *plan, const uint16_t *matrix,
    const uint16_t *input, float *output, const cublasLtMatmulAlgo_t *algo,
    uint64_t *avg_ns) {
  if (avg_ns == nullptr) {
    return cudaErrorInvalidValue;
  }
  *avg_ns = 0;
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return err;
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (err != cudaSuccess) {
    return err;
  }
  err = launch_lt_gemv_plan(handle, stream, workspace, workspace_bytes, plan,
                            matrix, input, 0.0f, output, algo);
  cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
  if (err != cudaSuccess) {
    if (end_err == cudaSuccess && graph != nullptr) cudaGraphDestroy(graph);
    return err;
  }
  if (end_err != cudaSuccess) {
    if (graph != nullptr) cudaGraphDestroy(graph);
    return end_err;
  }
  err = cudaGraphInstantiate(&graph_exec, graph, 0);
  if (err != cudaSuccess) {
    if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
    if (graph != nullptr) cudaGraphDestroy(graph);
    return err;
  }

  for (uint32_t index = 0; index < kLtGemvAutotuneWarmups; ++index) {
    err = cudaGraphLaunch(graph_exec, stream);
    if (err != cudaSuccess) break;
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (err == cudaSuccess) err = cudaEventCreate(&start);
  if (err == cudaSuccess) err = cudaEventCreate(&stop);
  if (err == cudaSuccess) err = cudaEventRecord(start, stream);
  for (uint32_t index = 0;
       err == cudaSuccess && index < kLtGemvAutotuneIterations; ++index) {
    err = cudaGraphLaunch(graph_exec, stream);
  }
  if (err == cudaSuccess) err = cudaEventRecord(stop, stream);
  if (err == cudaSuccess) err = cudaEventSynchronize(stop);
  if (err == cudaSuccess) {
    const uint64_t total_ns = cuda_event_elapsed_ns(start, stop);
    *avg_ns = total_ns / kLtGemvAutotuneIterations;
  }

  if (stop != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(stop);
    if (err == cudaSuccess) err = cleanup_err;
  }
  if (start != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(start);
    if (err == cudaSuccess) err = cleanup_err;
  }
  cudaError_t cleanup_err = cudaGraphExecDestroy(graph_exec);
  if (err == cudaSuccess && cleanup_err != cudaSuccess) err = cleanup_err;
  cleanup_err = cudaGraphDestroy(graph);
  if (err == cudaSuccess && cleanup_err != cudaSuccess) err = cleanup_err;
  return err;
}

cudaError_t time_cublas_gemv_graph_candidate(
    cublasHandle_t handle, cudaStream_t stream, const LtGemvPlan *plan,
    const uint16_t *matrix, const uint16_t *input, float *output,
    uint64_t *avg_ns) {
  if (avg_ns == nullptr || plan == nullptr || !plan->ready) {
    return cudaErrorInvalidValue;
  }
  *avg_ns = 0;
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return err;
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (err != cudaSuccess) {
    return err;
  }
  err = encoded_row_major_gemv_beta(handle, matrix, input, plan->rows,
                                    plan->cols, plan->dtype, 0.0f, output);
  cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
  if (err != cudaSuccess) {
    if (end_err == cudaSuccess && graph != nullptr) cudaGraphDestroy(graph);
    return err;
  }
  if (end_err != cudaSuccess) {
    if (graph != nullptr) cudaGraphDestroy(graph);
    return end_err;
  }
  err = cudaGraphInstantiate(&graph_exec, graph, 0);
  if (err != cudaSuccess) {
    if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
    if (graph != nullptr) cudaGraphDestroy(graph);
    return err;
  }

  for (uint32_t index = 0; index < kLtGemvAutotuneWarmups; ++index) {
    err = cudaGraphLaunch(graph_exec, stream);
    if (err != cudaSuccess) break;
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (err == cudaSuccess) err = cudaEventCreate(&start);
  if (err == cudaSuccess) err = cudaEventCreate(&stop);
  if (err == cudaSuccess) err = cudaEventRecord(start, stream);
  for (uint32_t index = 0;
       err == cudaSuccess && index < kLtGemvAutotuneIterations; ++index) {
    err = cudaGraphLaunch(graph_exec, stream);
  }
  if (err == cudaSuccess) err = cudaEventRecord(stop, stream);
  if (err == cudaSuccess) err = cudaEventSynchronize(stop);
  if (err == cudaSuccess) {
    const uint64_t total_ns = cuda_event_elapsed_ns(start, stop);
    *avg_ns = total_ns / kLtGemvAutotuneIterations;
  }

  if (stop != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(stop);
    if (err == cudaSuccess) err = cleanup_err;
  }
  if (start != nullptr) {
    cudaError_t cleanup_err = cudaEventDestroy(start);
    if (err == cudaSuccess) err = cleanup_err;
  }
  cudaError_t cleanup_err = cudaGraphExecDestroy(graph_exec);
  if (err == cudaSuccess && cleanup_err != cudaSuccess) err = cleanup_err;
  cleanup_err = cudaGraphDestroy(graph);
  if (err == cudaSuccess && cleanup_err != cudaSuccess) err = cleanup_err;
  return err;
}

cudaError_t autotune_lt_gemv_plan(
    cublasHandle_t cublas, cublasLtHandle_t cublas_lt, cudaStream_t stream,
    void *workspace, size_t workspace_bytes, LtGemvPlan *plan,
    const uint16_t *matrix, const uint16_t *input, float *output) {
  if (plan == nullptr || !plan->ready) {
    return cudaErrorInvalidValue;
  }
  uint64_t best_avg_ns = 0;
  cudaError_t err = time_lt_gemv_candidate(
      cublas_lt, stream, workspace, workspace_bytes, plan, matrix, input,
      output, nullptr, &best_avg_ns);
  if (err != cudaSuccess) {
    return err;
  }
  plan->backend = kGemvBackendLt;
  plan->has_algo = false;
  plan->selected_heuristic = UINT32_MAX;
  plan->tuned_avg_ns = best_avg_ns;

  uint64_t cublas_avg_ns = 0;
  const cudaError_t cublas_err =
      time_cublas_gemv_candidate(cublas, stream, plan, matrix, input, output,
                                 &cublas_avg_ns);
  if (cublas_err == cudaSuccess && cublas_avg_ns != 0 &&
      (best_avg_ns == 0 || cublas_avg_ns < best_avg_ns)) {
    best_avg_ns = cublas_avg_ns;
    plan->backend = kGemvBackendCublas;
    plan->has_algo = false;
    plan->selected_heuristic = UINT32_MAX;
    plan->tuned_avg_ns = cublas_avg_ns;
  }

  cublasLtMatmulHeuristicResult_t heuristics[kLtGemvMaxHeuristics]{};
  uint32_t heuristic_count = 0;
  const cudaError_t heuristic_err = find_lt_gemv_heuristics(
      cublas_lt, plan, workspace_bytes, heuristics, &heuristic_count);
  if (heuristic_err != cudaSuccess) {
    return cudaSuccess;
  }
  plan->heuristic_count = heuristic_count;
  for (uint32_t index = 0; index < heuristic_count; ++index) {
    uint64_t avg_ns = 0;
    err = time_lt_gemv_candidate(
        cublas_lt, stream, workspace, workspace_bytes, plan, matrix, input,
        output, &heuristics[index].algo, &avg_ns);
    if (err != cudaSuccess || avg_ns == 0) {
      continue;
    }
    if (best_avg_ns == 0 || avg_ns < best_avg_ns) {
      best_avg_ns = avg_ns;
      plan->backend = kGemvBackendLt;
      plan->algo = heuristics[index].algo;
      plan->has_algo = true;
      plan->selected_heuristic = index;
      plan->tuned_avg_ns = avg_ns;
    }
  }
  uint64_t graph_best_avg_ns = 0;
  uint32_t graph_best_backend = plan->backend;
  bool graph_best_has_algo = plan->has_algo;
  uint32_t graph_best_heuristic = plan->selected_heuristic;
  cublasLtMatmulAlgo_t graph_best_algo = plan->algo;

  auto consider_graph_candidate = [&](uint32_t backend, bool has_algo,
                                      uint32_t heuristic_index,
                                      const cublasLtMatmulAlgo_t *algo,
                                      uint64_t avg_ns) {
    if (avg_ns == 0) {
      return;
    }
    if (graph_best_avg_ns == 0 || avg_ns < graph_best_avg_ns) {
      graph_best_avg_ns = avg_ns;
      graph_best_backend = backend;
      graph_best_has_algo = has_algo;
      graph_best_heuristic = heuristic_index;
      if (algo != nullptr) {
        graph_best_algo = *algo;
      }
    }
  };

  uint64_t graph_avg_ns = 0;
  cudaError_t graph_err = time_cublas_gemv_graph_candidate(
      cublas, stream, plan, matrix, input, output, &graph_avg_ns);
  if (graph_err == cudaSuccess) {
    consider_graph_candidate(kGemvBackendCublas, false, UINT32_MAX, nullptr,
                             graph_avg_ns);
  }
  graph_avg_ns = 0;
  graph_err = time_lt_gemv_graph_candidate(
      cublas_lt, stream, workspace, workspace_bytes, plan, matrix, input,
      output, nullptr, &graph_avg_ns);
  if (graph_err == cudaSuccess) {
    consider_graph_candidate(kGemvBackendLt, false, UINT32_MAX, nullptr,
                             graph_avg_ns);
  }
  for (uint32_t index = 0; index < heuristic_count; ++index) {
    graph_avg_ns = 0;
    graph_err = time_lt_gemv_graph_candidate(
        cublas_lt, stream, workspace, workspace_bytes, plan, matrix, input,
        output, &heuristics[index].algo, &graph_avg_ns);
    if (graph_err == cudaSuccess) {
      consider_graph_candidate(kGemvBackendLt, true, index,
                               &heuristics[index].algo, graph_avg_ns);
    }
  }
  if (graph_best_avg_ns != 0) {
    plan->backend = graph_best_backend;
    plan->has_algo = graph_best_has_algo;
    plan->selected_heuristic = graph_best_heuristic;
    plan->tuned_avg_ns = graph_best_avg_ns;
    if (graph_best_has_algo) {
      plan->algo = graph_best_algo;
    }
  }
  return cudaSuccess;
}

cudaError_t encoded_row_major_gemm_tokens(
    cublasHandle_t handle, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t tokens, uint32_t dtype, float beta,
    float *output) {
  if (rows == 0 || cols == 0 || tokens == 0 || rows > INT32_MAX ||
      cols > INT32_MAX || tokens > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  const float alpha = 1.0f;
  const cudaDataType_t data_type = encoded_cuda_type(dtype);
  cublasStatus_t status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rows),
      static_cast<int>(tokens), static_cast<int>(cols), &alpha, matrix,
      data_type, static_cast<int>(cols), input, data_type,
      static_cast<int>(cols), &beta, output, CUDA_R_32F,
      static_cast<int>(rows), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return cublas_to_cuda(status);
}

cudaError_t encoded_row_major_gemm_tokens_lt(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t tokens, uint32_t dtype, float beta,
    float *output) {
  if (handle == nullptr || rows == 0 || cols == 0 || tokens == 0 ||
      rows > INT32_MAX || cols > INT32_MAX || tokens > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  const cudaDataType_t data_type = encoded_cuda_type(dtype);
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasStatus_t status =
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t op_a = CUBLAS_OP_N;
  cublasOperation_t op_b = CUBLAS_OP_T;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(
        &a_desc, data_type, tokens, cols, cols);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&b_desc, data_type, rows, cols, cols);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, tokens, rows, rows);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_32F, tokens, rows, rows);
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutSetAttribute(
        d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (status == CUBLAS_STATUS_SUCCESS) {
    const float alpha = 1.0f;
    status = cublasLtMatmul(handle, op_desc, &alpha, input, a_desc, matrix,
                            b_desc, &beta, output, c_desc, output, d_desc,
                            nullptr, workspace, workspace_bytes, stream);
  }
  destroy_lt_descriptors(op_desc, a_desc, b_desc, c_desc, d_desc);
  return cublas_to_cuda(status);
}

cudaError_t encoded_row_major_gemm_tokens_best(
    cublasHandle_t cublas, cublasLtHandle_t cublas_lt, cudaStream_t stream,
    void *workspace, size_t workspace_bytes, const uint16_t *matrix,
    const uint16_t *input, uint32_t rows, uint32_t cols, uint32_t tokens,
    uint32_t dtype, float beta, float *output) {
  cudaError_t err = encoded_row_major_gemm_tokens_lt(
      cublas_lt, stream, workspace, workspace_bytes, matrix, input, rows, cols,
      tokens, dtype, beta, output);
  if (err == cudaSuccess) {
    return err;
  }
  return encoded_row_major_gemm_tokens(cublas, matrix, input, rows, cols,
                                       tokens, dtype, beta, output);
}
