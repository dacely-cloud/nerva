#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

constexpr uint32_t kGemvBackendLt = 0;
constexpr uint32_t kGemvBackendCublas = 1;

struct LtGemvPlan {
  uint32_t rows = 0;
  uint32_t cols = 0;
  uint32_t dtype = 0;
  uint32_t backend = 0;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulAlgo_t algo{};
  bool ready = false;
  bool has_algo = false;
  uint32_t heuristic_count = 0;
  uint32_t selected_heuristic = UINT32_MAX;
  uint64_t tuned_avg_ns = 0;
};

struct LtGemmTokensPlan {
  uint32_t rows = 0;
  uint32_t cols = 0;
  uint32_t tokens = 0;
  uint32_t dtype = 0;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  bool ready = false;
};

cudaError_t cublas_to_cuda(cublasStatus_t status);
cudaError_t configure_cublas(cublasHandle_t handle, cudaStream_t stream,
                             void *workspace, size_t workspace_bytes);

cudaError_t encoded_row_major_gemv_beta(
    cublasHandle_t handle, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t dtype, float beta, float *output);
cudaError_t encoded_row_major_gemv(cublasHandle_t handle, const uint16_t *matrix,
                                   const uint16_t *input, uint32_t rows,
                                   uint32_t cols, uint32_t dtype,
                                   float *output);
cudaError_t encoded_row_major_gemv_strided_batched(
    cublasHandle_t handle, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t tokens, uint32_t dtype, float beta,
    float *output);

void destroy_lt_gemv_plan(LtGemvPlan *plan);
void destroy_lt_gemm_tokens_plan(LtGemmTokensPlan *plan);
cudaError_t create_lt_gemm_tokens_plan(LtGemmTokensPlan *plan, uint32_t rows,
                                       uint32_t cols, uint32_t tokens,
                                       uint32_t dtype);
cudaError_t launch_lt_gemm_tokens_plan(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const LtGemmTokensPlan *plan,
    const uint16_t *matrix, const uint16_t *input, float beta, float *output);
cudaError_t create_lt_gemv_plan(LtGemvPlan *plan, uint32_t rows,
                                uint32_t cols, uint32_t dtype);
cudaError_t encoded_row_major_gemv_planned(
    cublasHandle_t cublas, cublasLtHandle_t cublas_lt, cudaStream_t stream,
    void *workspace, size_t workspace_bytes, const LtGemvPlan *plan,
    const uint16_t *matrix, const uint16_t *input, float beta, float *output);
cudaError_t autotune_lt_gemv_plan(
    cublasHandle_t cublas, cublasLtHandle_t cublas_lt, cudaStream_t stream,
    void *workspace, size_t workspace_bytes, LtGemvPlan *plan,
    const uint16_t *matrix, const uint16_t *input, float *output);

cudaError_t encoded_row_major_gemm_tokens(
    cublasHandle_t handle, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t tokens, uint32_t dtype, float beta,
    float *output);
cudaError_t encoded_row_major_gemm_tokens_lt(
    cublasLtHandle_t handle, cudaStream_t stream, void *workspace,
    size_t workspace_bytes, const uint16_t *matrix, const uint16_t *input,
    uint32_t rows, uint32_t cols, uint32_t tokens, uint32_t dtype, float beta,
    float *output);
cudaError_t encoded_row_major_gemm_tokens_best(
    cublasHandle_t cublas, cublasLtHandle_t cublas_lt, cudaStream_t stream,
    void *workspace, size_t workspace_bytes, const uint16_t *matrix,
    const uint16_t *input, uint32_t rows, uint32_t cols, uint32_t tokens,
    uint32_t dtype, float beta, float *output);

cudaError_t launch_deepseek_fp8_f32_scale_matvec(
    cudaStream_t stream, const uint8_t *weights, const float *scales,
    const float *input, uint32_t rows, uint32_t cols, uint32_t block_rows,
    uint32_t block_cols, float *output);
