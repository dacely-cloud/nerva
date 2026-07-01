#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

cudaError_t launch_deepseek_fp8_f32_scale_matvec(
    cudaStream_t stream, const uint8_t *weights, const float *scales,
    const float *input, uint32_t rows, uint32_t cols, uint32_t block_rows,
    uint32_t block_cols, float *output);
cudaError_t launch_deepseek_fp8_f32_scale_encoded_matvec(
    cudaStream_t stream, const uint8_t *weights, const float *scales,
    const uint16_t *input, uint32_t input_dtype, uint32_t rows, uint32_t cols,
    uint32_t block_rows, uint32_t block_cols, float *output);
cudaError_t launch_deepseek_fp8_f32_scale_dual_encoded_matvec(
    cudaStream_t stream, const uint8_t *weights_a, const float *scales_a,
    const uint8_t *weights_b, const float *scales_b, const uint16_t *input,
    uint32_t input_dtype, uint32_t rows, uint32_t cols, uint32_t block_rows,
    uint32_t block_cols, float *output_a, float *output_b);
cudaError_t launch_deepseek_fp8_f32_scale_slots_encoded_matvec(
    cudaStream_t stream, const uint8_t *weights, const uint16_t *scale_slots,
    const uint16_t *input, uint32_t input_dtype, uint32_t rows, uint32_t cols,
    uint32_t block_rows, uint32_t block_cols, float *output);
cudaError_t launch_deepseek_fp8_f32_scale_encoded_gemm_tokens(
    cudaStream_t stream, const uint8_t *weights, const float *scales,
    const uint16_t *input, uint32_t input_dtype, uint32_t rows, uint32_t cols,
    uint32_t tokens, uint32_t block_rows, uint32_t block_cols,
    float *output);
cudaError_t launch_deepseek_fp8_e8m0_scale_encoded_matvec(
    cudaStream_t stream, const uint8_t *weights, const uint8_t *scales,
    const uint16_t *input, uint32_t input_dtype, uint32_t rows, uint32_t cols,
    uint32_t block_rows, uint32_t block_cols, float *output);
cudaError_t launch_deepseek_fp8_e8m0_scale_dual_encoded_matvec(
    cudaStream_t stream, const uint8_t *weights_a, const uint8_t *scales_a,
    const uint8_t *weights_b, const uint8_t *scales_b, const uint16_t *input,
    uint32_t input_dtype, uint32_t rows, uint32_t cols, uint32_t block_rows,
    uint32_t block_cols, float *output_a, float *output_b);
cudaError_t launch_deepseek_fp8_e8m0_scale_encoded_gemm_tokens(
    cudaStream_t stream, const uint8_t *weights, const uint8_t *scales,
    const uint16_t *input, uint32_t input_dtype, uint32_t rows, uint32_t cols,
    uint32_t tokens, uint32_t block_rows, uint32_t block_cols,
    float *output);
cudaError_t launch_deepseek_fp8_e8m0_scale_matvec(
    cudaStream_t stream, const uint8_t *weights, const uint8_t *scales,
    const float *input, uint32_t rows, uint32_t cols, uint32_t block_rows,
    uint32_t block_cols, float *output);
cudaError_t launch_deepseek_fp8_e8m0_scale_grouped_matvec(
    cudaStream_t stream, const uint8_t *weights, const uint8_t *scales,
    const float *input, uint32_t groups, uint32_t rows_per_group,
    uint32_t cols_per_group, uint32_t block_rows, uint32_t block_cols,
    float *output);
