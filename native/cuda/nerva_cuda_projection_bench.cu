#include "nerva_cuda_api.h"

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kDTypeF16 = 0;
constexpr uint32_t kDTypeBF16 = 1;
constexpr uint32_t kStrategyCublasLt = 1;
constexpr uint32_t kStrategyCustom = 2;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kInitBlocks = 4096;
constexpr uint32_t kMaxHeuristics = 8;
constexpr uint32_t kNoHeuristicIndex = 0xffffffffu;
constexpr size_t kWorkspaceBytes = 16ull * 1024ull * 1024ull;

__device__ uint16_t f32_to_encoded(float value, uint32_t dtype) {
  if (dtype == kDTypeBF16) {
    uint32_t bits = __float_as_uint(value);
    uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
  }
  return __half_as_ushort(__float2half_rn(value));
}

__device__ float encoded_to_f32(uint16_t value, uint32_t dtype) {
  if (dtype == kDTypeBF16) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
  }
  return __half2float(__ushort_as_half(value));
}

__device__ float deterministic_value(uint64_t index) {
  const int32_t centered = static_cast<int32_t>((index * 17ull + 13ull) % 127ull) - 63;
  return static_cast<float>(centered) * 0.0078125f;
}

__global__ void init_encoded_kernel(uint16_t *data, uint64_t elements,
                                    uint32_t dtype) {
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  while (index < elements) {
    data[index] = f32_to_encoded(deterministic_value(index), dtype);
    index += stride;
  }
}

__global__ void row_major_gemv_kernel(const uint16_t *matrix,
                                      const uint16_t *input,
                                      uint32_t rows,
                                      uint32_t cols,
                                      uint32_t dtype,
                                      float *output) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  float sum = 0.0f;
  const uint64_t row_base = static_cast<uint64_t>(row) * cols;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    sum += encoded_to_f32(matrix[row_base + col], dtype) *
           encoded_to_f32(input[col], dtype);
  }
  __shared__ float partials[kThreads];
  partials[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partials[threadIdx.x] += partials[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[row] = partials[0];
  }
}

__global__ void compare_outputs_kernel(const float *baseline,
                                       const float *candidate,
                                       uint32_t rows,
                                       float tolerance,
                                       uint32_t *mismatches,
                                       uint32_t *max_diff_bits) {
  const uint32_t stride = gridDim.x * blockDim.x;
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < rows) {
    const float diff = fabsf(baseline[index] - candidate[index]);
    if (!isfinite(diff) || diff > tolerance) {
      atomicAdd(mismatches, 1u);
    }
    atomicMax(max_diff_bits, __float_as_uint(diff));
    index += stride;
  }
}

cudaDataType_t encoded_cuda_type(uint32_t dtype) {
  return dtype == kDTypeBF16 ? CUDA_R_16BF : CUDA_R_16F;
}

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

void clear_result(const NervaCudaProjectionBenchRequest *request,
                  NervaCudaProjectionBenchResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request != nullptr) {
    out->dtype = request->dtype;
    out->rows = request->rows;
    out->cols = request->cols;
    out->iterations = request->iterations;
    out->warmup_iterations = request->warmup_iterations;
    out->matrix_bytes = static_cast<uint64_t>(request->rows) * request->cols *
                        sizeof(uint16_t);
    out->input_bytes = static_cast<uint64_t>(request->cols) * sizeof(uint16_t);
    out->output_bytes = static_cast<uint64_t>(request->rows) * sizeof(float);
    out->device_arena_bytes = out->matrix_bytes + out->input_bytes +
                              out->output_bytes * 2ull + sizeof(uint32_t) * 2ull +
                              kWorkspaceBytes;
  }
}

int fail(NervaCudaProjectionBenchResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

void record_cleanup_error(NervaCudaProjectionBenchResult *out, cudaError_t err) {
  if (err != cudaSuccess && out->cuda_error == 0) {
    out->cuda_error = static_cast<int32_t>(err);
    out->status = -1;
  }
}

uint64_t elapsed_ns(cudaEvent_t start, cudaEvent_t stop) {
  float elapsed_ms = 0.0f;
  cudaError_t err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (err != cudaSuccess || elapsed_ms <= 0.0f) {
    return 0;
  }
  const uint64_t ns = static_cast<uint64_t>(elapsed_ms * 1000000.0f);
  return ns == 0 ? 1 : ns;
}

uint64_t effective_bandwidth(uint64_t bytes_per_iteration,
                             uint32_t iterations,
                             uint64_t total_ns) {
  if (bytes_per_iteration == 0 || iterations == 0 || total_ns == 0) {
    return 0;
  }
  const long double bytes =
      static_cast<long double>(bytes_per_iteration) * iterations;
  const long double seconds = static_cast<long double>(total_ns) / 1000000000.0L;
  return static_cast<uint64_t>(bytes / seconds);
}

cudaError_t create_lt_layouts(uint32_t rows, uint32_t cols, uint32_t dtype,
                              cublasLtMatmulDesc_t *op_desc,
                              cublasLtMatrixLayout_t *a_desc,
                              cublasLtMatrixLayout_t *b_desc,
                              cublasLtMatrixLayout_t *c_desc,
                              cublasLtMatrixLayout_t *d_desc) {
  const cudaDataType_t data_type = encoded_cuda_type(dtype);
  cublasStatus_t status =
      cublasLtMatmulDescCreate(op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t op = CUBLAS_OP_N;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        *op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        *op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(a_desc, data_type, rows, cols, cols);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(b_desc, data_type, cols, 1, 1);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(c_desc, CUDA_R_32F, rows, 1, 1);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(d_desc, CUDA_R_32F, rows, 1, 1);
  }
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutSetAttribute(
        *a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutSetAttribute(
        *b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutSetAttribute(
        *c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutSetAttribute(
        *d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  }
  return cublas_to_cuda(status);
}

void destroy_lt_layouts(cublasLtMatmulDesc_t op_desc,
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

cudaError_t launch_cublaslt(cublasLtHandle_t handle,
                            cublasLtMatmulDesc_t op_desc,
                            cublasLtMatrixLayout_t a_desc,
                            cublasLtMatrixLayout_t b_desc,
                            cublasLtMatrixLayout_t c_desc,
                            cublasLtMatrixLayout_t d_desc,
                            const uint16_t *matrix,
                            const uint16_t *input,
                            float *output,
                            void *workspace,
                            cudaStream_t stream,
                            const cublasLtMatmulAlgo_t *algo) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const cublasStatus_t status =
      cublasLtMatmul(handle, op_desc, &alpha, matrix, a_desc, input, b_desc,
                     &beta, output, c_desc, output, d_desc, algo, workspace,
                     kWorkspaceBytes, stream);
  return cublas_to_cuda(status);
}

cudaError_t time_cublaslt(cublasLtHandle_t handle,
                          cublasLtMatmulDesc_t op_desc,
                          cublasLtMatrixLayout_t a_desc,
                          cublasLtMatrixLayout_t b_desc,
                          cublasLtMatrixLayout_t c_desc,
                          cublasLtMatrixLayout_t d_desc,
                          const NervaCudaProjectionBenchRequest *request,
                          const uint16_t *matrix,
                          const uint16_t *input,
                          float *output,
                          void *workspace,
                          cudaStream_t stream,
                          cudaEvent_t start,
                          cudaEvent_t stop,
                          uint64_t *total_ns,
                          const cublasLtMatmulAlgo_t *algo) {
  for (uint32_t index = 0; index < request->warmup_iterations; ++index) {
    cudaError_t err = launch_cublaslt(handle, op_desc, a_desc, b_desc, c_desc,
                                      d_desc, matrix, input, output, workspace,
                                      stream, algo);
    if (err != cudaSuccess) return err;
  }
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return err;
  err = cudaEventRecord(start, stream);
  if (err != cudaSuccess) return err;
  for (uint32_t index = 0; index < request->iterations; ++index) {
    err = launch_cublaslt(handle, op_desc, a_desc, b_desc, c_desc, d_desc,
                          matrix, input, output, workspace, stream, algo);
    if (err != cudaSuccess) return err;
  }
  err = cudaEventRecord(stop, stream);
  if (err != cudaSuccess) return err;
  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess) return err;
  *total_ns = elapsed_ns(start, stop);
  return cudaSuccess;
}

cudaError_t find_cublaslt_heuristics(
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t op_desc,
    cublasLtMatrixLayout_t a_desc,
    cublasLtMatrixLayout_t b_desc,
    cublasLtMatrixLayout_t c_desc,
    cublasLtMatrixLayout_t d_desc,
    cublasLtMatmulHeuristicResult_t *heuristics,
    uint32_t *heuristic_count) {
  *heuristic_count = 0;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasStatus_t status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_to_cuda(status);
  }
  size_t workspace_bytes = kWorkspaceBytes;
  status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes));
  int returned_count = 0;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoGetHeuristic(
        handle, op_desc, a_desc, b_desc, c_desc, d_desc, preference,
        kMaxHeuristics, heuristics, &returned_count);
  }
  cublasLtMatmulPreferenceDestroy(preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_to_cuda(status);
  }
  if (returned_count > 0) {
    *heuristic_count = static_cast<uint32_t>(returned_count);
  }
  return cudaSuccess;
}

cudaError_t time_custom(const NervaCudaProjectionBenchRequest *request,
                        const uint16_t *matrix,
                        const uint16_t *input,
                        float *output,
                        cudaStream_t stream,
                        cudaEvent_t start,
                        cudaEvent_t stop,
                        uint64_t *total_ns) {
  const dim3 grid(request->rows);
  for (uint32_t index = 0; index < request->warmup_iterations; ++index) {
    row_major_gemv_kernel<<<grid, kThreads, 0, stream>>>(
        matrix, input, request->rows, request->cols, request->dtype, output);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return err;
  err = cudaEventRecord(start, stream);
  if (err != cudaSuccess) return err;
  for (uint32_t index = 0; index < request->iterations; ++index) {
    row_major_gemv_kernel<<<grid, kThreads, 0, stream>>>(
        matrix, input, request->rows, request->cols, request->dtype, output);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }
  err = cudaEventRecord(stop, stream);
  if (err != cudaSuccess) return err;
  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess) return err;
  *total_ns = elapsed_ns(start, stop);
  return cudaSuccess;
}

}  // namespace

extern "C" int nerva_cuda_projection_bench(
    const NervaCudaProjectionBenchRequest *request,
    NervaCudaProjectionBenchResult *out) {
  if (request == nullptr || out == nullptr) {
    return -1;
  }
  clear_result(request, out);
  if ((request->dtype != kDTypeF16 && request->dtype != kDTypeBF16) ||
      request->rows == 0 || request->cols == 0 || request->iterations == 0) {
    return fail(out, cudaErrorInvalidValue);
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorNoDevice);
    return -1;
  }
  out->device_ordinal = 0;
  err = cudaSetDevice(out->device_ordinal);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  cudaDeviceProp props{};
  err = cudaGetDeviceProperties(&props, out->device_ordinal);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  out->compute_capability_major = props.major;
  out->compute_capability_minor = props.minor;

  uint16_t *matrix = nullptr;
  uint16_t *input = nullptr;
  float *cublas_output = nullptr;
  float *custom_output = nullptr;
  uint32_t *mismatches = nullptr;
  uint32_t *max_diff_bits = nullptr;
  void *workspace = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cublasLtHandle_t lt = nullptr;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;

  auto cleanup = [&]() {
    destroy_lt_layouts(op_desc, a_desc, b_desc, c_desc, d_desc);
    if (lt != nullptr) cublasLtDestroy(lt);
    if (workspace != nullptr && cudaFree(workspace) == cudaSuccess) out->device_frees += 1;
    if (max_diff_bits != nullptr && cudaFree(max_diff_bits) == cudaSuccess) out->device_frees += 1;
    if (mismatches != nullptr && cudaFree(mismatches) == cudaSuccess) out->device_frees += 1;
    if (custom_output != nullptr && cudaFree(custom_output) == cudaSuccess) out->device_frees += 1;
    if (cublas_output != nullptr && cudaFree(cublas_output) == cudaSuccess) out->device_frees += 1;
    if (input != nullptr && cudaFree(input) == cudaSuccess) out->device_frees += 1;
    if (matrix != nullptr && cudaFree(matrix) == cudaSuccess) out->device_frees += 1;
    if (stop != nullptr) record_cleanup_error(out, cudaEventDestroy(stop));
    if (start != nullptr) record_cleanup_error(out, cudaEventDestroy(start));
    if (stream != nullptr) record_cleanup_error(out, cudaStreamDestroy(stream));
  };
  auto fail_with_cleanup = [&](cudaError_t cleanup_err) {
    fail(out, cleanup_err);
    cleanup();
    return -1;
  };

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) return fail_with_cleanup(err);
  err = cudaEventCreate(&start);
  if (err != cudaSuccess) return fail_with_cleanup(err);
  err = cudaEventCreate(&stop);
  if (err != cudaSuccess) return fail_with_cleanup(err);
  err = cudaMalloc(reinterpret_cast<void **>(&matrix), out->matrix_bytes);
  if (err != cudaSuccess) return fail_with_cleanup(err);
  out->device_allocations += 1;
  err = cudaMalloc(reinterpret_cast<void **>(&input), out->input_bytes);
  if (err != cudaSuccess) return fail_with_cleanup(err);
  out->device_allocations += 1;
  err = cudaMalloc(reinterpret_cast<void **>(&cublas_output), out->output_bytes);
  if (err != cudaSuccess) return fail_with_cleanup(err);
  out->device_allocations += 1;
  err = cudaMalloc(reinterpret_cast<void **>(&custom_output), out->output_bytes);
  if (err != cudaSuccess) return fail_with_cleanup(err);
  out->device_allocations += 1;
  err = cudaMalloc(reinterpret_cast<void **>(&mismatches), sizeof(uint32_t));
  if (err != cudaSuccess) return fail_with_cleanup(err);
  out->device_allocations += 1;
  err = cudaMalloc(reinterpret_cast<void **>(&max_diff_bits), sizeof(uint32_t));
  if (err != cudaSuccess) return fail_with_cleanup(err);
  out->device_allocations += 1;
  err = cudaMalloc(&workspace, kWorkspaceBytes);
  if (err != cudaSuccess) return fail_with_cleanup(err);
  out->device_allocations += 1;

  init_encoded_kernel<<<kInitBlocks, kThreads, 0, stream>>>(
      matrix, static_cast<uint64_t>(request->rows) * request->cols,
      request->dtype);
  err = cudaGetLastError();
  if (err == cudaSuccess) {
    init_encoded_kernel<<<1, kThreads, 0, stream>>>(input, request->cols,
                                                    request->dtype);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return fail_with_cleanup(err);
  out->sync_calls += 1;

  cublasStatus_t blas_status = cublasLtCreate(&lt);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return fail_with_cleanup(cublas_to_cuda(blas_status));
  }
  err = create_lt_layouts(request->rows, request->cols, request->dtype,
                          &op_desc, &a_desc, &b_desc, &c_desc, &d_desc);
  if (err != cudaSuccess) return fail_with_cleanup(err);

  err = time_cublaslt(lt, op_desc, a_desc, b_desc, c_desc, d_desc, request,
                      matrix, input, cublas_output, workspace, stream, start,
                      stop, &out->cublaslt_default_total_ns, nullptr);
  if (err != cudaSuccess) return fail_with_cleanup(err);
  out->sync_calls += 2;
  out->kernel_launches += request->warmup_iterations + request->iterations;
  out->cublaslt_default_avg_ns =
      out->cublaslt_default_total_ns / request->iterations;
  out->cublaslt_total_ns = out->cublaslt_default_total_ns;
  out->cublaslt_avg_ns = out->cublaslt_default_avg_ns;
  out->cublaslt_best_heuristic_index = kNoHeuristicIndex;

  cublasLtMatmulHeuristicResult_t heuristics[kMaxHeuristics]{};
  uint32_t heuristic_count = 0;
  cudaError_t heuristic_err = find_cublaslt_heuristics(
      lt, op_desc, a_desc, b_desc, c_desc, d_desc, heuristics,
      &heuristic_count);
  if (heuristic_err == cudaSuccess) {
    out->cublaslt_heuristic_count = heuristic_count;
    for (uint32_t index = 0; index < heuristic_count; ++index) {
      uint64_t heuristic_total_ns = 0;
      heuristic_err = time_cublaslt(
          lt, op_desc, a_desc, b_desc, c_desc, d_desc, request, matrix, input,
          cublas_output, workspace, stream, start, stop, &heuristic_total_ns,
          &heuristics[index].algo);
      if (heuristic_err != cudaSuccess || heuristic_total_ns == 0) {
        continue;
      }
      out->sync_calls += 2;
      out->kernel_launches +=
          request->warmup_iterations + request->iterations;
      const uint64_t heuristic_avg_ns =
          heuristic_total_ns / request->iterations;
      if (out->cublaslt_best_heuristic_avg_ns == 0 ||
          heuristic_avg_ns < out->cublaslt_best_heuristic_avg_ns) {
        out->cublaslt_best_heuristic_index = index;
        out->cublaslt_best_heuristic_total_ns = heuristic_total_ns;
        out->cublaslt_best_heuristic_avg_ns = heuristic_avg_ns;
      }
      if (heuristic_avg_ns < out->cublaslt_avg_ns) {
        out->cublaslt_total_ns = heuristic_total_ns;
        out->cublaslt_avg_ns = heuristic_avg_ns;
      }
    }
  }

  err = time_custom(request, matrix, input, custom_output, stream, start, stop,
                    &out->custom_total_ns);
  if (err != cudaSuccess) return fail_with_cleanup(err);
  out->sync_calls += 2;
  out->kernel_launches += request->warmup_iterations + request->iterations;

  if (out->custom_total_ns > 0) {
    out->custom_avg_ns = out->custom_total_ns / request->iterations;
  }
  const uint64_t bytes_per_projection =
      out->matrix_bytes + out->input_bytes + out->output_bytes;
  out->cublaslt_effective_bandwidth_bps = effective_bandwidth(
      bytes_per_projection, request->iterations, out->cublaslt_total_ns);
  out->custom_effective_bandwidth_bps = effective_bandwidth(
      bytes_per_projection, request->iterations, out->custom_total_ns);
  out->selected_strategy =
      out->custom_avg_ns > 0 && out->custom_avg_ns < out->cublaslt_avg_ns
          ? kStrategyCustom
          : kStrategyCublasLt;

  err = cudaMemsetAsync(mismatches, 0, sizeof(uint32_t), stream);
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(max_diff_bits, 0, sizeof(uint32_t), stream);
  }
  if (err == cudaSuccess) {
    const uint32_t compare_blocks =
        request->rows < kInitBlocks ? request->rows : kInitBlocks;
    compare_outputs_kernel<<<compare_blocks, kThreads, 0, stream>>>(
        cublas_output, custom_output, request->rows, 0.25f, mismatches,
        max_diff_bits);
    err = cudaGetLastError();
  }
  uint32_t host_mismatches = 0;
  uint32_t host_max_diff_bits = 0;
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(&host_mismatches, mismatches, sizeof(host_mismatches),
                          cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(&host_max_diff_bits, max_diff_bits,
                          sizeof(host_max_diff_bits), cudaMemcpyDeviceToHost,
                          stream);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return fail_with_cleanup(err);
  out->sync_calls += 1;
  out->kernel_launches += 1;
  out->mismatch_count = host_mismatches;
  float max_abs_diff = 0.0f;
  memcpy(&max_abs_diff, &host_max_diff_bits, sizeof(max_abs_diff));
  out->max_abs_diff = max_abs_diff;
  out->hot_path_allocations = 0;
  out->status = 0;

  cleanup();
  return out->status == 0 ? 0 : -1;
}
