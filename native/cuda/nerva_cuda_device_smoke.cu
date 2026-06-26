#include "nerva_cuda_api.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kSmokeWord = 0x4e455256u;

__global__ void nerva_runtime_smoke_kernel(uint32_t *out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = kSmokeWord;
  }
}

void clear_result(NervaCudaDeviceSmokeResult *out) {
  out->status = -1;
  out->cuda_error = 0;
  out->value = 0;
  out->device_count = 0;
  out->device_ordinal = -1;
  out->driver_version = 0;
  out->runtime_version = 0;
  out->compute_capability_major = 0;
  out->compute_capability_minor = 0;
  out->posix_fd_handle_supported = -1;
  out->gpu_direct_rdma_supported = -1;
  out->gpu_direct_rdma_with_cuda_vmm_supported = -1;
  out->total_global_mem = 0;
  out->gpu_name[0] = '\0';
  out->pci_bus_id[0] = '\0';
}

int fail(NervaCudaDeviceSmokeResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

void record_driver_attribute(
    int32_t *out,
    CUdevice device,
    CUdevice_attribute attribute) {
  int value = 0;
  CUresult result = cuDeviceGetAttribute(&value, attribute, device);
  if (result == CUDA_SUCCESS) {
    *out = value != 0 ? 1 : 0;
  }
}

}  // namespace

extern "C" int nerva_cuda_device_smoke(NervaCudaDeviceSmokeResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);

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
  out->total_global_mem = static_cast<uint64_t>(props.totalGlobalMem);
  strncpy(out->gpu_name, props.name, sizeof(out->gpu_name) - 1);
  out->gpu_name[sizeof(out->gpu_name) - 1] = '\0';

  int driver_version = 0;
  err = cudaDriverGetVersion(&driver_version);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  out->driver_version = driver_version;

  int runtime_version = 0;
  err = cudaRuntimeGetVersion(&runtime_version);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  out->runtime_version = runtime_version;

  CUdevice driver_device = 0;
  if (cuInit(0) == CUDA_SUCCESS &&
      cuDeviceGet(&driver_device, out->device_ordinal) == CUDA_SUCCESS) {
    record_driver_attribute(
        &out->posix_fd_handle_supported,
        driver_device,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED);
    record_driver_attribute(
        &out->gpu_direct_rdma_supported,
        driver_device,
        CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED);
    record_driver_attribute(
        &out->gpu_direct_rdma_with_cuda_vmm_supported,
        driver_device,
        CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED);
  }

  char pci_bus_id[32]{};
  err = cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), out->device_ordinal);
  if (err == cudaSuccess) {
    strncpy(out->pci_bus_id, pci_bus_id, sizeof(out->pci_bus_id) - 1);
    out->pci_bus_id[sizeof(out->pci_bus_id) - 1] = '\0';
  } else {
    out->pci_bus_id[0] = '\0';
  }

  uint32_t *device_word = nullptr;
  err = cudaMalloc(reinterpret_cast<void **>(&device_word), sizeof(uint32_t));
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  uint32_t *host_word = nullptr;
  err = cudaHostAlloc(reinterpret_cast<void **>(&host_word), sizeof(uint32_t), cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFree(device_word);
    return fail(out, err);
  }
  *host_word = 0;

  cudaStream_t stream = nullptr;
  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    cudaFreeHost(host_word);
    cudaFree(device_word);
    return fail(out, err);
  }

  nerva_runtime_smoke_kernel<<<1, 1, 0, stream>>>(device_word);
  err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_word, device_word, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }

  if (err == cudaSuccess) {
    out->value = *host_word;
    out->status = (*host_word == kSmokeWord) ? 0 : -1;
  } else {
    out->cuda_error = static_cast<int32_t>(err);
    out->status = -1;
  }

  cudaStreamDestroy(stream);
  cudaFreeHost(host_word);
  cudaFree(device_word);
  return out->status == 0 ? 0 : -1;
}
