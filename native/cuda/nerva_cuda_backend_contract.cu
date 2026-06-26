#include "nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kBackendPatternWord = 0x5a5a5a5au;

void clear_result(NervaCudaBackendContractResult *out) {
  out->status = -1;
  out->cuda_error = 0;
  out->device_count = 0;
  out->device_ordinal = -1;
  out->driver_version = 0;
  out->runtime_version = 0;
  out->compute_capability_major = 0;
  out->compute_capability_minor = 0;
  out->total_global_mem = 0;
  out->requested_device_bytes = 0;
  out->requested_pinned_bytes = 0;
  out->allocated_device_bytes = 0;
  out->allocated_pinned_bytes = 0;
  out->stream_creations = 0;
  out->stream_destroys = 0;
  out->event_creations = 0;
  out->event_destroys = 0;
  out->device_allocations = 0;
  out->device_frees = 0;
  out->pinned_allocations = 0;
  out->pinned_frees = 0;
  out->memset_bytes = 0;
  out->d2h_bytes = 0;
  out->sync_calls = 0;
  out->observed_word = 0;
  out->hot_path_allocations = 0;
  out->gpu_name[0] = '\0';
  out->pci_bus_id[0] = '\0';
}

int fail(NervaCudaBackendContractResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

void record_cleanup_error(NervaCudaBackendContractResult *out, cudaError_t err) {
  if (err != cudaSuccess && out->cuda_error == 0) {
    out->cuda_error = static_cast<int32_t>(err);
    out->status = -1;
  }
}

}  // namespace

extern "C" int nerva_cuda_backend_contract_smoke(
    NervaCudaBackendContractResult *out,
    uint64_t device_bytes,
    uint64_t pinned_bytes) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);
  out->requested_device_bytes = device_bytes;
  out->requested_pinned_bytes = pinned_bytes;

  if (device_bytes < sizeof(uint32_t) || pinned_bytes < sizeof(uint32_t)) {
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

  char pci_bus_id[32]{};
  err = cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), out->device_ordinal);
  if (err == cudaSuccess) {
    strncpy(out->pci_bus_id, pci_bus_id, sizeof(out->pci_bus_id) - 1);
    out->pci_bus_id[sizeof(out->pci_bus_id) - 1] = '\0';
  } else {
    out->pci_bus_id[0] = '\0';
  }

  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
  void *device_ptr = nullptr;
  void *pinned_ptr = nullptr;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  out->stream_creations = 1;

  err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  if (err != cudaSuccess) {
    cudaError_t cleanup = cudaStreamDestroy(stream);
    if (cleanup == cudaSuccess) {
      out->stream_destroys = 1;
    }
    return fail(out, err);
  }
  out->event_creations = 1;

  err = cudaMalloc(&device_ptr, static_cast<size_t>(device_bytes));
  if (err != cudaSuccess) {
    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
    return fail(out, err);
  }
  out->device_allocations = 1;
  out->allocated_device_bytes = device_bytes;

  err = cudaHostAlloc(&pinned_ptr, static_cast<size_t>(pinned_bytes), cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cudaFree(device_ptr);
    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
    return fail(out, err);
  }
  out->pinned_allocations = 1;
  out->allocated_pinned_bytes = pinned_bytes;
  memset(pinned_ptr, 0, static_cast<size_t>(pinned_bytes));

  err = cudaMemsetAsync(device_ptr, 0x5a, static_cast<size_t>(device_bytes), stream);
  if (err == cudaSuccess) {
    out->memset_bytes = device_bytes;
    err = cudaMemcpyAsync(
        pinned_ptr,
        device_ptr,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost,
        stream);
  }
  if (err == cudaSuccess) {
    out->d2h_bytes = sizeof(uint32_t);
    err = cudaEventRecord(event, stream);
  }
  if (err == cudaSuccess) {
    err = cudaEventSynchronize(event);
    out->sync_calls = 1;
  }
  if (err == cudaSuccess) {
    const uint32_t observed = *reinterpret_cast<uint32_t *>(pinned_ptr);
    out->observed_word = observed;
    out->status = (observed == kBackendPatternWord) ? 0 : -1;
  } else {
    out->cuda_error = static_cast<int32_t>(err);
    out->status = -1;
  }

  cudaError_t cleanup = cudaFreeHost(pinned_ptr);
  if (cleanup == cudaSuccess) {
    out->pinned_frees = 1;
  }
  record_cleanup_error(out, cleanup);

  cleanup = cudaFree(device_ptr);
  if (cleanup == cudaSuccess) {
    out->device_frees = 1;
  }
  record_cleanup_error(out, cleanup);

  cleanup = cudaEventDestroy(event);
  if (cleanup == cudaSuccess) {
    out->event_destroys = 1;
  }
  record_cleanup_error(out, cleanup);

  cleanup = cudaStreamDestroy(stream);
  if (cleanup == cudaSuccess) {
    out->stream_destroys = 1;
  }
  record_cleanup_error(out, cleanup);

  return out->status == 0 ? 0 : -1;
}
