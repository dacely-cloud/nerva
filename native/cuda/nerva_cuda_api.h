#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NervaCudaDeviceSmokeResult {
  int32_t status;
  int32_t cuda_error;
  uint32_t value;
  int32_t device_count;
  int32_t device_ordinal;
  int32_t driver_version;
  int32_t runtime_version;
  int32_t compute_capability_major;
  int32_t compute_capability_minor;
  uint64_t total_global_mem;
  char gpu_name[128];
  char pci_bus_id[32];
} NervaCudaDeviceSmokeResult;

int nerva_cuda_device_smoke(NervaCudaDeviceSmokeResult *out);

#ifdef __cplusplus
}
#endif
