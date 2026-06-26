#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NervaCudaSmokeResult {
  int32_t status;
  uint32_t value;
} NervaCudaSmokeResult;

int nerva_cuda_smoke(NervaCudaSmokeResult *out);

#ifdef __cplusplus
}
#endif
