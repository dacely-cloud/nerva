#include "nerva_cuda_api.h"

int nerva_cuda_smoke(NervaCudaSmokeResult *out) {
  if (out == nullptr) {
    return -1;
  }
  out->status = 0;
  out->value = 0x4e455256u;
  return 0;
}
