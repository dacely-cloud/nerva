#include <stdint.h>

extern "C" __global__ void nerva_smoke_kernel(uint32_t *out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = 0x4e455256u;
  }
}
