#include <stdint.h>

extern "C" __global__ void nerva_synthetic_decode_step(
    const uint32_t *input_token,
    uint32_t *output_token) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output_token = *input_token + 1u;
  }
}
