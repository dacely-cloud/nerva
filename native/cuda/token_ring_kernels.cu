#include <stdint.h>

extern "C" __global__ void nerva_token_ring_write_u32(
    uint32_t *ring,
    uint32_t capacity,
    uint32_t index,
    uint32_t token) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && capacity != 0) {
    ring[index % capacity] = token;
  }
}
