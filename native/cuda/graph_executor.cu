#include <stdint.h>

extern "C" __global__ void nerva_graph_executor_placeholder(uint32_t *out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = 1u;
  }
}
