#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace nerva {
namespace deepseek {

__device__ __forceinline__ float e8m0_exponent_bits_to_f32(uint8_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 23);
}

__device__ __forceinline__ float f8_e4m3fn_bits_to_f32(uint8_t bits) {
  const float sign = (bits & 0x80u) ? -1.0f : 1.0f;
  const uint8_t exp = (bits >> 3) & 0x0fu;
  const uint8_t frac = bits & 0x07u;

  if (exp == 0) {
    if (frac == 0) {
      return sign * 0.0f;
    }
    return sign * ldexpf(static_cast<float>(frac) * 0.125f, -6);
  }
  if (exp == 0x0fu && frac == 0x07u) {
    return NAN;
  }
  return sign * ldexpf(1.0f + static_cast<float>(frac) * 0.125f,
                       static_cast<int>(exp) - 7);
}

__device__ __forceinline__ float mxfp4_e2m1_nibble_to_f32(uint8_t nibble) {
  constexpr float kTable[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  return kTable[nibble & 0x0fu];
}

}  // namespace deepseek
}  // namespace nerva
