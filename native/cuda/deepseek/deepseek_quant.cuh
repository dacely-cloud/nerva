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
  const uint32_t sign = static_cast<uint32_t>(bits & 0x80u) << 24;
  const uint8_t exp = (bits >> 3) & 0x0fu;
  const uint8_t frac = bits & 0x07u;

  if (exp == 0) {
    if (frac == 0) {
      return __uint_as_float(sign);
    }
    const float value = static_cast<float>(frac) * 0.001953125f;
    return sign == 0 ? value : -value;
  }
  if (exp == 0x0fu && frac == 0x07u) {
    return __uint_as_float(0x7fffffffu);
  }
  return __uint_as_float(sign | ((static_cast<uint32_t>(exp) + 120u) << 23) |
                         (static_cast<uint32_t>(frac) << 20));
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
