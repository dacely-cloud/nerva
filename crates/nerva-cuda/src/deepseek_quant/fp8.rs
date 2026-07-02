//! Host-side mirror of the device f32 -> fp8 e4m3fn conversion.
//!
//! Must stay bit-for-bit identical to `f32_to_f8_e4m3fn_bits` in
//! `native/cuda/deepseek/deepseek_quant.cuh`, which uses the hardware
//! `__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3)` conversion:
//! round-to-nearest-even with saturation to the largest finite value.

/// Converts an `f32` to fp8 e4m3fn bits with round-to-nearest-even and
/// saturate-to-finite semantics:
///
/// - NaN (any payload) -> `0x7f`
/// - sign is preserved for all values, including zeros (`-0.0` -> `0x80`)
/// - |x| > 448.0 (including infinities) -> `sign | 0x7e`
/// - normals: `(1 + m/8) * 2^(e - 7)` for exponent field `e` in `1..=15`
/// - subnormals: `(m/8) * 2^-6` (step `2^-9`) for `m` in `1..=7`
pub(crate) fn f32_to_f8_e4m3fn_bits(value: f32) -> u8 {
    let bits = value.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    let abs_bits = bits & 0x7fff_ffff;

    // NaN (any payload) maps to the canonical e4m3fn NaN.
    if abs_bits > 0x7f80_0000 {
        return 0x7f;
    }

    let abs = f32::from_bits(abs_bits);

    // Saturate-to-finite: everything above the largest representable
    // magnitude (448.0 = 0x7e), including infinity, clamps to it. Values in
    // (448, 464] would also round to 448 under round-to-nearest-even because
    // 464 is the tie against the non-existent 480 and 448 has an even
    // mantissa, so this branch matches __NV_SATFINITE exactly.
    if abs > 448.0 {
        return sign | 0x7e;
    }

    // Subnormal target range: |x| < 2^-6 rounds on the subnormal grid with
    // step 2^-9. The scaling by 512 = 2^9 is exact (power-of-two), so
    // round_ties_even lands on the exact nearest-even quantum in 0..=8.
    // A quantum of 8 is exactly the smallest normal (0x08), so the plain
    // bit-or below is correct for the whole 0..=8 range.
    if abs < 0.015625 {
        let quantum = (abs * 512.0).round_ties_even() as u32;
        return sign | quantum as u8;
    }

    // Normal range: exponent is at least -6 here, so the fp8 exponent field
    // (biased by 7) is at least 1.
    let mut exp32 = ((abs_bits >> 23) as i32) - 127;
    let mant = abs_bits & 0x007f_ffff;
    // Keep the top 3 mantissa bits, round-to-nearest-even on the remainder.
    let mut q = mant >> 20;
    let r = mant & 0x000f_ffff;
    if r > 0x8_0000 || (r == 0x8_0000 && (q & 1) == 1) {
        q += 1;
        if q == 8 {
            // Mantissa overflow carries into the exponent. This cannot
            // overflow past 448 because abs <= 448 in this branch.
            q = 0;
            exp32 += 1;
        }
    }
    let fp8 = (((exp32 + 7) as u32) << 3) | q;
    sign | fp8 as u8
}
