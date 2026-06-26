use nerva_core::types::{DType, NervaError, Result};

pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 && mantissa != 0 {
        return ((bits >> 16) as u16) | 1;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    bits.wrapping_add(rounding_bias).wrapping_shr(16) as u16
}

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = u32::from(bits & 0x03ff);

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        let mut mantissa = frac;
        let mut exponent = -14i32;
        while (mantissa & 0x0400) == 0 {
            mantissa <<= 1;
            exponent -= 1;
        }
        mantissa &= 0x03ff;
        let f32_exp = u32::try_from(exponent + 127).unwrap_or(0);
        return f32::from_bits(sign | (f32_exp << 23) | (mantissa << 13));
    }

    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (frac << 13));
    }

    let f32_exp = u32::from(exp) + 112;
    f32::from_bits(sign | (f32_exp << 23) | (frac << 13))
}

pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }

    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        let mut half = (mantissa >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        let round_mask = round_bit - 1;
        let remainder = mantissa & (round_bit | round_mask);
        if remainder > round_bit || (remainder == round_bit && (half & 1) != 0) {
            half = half.saturating_add(1);
        }
        return sign | half;
    }

    let mut half = ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    let remainder = mant & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half & 1) != 0) {
        half = half.saturating_add(1);
    }
    sign | half
}

pub fn encode_f32_for_dtype(value: f32, dtype: DType) -> Result<u16> {
    match dtype {
        DType::F16 => Ok(f32_to_f16_bits(value)),
        DType::BF16 => Ok(f32_to_bf16_bits(value)),
        _ => Err(NervaError::InvalidArgument {
            reason: "precision block supports only FP16 and BF16".to_string(),
        }),
    }
}

pub fn decode_f32_for_dtype(bits: u16, dtype: DType) -> Result<f32> {
    match dtype {
        DType::F16 => Ok(f16_bits_to_f32(bits)),
        DType::BF16 => Ok(bf16_bits_to_f32(bits)),
        _ => Err(NervaError::InvalidArgument {
            reason: "precision block supports only FP16 and BF16".to_string(),
        }),
    }
}

pub fn dtype_label(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F16 => Ok("float16"),
        DType::BF16 => Ok("bfloat16"),
        _ => Err(NervaError::InvalidArgument {
            reason: "precision block supports only FP16 and BF16".to_string(),
        }),
    }
}

pub(crate) fn hash_u16s(values: &[u16]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}
