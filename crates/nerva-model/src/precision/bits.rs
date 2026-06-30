use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

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

pub fn f8_e4m3fn_bits_to_f32(bits: u8) -> f32 {
    let sign = if (bits & 0x80) != 0 { -1.0 } else { 1.0 };
    let exp = (bits >> 3) & 0x0f;
    let frac = bits & 0x07;

    if exp == 0 {
        if frac == 0 {
            return sign * 0.0;
        }
        return sign * f32::from(frac) * (1.0 / 8.0) * 2f32.powi(-6);
    }

    if exp == 0x0f && frac == 0x07 {
        return f32::NAN;
    }

    sign * (1.0 + f32::from(frac) * (1.0 / 8.0)) * 2f32.powi(i32::from(exp) - 7)
}

pub fn e8m0_exponent_bits_to_f32(bits: u8) -> f32 {
    f32::from_bits(u32::from(bits) << 23)
}

pub fn mxfp4_e2m1_nibble_to_f32(nibble: u8) -> f32 {
    match nibble & 0x0f {
        0x0 => 0.0,
        0x1 => 0.5,
        0x2 => 1.0,
        0x3 => 1.5,
        0x4 => 2.0,
        0x5 => 3.0,
        0x6 => 4.0,
        0x7 => 6.0,
        0x8 => -0.0,
        0x9 => -0.5,
        0x0a => -1.0,
        0x0b => -1.5,
        0x0c => -2.0,
        0x0d => -3.0,
        0x0e => -4.0,
        _ => -6.0,
    }
}

pub fn dequantize_f8_e4m3fn_block_scaled_into(
    weights: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    block_rows: usize,
    block_cols: usize,
    output: &mut [f32],
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "FP8 block dequantization requires non-empty matrix dimensions".to_string(),
        });
    }
    if block_rows == 0 || block_cols == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "FP8 block dequantization requires non-zero block dimensions".to_string(),
        });
    }

    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "FP8 block dequantization matrix dimensions overflow".to_string(),
        })?;
    if weights.len() != elements {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "FP8 block dequantization expected {elements} weight bytes, got {}",
                weights.len()
            ),
        });
    }
    if output.len() != elements {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "FP8 block dequantization expected {elements} output values, got {}",
                output.len()
            ),
        });
    }

    let scale_rows = div_ceil_usize(rows, block_rows);
    let scale_cols = div_ceil_usize(cols, block_cols);
    let scale_elements =
        scale_rows
            .checked_mul(scale_cols)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "FP8 block dequantization scale dimensions overflow".to_string(),
            })?;
    if scales.len() != scale_elements {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "FP8 block dequantization expected {scale_elements} E8M0 scale bytes, got {}",
                scales.len()
            ),
        });
    }

    for row in 0..rows {
        let scale_row = row / block_rows;
        for col in 0..cols {
            let value_idx = row * cols + col;
            let scale_col = col / block_cols;
            let scale_idx = scale_row * scale_cols + scale_col;
            output[value_idx] = f8_e4m3fn_bits_to_f32(weights[value_idx])
                * e8m0_exponent_bits_to_f32(scales[scale_idx]);
        }
    }
    Ok(())
}

pub fn dequantize_f8_e4m3fn_block_scaled(
    weights: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    block_rows: usize,
    block_cols: usize,
) -> Result<Vec<f32>> {
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "FP8 block dequantization matrix dimensions overflow".to_string(),
        })?;
    let mut output = vec![0.0; elements];
    dequantize_f8_e4m3fn_block_scaled_into(
        weights,
        scales,
        rows,
        cols,
        block_rows,
        block_cols,
        &mut output,
    )?;
    Ok(output)
}

pub fn dequantize_mxfp4_e2m1_block_scaled_into(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    packed_cols: usize,
    scale_packed_cols: usize,
    output: &mut [f32],
) -> Result<()> {
    if rows == 0 || packed_cols == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "MXFP4 block dequantization requires non-empty matrix dimensions".to_string(),
        });
    }
    if scale_packed_cols == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "MXFP4 block dequantization requires non-zero scale block columns".to_string(),
        });
    }

    let packed_elements =
        rows.checked_mul(packed_cols)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "MXFP4 block dequantization packed dimensions overflow".to_string(),
            })?;
    if packed.len() != packed_elements {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "MXFP4 block dequantization expected {packed_elements} packed bytes, got {}",
                packed.len()
            ),
        });
    }

    let output_elements =
        packed_elements
            .checked_mul(2)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "MXFP4 block dequantization output dimensions overflow".to_string(),
            })?;
    if output.len() != output_elements {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "MXFP4 block dequantization expected {output_elements} output values, got {}",
                output.len()
            ),
        });
    }

    let scale_cols = div_ceil_usize(packed_cols, scale_packed_cols);
    let scale_elements =
        rows.checked_mul(scale_cols)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "MXFP4 block dequantization scale dimensions overflow".to_string(),
            })?;
    if scales.len() != scale_elements {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "MXFP4 block dequantization expected {scale_elements} E8M0 scale bytes, got {}",
                scales.len()
            ),
        });
    }

    for row in 0..rows {
        for packed_col in 0..packed_cols {
            let packed_idx = row * packed_cols + packed_col;
            let scale_idx = row * scale_cols + packed_col / scale_packed_cols;
            let scale = e8m0_exponent_bits_to_f32(scales[scale_idx]);
            let byte = packed[packed_idx];
            let out_idx = packed_idx * 2;
            output[out_idx] = mxfp4_e2m1_nibble_to_f32(byte & 0x0f) * scale;
            output[out_idx + 1] = mxfp4_e2m1_nibble_to_f32(byte >> 4) * scale;
        }
    }

    Ok(())
}

pub fn dequantize_mxfp4_e2m1_block_scaled(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    packed_cols: usize,
    scale_packed_cols: usize,
) -> Result<Vec<f32>> {
    let output_elements = rows
        .checked_mul(packed_cols)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "MXFP4 block dequantization output dimensions overflow".to_string(),
        })?;
    let mut output = vec![0.0; output_elements];
    dequantize_mxfp4_e2m1_block_scaled_into(
        packed,
        scales,
        rows,
        packed_cols,
        scale_packed_cols,
        &mut output,
    )?;
    Ok(output)
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

fn div_ceil_usize(value: usize, divisor: usize) -> usize {
    value / divisor + usize::from(value % divisor != 0)
}
