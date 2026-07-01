use crate::deepseek_kv::ffi::{
    NervaCudaDeepSeekCompressNormRopeFp8CacheRequest,
    NervaCudaDeepSeekCompressNormRopeFp8CacheResult, run_deepseek_compress_norm_rope_fp8_cache,
};
use crate::deepseek_kv::summary::CudaDeepSeekCompressNormRopeFp8CacheSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

pub const DEEPSEEK_COMPRESS_SCALE_E8M0: u32 = 0;
pub const DEEPSEEK_COMPRESS_SCALE_F32: u32 = 1;
pub const DEEPSEEK_COMPRESS_SCALE_MXFP4: u32 = 2;

#[derive(Clone, Debug)]
pub struct CudaDeepSeekCompressNormRopeFp8CacheInput<'a> {
    pub state_cache: &'a [f32],
    pub token_to_req_indices: &'a [i32],
    pub positions: &'a [i64],
    pub slot_mapping: &'a [i64],
    pub block_table: &'a [i32],
    pub kv_slot_mapping: &'a [i64],
    pub rms_norm_weight: &'a [f32],
    pub cos_sin_cache: &'a [f32],
    pub num_reqs: u32,
    pub block_table_stride: u32,
    pub state_block_size: u32,
    pub kv_cache_block_size: u32,
    pub head_size: u32,
    pub state_width: u32,
    pub rope_head_dim: u32,
    pub compress_ratio: u32,
    pub overlap: u32,
    pub quant_block: u32,
    pub token_stride: u32,
    pub scale_dim: u32,
    pub scale_format: u32,
    pub num_state_blocks: u32,
    pub num_kv_blocks: u32,
    pub kv_cache_block_stride: u32,
    pub cos_sin_stride: u32,
    pub rms_norm_eps: f32,
    pub fp8_max: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekCompressNormRopeFp8CacheReference {
    pub kv_cache: Vec<u8>,
    pub written_tokens: u32,
    pub skipped_tokens: u32,
}

pub fn deepseek_compress_norm_rope_fp8_cache_reference(
    input: &CudaDeepSeekCompressNormRopeFp8CacheInput<'_>,
) -> Result<DeepSeekCompressNormRopeFp8CacheReference, String> {
    let shape = validate_compress_norm_rope_fp8_cache_shape(input)?;
    let mut kv_cache = vec![0u8; shape.kv_cache_bytes];
    let mut written_tokens = 0u32;
    let mut skipped_tokens = 0u32;

    for token_idx in 0..shape.num_tokens {
        let position = input.positions[token_idx];
        let req_idx = input.token_to_req_indices[token_idx];
        let kv_slot_idx = input.kv_slot_mapping[token_idx];
        let valid = input.slot_mapping[token_idx] >= 0
            && position >= 0
            && (position + 1) % input.compress_ratio as i64 == 0
            && req_idx >= 0
            && (req_idx as usize) < input.num_reqs as usize
            && kv_slot_idx >= 0
            && (kv_slot_idx as u64)
                < u64::from(input.num_kv_blocks) * u64::from(input.kv_cache_block_size);
        if !valid {
            skipped_tokens += 1;
            continue;
        }

        let compressed = reference_compressed_state(input, token_idx);
        let variance =
            compressed.iter().map(|value| value * value).sum::<f32>() / input.head_size as f32;
        let rrms = 1.0 / (variance + input.rms_norm_eps).sqrt();
        let normed = compressed
            .iter()
            .zip(input.rms_norm_weight.iter())
            .map(|(value, weight)| value * rrms * weight)
            .collect::<Vec<_>>();
        let rotated = reference_rope(input, position, &normed);

        let kv_slot = kv_slot_idx as usize;
        let kv_block = kv_slot / input.kv_cache_block_size as usize;
        let kv_pos = kv_slot % input.kv_cache_block_size as usize;
        let block_base = kv_block * input.kv_cache_block_stride as usize;
        let data_base = block_base + kv_pos * input.token_stride as usize;
        let scale_base = block_base
            + input.kv_cache_block_size as usize * input.token_stride as usize
            + kv_pos * input.scale_dim as usize;

        match input.scale_format {
            DEEPSEEK_COMPRESS_SCALE_E8M0 => {
                write_reference_e8m0_cache(
                    input,
                    &mut kv_cache,
                    data_base,
                    scale_base,
                    &normed,
                    &rotated,
                );
            }
            DEEPSEEK_COMPRESS_SCALE_F32 => {
                write_reference_f32_scale_cache(
                    input,
                    &mut kv_cache,
                    data_base,
                    scale_base,
                    &rotated,
                );
            }
            DEEPSEEK_COMPRESS_SCALE_MXFP4 => {
                write_reference_mxfp4_cache(input, &mut kv_cache, data_base, scale_base, &rotated);
            }
            _ => unreachable!("scale format is validated"),
        }
        written_tokens += 1;
    }

    Ok(DeepSeekCompressNormRopeFp8CacheReference {
        kv_cache,
        written_tokens,
        skipped_tokens,
    })
}

pub fn deepseek_compress_norm_rope_fp8_cache(
    input: CudaDeepSeekCompressNormRopeFp8CacheInput<'_>,
) -> CudaDeepSeekCompressNormRopeFp8CacheSummary {
    let num_tokens = input.positions.len();
    let Ok(shape) = validate_compress_norm_rope_fp8_cache_shape(&input) else {
        return failed_summary(
            num_tokens as u32,
            input.head_size,
            input.rope_head_dim,
            input.compress_ratio,
            input.quant_block,
            input.token_stride,
            input.scale_dim,
            input.scale_format,
            Vec::new(),
            "invalid DeepSeek fused compress/norm/RoPE FP8 cache shape",
        );
    };

    let mut kv_cache = vec![0u8; shape.kv_cache_bytes];
    let request = NervaCudaDeepSeekCompressNormRopeFp8CacheRequest {
        num_tokens: num_tokens as u32,
        num_reqs: input.num_reqs,
        block_table_stride: input.block_table_stride,
        state_block_size: input.state_block_size,
        kv_cache_block_size: input.kv_cache_block_size,
        head_size: input.head_size,
        state_width: input.state_width,
        rope_head_dim: input.rope_head_dim,
        compress_ratio: input.compress_ratio,
        overlap: input.overlap,
        quant_block: input.quant_block,
        token_stride: input.token_stride,
        scale_dim: input.scale_dim,
        scale_format: input.scale_format,
        num_state_blocks: input.num_state_blocks,
        num_kv_blocks: input.num_kv_blocks,
        kv_cache_block_stride: input.kv_cache_block_stride,
        cos_sin_stride: input.cos_sin_stride,
        cos_sin_values: input.cos_sin_cache.len() as u32,
        rms_norm_eps: input.rms_norm_eps,
        fp8_max: input.fp8_max,
        state_cache: input.state_cache.as_ptr(),
        token_to_req_indices: input.token_to_req_indices.as_ptr(),
        positions: input.positions.as_ptr(),
        slot_mapping: input.slot_mapping.as_ptr(),
        block_table: input.block_table.as_ptr(),
        kv_slot_mapping: input.kv_slot_mapping.as_ptr(),
        rms_norm_weight: input.rms_norm_weight.as_ptr(),
        cos_sin_cache: input.cos_sin_cache.as_ptr(),
        kv_cache: kv_cache.as_mut_ptr(),
    };
    let mut out = NervaCudaDeepSeekCompressNormRopeFp8CacheResult::default();
    let return_code = run_deepseek_compress_norm_rope_fp8_cache(&request, &mut out);
    summarize(return_code, out, kv_cache)
}

#[derive(Clone, Copy)]
struct CompressNormRopeFp8CacheShape {
    num_tokens: usize,
    kv_cache_bytes: usize,
}

fn validate_compress_norm_rope_fp8_cache_shape(
    input: &CudaDeepSeekCompressNormRopeFp8CacheInput<'_>,
) -> Result<CompressNormRopeFp8CacheShape, String> {
    let num_tokens = input.positions.len();
    let state_values = (input.num_state_blocks as usize)
        .checked_mul(input.state_block_size as usize)
        .and_then(|value| value.checked_mul(input.state_width as usize))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| "DeepSeek fused compress/norm/RoPE state shape overflow".to_string())?;
    let block_table_values = (input.num_reqs as usize)
        .checked_mul(input.block_table_stride as usize)
        .ok_or_else(|| {
            "DeepSeek fused compress/norm/RoPE block table shape overflow".to_string()
        })?;
    let kv_cache_bytes = (input.num_kv_blocks as usize)
        .checked_mul(input.kv_cache_block_stride as usize)
        .ok_or_else(|| "DeepSeek fused compress/norm/RoPE KV shape overflow".to_string())?;
    let max_compressed_pos = input
        .positions
        .iter()
        .copied()
        .filter(|position| *position >= 0 && input.compress_ratio != 0)
        .map(|position| {
            (position as usize / input.compress_ratio as usize) * input.compress_ratio as usize
        })
        .max()
        .unwrap_or(0);
    let cos_sin_values = (max_compressed_pos + 1)
        .checked_mul(input.cos_sin_stride as usize)
        .ok_or_else(|| "DeepSeek fused compress/norm/RoPE cos/sin shape overflow".to_string())?;
    let nope_head_dim = input.head_size.saturating_sub(input.rope_head_dim);
    let scale_layout_valid = match input.scale_format {
        DEEPSEEK_COMPRESS_SCALE_E8M0 => {
            input.rope_head_dim > 0
                && input.token_stride == nope_head_dim + input.rope_head_dim.saturating_mul(2)
                && input.quant_block != 0
                && input.scale_dim >= nope_head_dim / input.quant_block + 1
        }
        DEEPSEEK_COMPRESS_SCALE_F32 => {
            input.token_stride == input.head_size && input.scale_dim == size_of::<f32>() as u32
        }
        DEEPSEEK_COMPRESS_SCALE_MXFP4 => {
            input.head_size == 128
                && input.rope_head_dim > 0
                && input.quant_block == 32
                && input.token_stride == input.head_size / 2
                && input.scale_dim == input.head_size / input.quant_block
        }
        _ => false,
    };
    if num_tokens == 0
        || num_tokens > u32::MAX as usize
        || input.num_reqs == 0
        || input.block_table_stride == 0
        || input.state_block_size == 0
        || input.kv_cache_block_size == 0
        || input.head_size == 0
        || input.head_size > 512
        || input.state_width < input.head_size.saturating_mul(1 + input.overlap)
        || input.rope_head_dim > input.head_size
        || input.rope_head_dim % 2 != 0
        || input.compress_ratio == 0
        || input.quant_block == 0
        || input.head_size % input.quant_block != 0
        || input.head_size / input.quant_block > 16
        || input.num_state_blocks == 0
        || input.num_kv_blocks == 0
        || input.kv_cache_block_stride == 0
        || input.cos_sin_stride < input.rope_head_dim
        || !input.rms_norm_eps.is_finite()
        || input.rms_norm_eps <= 0.0
        || !input.fp8_max.is_finite()
        || input.fp8_max <= 0.0
        || input.token_to_req_indices.len() != num_tokens
        || input.slot_mapping.len() != num_tokens
        || input.kv_slot_mapping.len() != num_tokens
        || input.rms_norm_weight.len() != input.head_size as usize
        || input.state_cache.len() != state_values
        || input.block_table.len() != block_table_values
        || input.cos_sin_cache.len() < cos_sin_values
        || input.cos_sin_cache.len() > u32::MAX as usize
        || kv_cache_bytes > u32::MAX as usize
        || !scale_layout_valid
        || input.kv_cache_block_stride
            < input
                .kv_cache_block_size
                .saturating_mul(input.token_stride.saturating_add(input.scale_dim))
    {
        return Err("invalid DeepSeek fused compress/norm/RoPE FP8 cache shape".to_string());
    }

    Ok(CompressNormRopeFp8CacheShape {
        num_tokens,
        kv_cache_bytes,
    })
}

fn reference_compressed_state(
    input: &CudaDeepSeekCompressNormRopeFp8CacheInput<'_>,
    token_idx: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; input.head_size as usize];
    let position = input.positions[token_idx];
    let req_idx = input.token_to_req_indices[token_idx] as usize;
    let window_tokens = (1 + input.overlap) * input.compress_ratio;
    let start = position - window_tokens as i64 + 1;
    let row_stride = input.state_width as usize * 2;
    let block_stride = input.state_block_size as usize * row_stride;
    for dim in 0..input.head_size as usize {
        let mut max_score = f32::NEG_INFINITY;
        for window in 0..window_tokens as usize {
            let Some(base) =
                reference_state_row_base(input, req_idx, start, window, block_stride, row_stride)
            else {
                continue;
            };
            max_score = max_score.max(input.state_cache[base + input.state_width as usize + dim]);
        }
        let mut weighted = 0.0f32;
        let mut denom = 0.0f32;
        for window in 0..window_tokens as usize {
            let Some(base) =
                reference_state_row_base(input, req_idx, start, window, block_stride, row_stride)
            else {
                continue;
            };
            let score = input.state_cache[base + input.state_width as usize + dim];
            let weight = (score - max_score).exp();
            weighted += input.state_cache[base + dim] * weight;
            denom += weight;
        }
        out[dim] = if denom > 0.0 { weighted / denom } else { 0.0 };
    }
    out
}

fn reference_state_row_base(
    input: &CudaDeepSeekCompressNormRopeFp8CacheInput<'_>,
    req_idx: usize,
    start: i64,
    window: usize,
    block_stride: usize,
    row_stride: usize,
) -> Option<usize> {
    let pos = start + window as i64;
    if pos < 0 {
        return None;
    }
    let block_index = pos as usize / input.state_block_size as usize;
    if block_index >= input.block_table_stride as usize {
        return None;
    }
    let block_number = input.block_table[req_idx * input.block_table_stride as usize + block_index];
    if block_number < 0 || block_number >= input.num_state_blocks as i32 {
        return None;
    }
    let block_offset = pos as usize % input.state_block_size as usize;
    let head_offset = if window as u32 >= input.compress_ratio {
        input.head_size as usize
    } else {
        0
    };
    Some(block_number as usize * block_stride + block_offset * row_stride + head_offset)
}

fn reference_rope(
    input: &CudaDeepSeekCompressNormRopeFp8CacheInput<'_>,
    position: i64,
    normed: &[f32],
) -> Vec<f32> {
    let mut rotated = normed.to_vec();
    let nope = (input.head_size - input.rope_head_dim) as usize;
    let half_rope = input.rope_head_dim as usize / 2;
    let compressed_pos =
        (position as usize / input.compress_ratio as usize) * input.compress_ratio as usize;
    let cs_base = compressed_pos * input.cos_sin_stride as usize;
    for pair in 0..half_rope {
        let base = nope + pair * 2;
        let even = normed[base];
        let odd = normed[base + 1];
        let cos = input.cos_sin_cache[cs_base + pair];
        let sin = input.cos_sin_cache[cs_base + half_rope + pair];
        rotated[base] = even * cos - odd * sin;
        rotated[base + 1] = odd * cos + even * sin;
    }
    rotated
}

fn write_reference_e8m0_cache(
    input: &CudaDeepSeekCompressNormRopeFp8CacheInput<'_>,
    kv_cache: &mut [u8],
    data_base: usize,
    scale_base: usize,
    normed: &[f32],
    rotated: &[f32],
) {
    let nope = (input.head_size - input.rope_head_dim) as usize;
    for block in 0..(input.head_size / input.quant_block) as usize {
        let start = block * input.quant_block as usize;
        let end = start + input.quant_block as usize;
        let absmax = normed[start..end]
            .iter()
            .copied()
            .map(|value| bf16_to_f32(f32_to_bf16_bits(value)).abs())
            .fold(0.0f32, f32::max);
        let scale = 2.0f32.powf(((absmax.max(1.0e-4) / input.fp8_max).log2()).ceil());
        if block < nope / input.quant_block as usize {
            kv_cache[scale_base + block] = encode_e8m0_scale(scale);
        }
        for dim in start..end.min(nope) {
            let quant_input = bf16_to_f32(f32_to_bf16_bits(normed[dim]));
            let scaled = (quant_input / scale).clamp(-input.fp8_max, input.fp8_max);
            kv_cache[data_base + dim] = f32_to_f8_e4m3fn_bits_nearest(scaled);
        }
    }
    kv_cache[scale_base + nope / input.quant_block as usize] = 0;
    for dim in nope..input.head_size as usize {
        let bits = f32_to_bf16_bits(rotated[dim]);
        let offset = data_base + nope + (dim - nope) * 2;
        kv_cache[offset] = (bits & 0xff) as u8;
        kv_cache[offset + 1] = (bits >> 8) as u8;
    }
}

fn write_reference_f32_scale_cache(
    input: &CudaDeepSeekCompressNormRopeFp8CacheInput<'_>,
    kv_cache: &mut [u8],
    data_base: usize,
    scale_base: usize,
    rotated: &[f32],
) {
    let bf16_rotated = rotated
        .iter()
        .copied()
        .map(|value| bf16_to_f32(f32_to_bf16_bits(value)))
        .collect::<Vec<_>>();
    let absmax = bf16_rotated
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    let scale = 2.0f32.powf(((absmax.max(1.0e-4) / input.fp8_max).log2()).ceil());
    kv_cache[scale_base..scale_base + size_of::<f32>()].copy_from_slice(&scale.to_ne_bytes());
    for (dim, value) in bf16_rotated.iter().copied().enumerate() {
        let scaled = (value / scale).clamp(-input.fp8_max, input.fp8_max);
        kv_cache[data_base + dim] = f32_to_f8_e4m3fn_bits_nearest(scaled);
    }
}

fn write_reference_mxfp4_cache(
    input: &CudaDeepSeekCompressNormRopeFp8CacheInput<'_>,
    kv_cache: &mut [u8],
    data_base: usize,
    scale_base: usize,
    rotated: &[f32],
) {
    let half_block = input.quant_block as usize / 2;
    for block in 0..(input.head_size / input.quant_block) as usize {
        let pair_start = block * half_block;
        let mut amax = 0.0f32;
        for pair in 0..half_block {
            let base = (pair_start + pair) * 2;
            let even = bf16_to_f32(f32_to_bf16_bits(rotated[base]));
            let odd = bf16_to_f32(f32_to_bf16_bits(rotated[base + 1]));
            amax = amax.max(even.abs()).max(odd.abs());
        }
        let exponent =
            ((amax.max(6.0 * f32::MIN_POSITIVE) / 6.0).log2().ceil()).clamp(-127.0, 127.0);
        let inv_scale = 2.0f32.powf(-exponent);
        kv_cache[scale_base + block] = (exponent as i32 + 127) as u8;
        for pair in 0..half_block {
            let base = (pair_start + pair) * 2;
            let even = bf16_to_f32(f32_to_bf16_bits(rotated[base]));
            let odd = bf16_to_f32(f32_to_bf16_bits(rotated[base + 1]));
            let lo = f32_to_mxfp4_e2m1_nibble_nearest(even * inv_scale);
            let hi = f32_to_mxfp4_e2m1_nibble_nearest(odd * inv_scale);
            kv_cache[data_base + block * half_block + pair] = (hi << 4) | (lo & 0x0f);
        }
    }
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn encode_e8m0_scale(scale: f32) -> u8 {
    (scale.log2().ceil() as i32 + 127).clamp(0, 255) as u8
}

fn f32_to_f8_e4m3fn_bits_nearest(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let mut best_bits = 0u8;
    let mut best_error = f32::INFINITY;
    for bits in 0u8..=254 {
        let candidate = f8_e4m3fn_bits_to_f32(bits);
        if candidate.is_nan() {
            continue;
        }
        let error = (candidate - value).abs();
        if error < best_error || (error == best_error && bits < best_bits) {
            best_error = error;
            best_bits = bits;
        }
    }
    best_bits
}

fn f8_e4m3fn_bits_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 3) & 0x0f;
    let frac = bits & 0x07;
    if exp == 0 {
        if frac == 0 {
            return sign * 0.0;
        }
        return sign * ((frac as f32) * 0.125) * 2.0f32.powi(-6);
    }
    if exp == 0x0f && frac == 0x07 {
        return f32::NAN;
    }
    sign * (1.0 + (frac as f32) * 0.125) * 2.0f32.powi(exp as i32 - 7)
}

fn f32_to_mxfp4_e2m1_nibble_nearest(value: f32) -> u8 {
    let mut best_bits = 0u8;
    let mut best_error = f32::INFINITY;
    for bits in 0u8..16 {
        let candidate = mxfp4_e2m1_nibble_to_f32(bits);
        let error = (candidate - value).abs();
        if error < best_error || (error == best_error && bits < best_bits) {
            best_error = error;
            best_bits = bits;
        }
    }
    best_bits
}

fn mxfp4_e2m1_nibble_to_f32(bits: u8) -> f32 {
    const TABLE: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    TABLE[(bits & 0x0f) as usize]
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekCompressNormRopeFp8CacheResult,
    kv_cache: Vec<u8>,
) -> CudaDeepSeekCompressNormRopeFp8CacheSummary {
    let status = if return_code == 0 && out.status == 0 {
        SmokeStatus::Ok
    } else if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        SmokeStatus::Unavailable
    } else {
        SmokeStatus::Failed
    };
    let error = if status == SmokeStatus::Ok {
        None
    } else {
        Some(format!(
            "CUDA DeepSeek fused compress/norm/RoPE FP8 cache failed: return_code={} status={} cuda_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.device_count
        ))
    };
    CudaDeepSeekCompressNormRopeFp8CacheSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        num_tokens: out.num_tokens,
        head_size: out.head_size,
        rope_head_dim: out.rope_head_dim,
        compress_ratio: out.compress_ratio,
        quant_block: out.quant_block,
        token_stride: out.token_stride,
        scale_dim: out.scale_dim,
        scale_format: out.scale_format,
        written_tokens: out.written_tokens,
        skipped_tokens: out.skipped_tokens,
        kv_cache_bytes: out.kv_cache_bytes,
        output_hash: out.output_hash,
        kv_cache,
        device_arena_bytes: out.device_arena_bytes,
        pinned_host_bytes: out.pinned_host_bytes,
        h2d_bytes: out.h2d_bytes,
        d2h_bytes: out.d2h_bytes,
        kernel_launches: out.kernel_launches,
        sync_calls: out.sync_calls,
        hot_path_allocations: out.hot_path_allocations,
        error,
    }
}

#[allow(clippy::too_many_arguments)]
fn failed_summary(
    num_tokens: u32,
    head_size: u32,
    rope_head_dim: u32,
    compress_ratio: u32,
    quant_block: u32,
    token_stride: u32,
    scale_dim: u32,
    scale_format: u32,
    kv_cache: Vec<u8>,
    error: impl Into<String>,
) -> CudaDeepSeekCompressNormRopeFp8CacheSummary {
    CudaDeepSeekCompressNormRopeFp8CacheSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        num_tokens,
        head_size,
        rope_head_dim,
        compress_ratio,
        quant_block,
        token_stride,
        scale_dim,
        scale_format,
        written_tokens: 0,
        skipped_tokens: 0,
        kv_cache_bytes: kv_cache.len() as u64,
        output_hash: 0,
        kv_cache,
        device_arena_bytes: 0,
        pinned_host_bytes: 0,
        h2d_bytes: 0,
        d2h_bytes: 0,
        kernel_launches: 0,
        sync_calls: 0,
        hot_path_allocations: 0,
        error: Some(error.into()),
    }
}
