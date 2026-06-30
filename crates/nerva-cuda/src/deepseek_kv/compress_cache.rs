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

pub fn deepseek_compress_norm_rope_fp8_cache(
    input: CudaDeepSeekCompressNormRopeFp8CacheInput<'_>,
) -> CudaDeepSeekCompressNormRopeFp8CacheSummary {
    let num_tokens = input.positions.len();
    let state_values = (input.num_state_blocks as usize)
        .checked_mul(input.state_block_size as usize)
        .and_then(|value| value.checked_mul(input.state_width as usize))
        .and_then(|value| value.checked_mul(2))
        .unwrap_or(usize::MAX);
    let block_table_values = (input.num_reqs as usize)
        .checked_mul(input.block_table_stride as usize)
        .unwrap_or(usize::MAX);
    let kv_cache_bytes = (input.num_kv_blocks as usize)
        .checked_mul(input.kv_cache_block_stride as usize)
        .unwrap_or(usize::MAX);
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
        .unwrap_or(usize::MAX);
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
        || state_values == usize::MAX
        || block_table_values == usize::MAX
        || kv_cache_bytes == usize::MAX
        || cos_sin_values == usize::MAX
        || kv_cache_bytes > u32::MAX as usize
        || !scale_layout_valid
        || input.kv_cache_block_stride
            < input
                .kv_cache_block_size
                .saturating_mul(input.token_stride.saturating_add(input.scale_dim))
    {
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
    }

    let mut kv_cache = vec![0u8; kv_cache_bytes];
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
