use crate::deepseek_kv::ffi::{
    NervaCudaDeepSeekKvFp8DsMlaPackRequest, NervaCudaDeepSeekKvFp8DsMlaPackResult,
    run_deepseek_kv_fp8_ds_mla_pack, run_deepseek_v32_kv_fp8_ds_mla_pack,
};
use crate::deepseek_kv::summary::CudaDeepSeekKvSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

pub fn deepseek_fp8_ds_mla_pack(
    block_size: u32,
    token_index: u32,
    nope_fp8: &[u8],
    rope_bf16: &[u16],
    scales: &[u8],
) -> CudaDeepSeekKvSummary {
    let token_stride = rope_bf16
        .len()
        .checked_mul(2)
        .and_then(|rope_bytes| nope_fp8.len().checked_add(rope_bytes));
    let block_bytes = token_stride
        .and_then(|stride| stride.checked_add(scales.len()))
        .and_then(|token_bytes| (block_size as usize).checked_mul(token_bytes));
    if block_size == 0
        || token_index >= block_size
        || nope_fp8.is_empty()
        || rope_bf16.is_empty()
        || scales.is_empty()
        || token_stride.is_none_or(|stride| stride > u32::MAX as usize)
        || block_bytes.is_none_or(|bytes| bytes == 0)
    {
        return failed_summary(
            block_size,
            token_index,
            token_stride
                .filter(|stride| *stride <= u32::MAX as usize)
                .unwrap_or(0) as u32,
            scales.len() as u32,
            block_bytes.unwrap_or(0) as u64,
            Vec::new(),
            "invalid DeepSeek fp8_ds_mla pack shape",
        );
    }
    let block_bytes = block_bytes.expect("validated block bytes");

    let mut output = vec![0u8; block_bytes];
    let request = NervaCudaDeepSeekKvFp8DsMlaPackRequest {
        block_size,
        token_index,
        nope_bytes: nope_fp8.len() as u32,
        rope_bf16_values: rope_bf16.len() as u32,
        scale_dim: scales.len() as u32,
        nope_fp8: nope_fp8.as_ptr(),
        rope_bf16: rope_bf16.as_ptr(),
        scales: scales.as_ptr(),
        output_block: output.as_mut_ptr(),
    };
    let mut out = NervaCudaDeepSeekKvFp8DsMlaPackResult::default();
    let return_code = run_deepseek_kv_fp8_ds_mla_pack(&request, &mut out);
    summarize(return_code, out, output)
}

pub fn deepseek_v32_fp8_ds_mla_pack(
    token_index: u32,
    nope_fp8: &[u8],
    rope_bf16: &[u16],
    scales_f32_bytes: &[u8],
) -> CudaDeepSeekKvSummary {
    const BLOCK_SIZE: u32 = 64;
    const NOPE_BYTES: usize = 512;
    const ROPE_VALUES: usize = 64;
    const SCALE_BYTES: usize = 16;
    const TOKEN_STRIDE: usize = NOPE_BYTES + SCALE_BYTES + ROPE_VALUES * 2;
    let block_bytes = BLOCK_SIZE as usize * TOKEN_STRIDE;
    if token_index >= BLOCK_SIZE
        || nope_fp8.len() != NOPE_BYTES
        || rope_bf16.len() != ROPE_VALUES
        || scales_f32_bytes.len() != SCALE_BYTES
    {
        return failed_summary(
            BLOCK_SIZE,
            token_index,
            TOKEN_STRIDE as u32,
            SCALE_BYTES as u32,
            block_bytes as u64,
            Vec::new(),
            "invalid DeepSeek V3.2 fp8_ds_mla pack shape",
        );
    }

    let mut output = vec![0u8; block_bytes];
    let request = NervaCudaDeepSeekKvFp8DsMlaPackRequest {
        block_size: BLOCK_SIZE,
        token_index,
        nope_bytes: NOPE_BYTES as u32,
        rope_bf16_values: ROPE_VALUES as u32,
        scale_dim: SCALE_BYTES as u32,
        nope_fp8: nope_fp8.as_ptr(),
        rope_bf16: rope_bf16.as_ptr(),
        scales: scales_f32_bytes.as_ptr(),
        output_block: output.as_mut_ptr(),
    };
    let mut out = NervaCudaDeepSeekKvFp8DsMlaPackResult::default();
    let return_code = run_deepseek_v32_kv_fp8_ds_mla_pack(&request, &mut out);
    summarize(return_code, out, output)
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekKvFp8DsMlaPackResult,
    output: Vec<u8>,
) -> CudaDeepSeekKvSummary {
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
            "CUDA DeepSeek fp8_ds_mla KV pack failed: return_code={} status={} cuda_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.device_count
        ))
    };
    CudaDeepSeekKvSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        block_size: out.block_size,
        token_index: out.token_index,
        token_stride: out.token_stride,
        scale_dim: out.scale_dim,
        block_bytes: out.block_bytes,
        output_hash: out.output_hash,
        output,
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

fn failed_summary(
    block_size: u32,
    token_index: u32,
    token_stride: u32,
    scale_dim: u32,
    block_bytes: u64,
    output: Vec<u8>,
    error: impl Into<String>,
) -> CudaDeepSeekKvSummary {
    CudaDeepSeekKvSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        block_size,
        token_index,
        token_stride,
        scale_dim,
        block_bytes,
        output_hash: 0,
        output,
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
