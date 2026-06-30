use std::ptr;

use crate::deepseek_moe::ffi::{
    NervaCudaDeepSeekMegaMoePrepareRequest, NervaCudaDeepSeekMegaMoePrepareResult,
    run_deepseek_megamoe_prepare,
};
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekMegaMoePrepareSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub prepare_error: i32,
    pub num_tokens: u32,
    pub hidden_size: u32,
    pub top_k: u32,
    pub hidden_blocks: u32,
    pub x_fp8: Vec<u8>,
    pub x_scales: Vec<u32>,
    pub topk_ids: Vec<i64>,
    pub topk_weights: Vec<f32>,
    pub x_fp8_hash: u64,
    pub x_scales_hash: u64,
    pub topk_hash: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CudaDeepSeekMegaMoePrepareInput<'a> {
    pub num_tokens: u32,
    pub hidden_size: u32,
    pub top_k: u32,
    pub hidden_states: &'a [f32],
    pub topk_ids: &'a [i64],
    pub topk_weights: &'a [f32],
    pub is_padding: Option<&'a [u8]>,
}

pub fn deepseek_megamoe_prepare(
    input: CudaDeepSeekMegaMoePrepareInput<'_>,
) -> CudaDeepSeekMegaMoePrepareSummary {
    let output_shape = output_shape(&input);
    if !valid_shape(&input) {
        return failed_summary(
            input,
            output_shape,
            "invalid DeepSeek MegaMoE prepare shape",
        );
    }

    let (x_fp8_len, x_scales_len, topk_len) = output_shape;
    let mut x_fp8 = vec![0u8; x_fp8_len];
    let mut x_scales = vec![0u32; x_scales_len];
    let mut topk_ids_out = vec![0i64; topk_len];
    let mut topk_weights_out = vec![0.0f32; topk_len];
    let mut out = NervaCudaDeepSeekMegaMoePrepareResult::default();
    let request = NervaCudaDeepSeekMegaMoePrepareRequest {
        num_tokens: input.num_tokens,
        hidden_size: input.hidden_size,
        top_k: input.top_k,
        hidden_states: input.hidden_states.as_ptr(),
        topk_ids: input.topk_ids.as_ptr(),
        topk_weights: input.topk_weights.as_ptr(),
        is_padding: input
            .is_padding
            .map_or(ptr::null(), |values| values.as_ptr()),
        x_fp8: x_fp8.as_mut_ptr(),
        x_scales: x_scales.as_mut_ptr(),
        topk_ids_out: topk_ids_out.as_mut_ptr(),
        topk_weights_out: topk_weights_out.as_mut_ptr(),
    };
    let return_code = run_deepseek_megamoe_prepare(&request, &mut out);
    summarize(
        return_code,
        out,
        x_fp8,
        x_scales,
        topk_ids_out,
        topk_weights_out,
    )
}

fn valid_shape(input: &CudaDeepSeekMegaMoePrepareInput<'_>) -> bool {
    let tokens = input.num_tokens as usize;
    let hidden = input.hidden_size as usize;
    let top_k = input.top_k as usize;
    input.num_tokens > 0
        && input.hidden_size > 0
        && input.hidden_size % 128 == 0
        && input.top_k > 0
        && input.hidden_states.len() == tokens * hidden
        && input.topk_ids.len() == tokens * top_k
        && input.topk_weights.len() == tokens * top_k
        && match input.is_padding {
            Some(padding) => padding.len() == tokens,
            None => true,
        }
}

fn output_shape(input: &CudaDeepSeekMegaMoePrepareInput<'_>) -> (usize, usize, usize) {
    let tokens = input.num_tokens as usize;
    let hidden = input.hidden_size as usize;
    let top_k = input.top_k as usize;
    let hidden_blocks = hidden.div_ceil(128);
    (tokens * hidden, tokens * hidden_blocks, tokens * top_k)
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekMegaMoePrepareResult,
    x_fp8: Vec<u8>,
    x_scales: Vec<u32>,
    topk_ids: Vec<i64>,
    topk_weights: Vec<f32>,
) -> CudaDeepSeekMegaMoePrepareSummary {
    let status = if return_code == 0 && out.status == 0 && out.prepare_error == 0 {
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
            "CUDA DeepSeek MegaMoE prepare failed: return_code={} status={} cuda_error={} prepare_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.prepare_error, out.device_count
        ))
    };
    CudaDeepSeekMegaMoePrepareSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        prepare_error: out.prepare_error,
        num_tokens: out.num_tokens,
        hidden_size: out.hidden_size,
        top_k: out.top_k,
        hidden_blocks: out.hidden_blocks,
        x_fp8,
        x_scales,
        topk_ids,
        topk_weights,
        x_fp8_hash: out.x_fp8_hash,
        x_scales_hash: out.x_scales_hash,
        topk_hash: out.topk_hash,
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
    input: CudaDeepSeekMegaMoePrepareInput<'_>,
    output_shape: (usize, usize, usize),
    error: impl Into<String>,
) -> CudaDeepSeekMegaMoePrepareSummary {
    CudaDeepSeekMegaMoePrepareSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        prepare_error: -1,
        num_tokens: input.num_tokens,
        hidden_size: input.hidden_size,
        top_k: input.top_k,
        hidden_blocks: input.hidden_size.div_ceil(128),
        x_fp8: vec![0u8; output_shape.0],
        x_scales: vec![0u32; output_shape.1],
        topk_ids: vec![0i64; output_shape.2],
        topk_weights: vec![0.0f32; output_shape.2],
        x_fp8_hash: 0,
        x_scales_hash: 0,
        topk_hash: 0,
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
