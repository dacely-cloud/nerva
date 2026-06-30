use crate::deepseek_moe::ffi::{
    NervaCudaDeepSeekMegaMoeExpertsRequest, NervaCudaDeepSeekMegaMoeExpertsResult,
    run_deepseek_megamoe_experts,
};
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekMegaMoeExpertsSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub expert_error: i32,
    pub num_tokens: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub output: Vec<f32>,
    pub output_hash: u64,
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
pub struct CudaDeepSeekMegaMoeExpertsInput<'a> {
    pub num_tokens: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub swiglu_limit: f32,
    pub x_fp8: &'a [u8],
    pub x_scales: &'a [u32],
    pub topk_ids: &'a [i64],
    pub topk_weights: &'a [f32],
    pub w13_packed: &'a [u8],
    pub w13_scales: &'a [u8],
    pub w2_packed: &'a [u8],
    pub w2_scales: &'a [u8],
}

pub fn deepseek_megamoe_experts(
    input: CudaDeepSeekMegaMoeExpertsInput<'_>,
) -> CudaDeepSeekMegaMoeExpertsSummary {
    let output_len = output_len(&input);
    if !valid_shape(&input) {
        return failed_summary(input, output_len, "invalid DeepSeek MegaMoE expert shape");
    }

    let mut output = vec![0.0f32; output_len];
    let mut out = NervaCudaDeepSeekMegaMoeExpertsResult::default();
    let request = NervaCudaDeepSeekMegaMoeExpertsRequest {
        num_tokens: input.num_tokens,
        hidden_size: input.hidden_size,
        intermediate_size: input.intermediate_size,
        num_experts: input.num_experts,
        top_k: input.top_k,
        swiglu_limit: input.swiglu_limit,
        x_fp8: input.x_fp8.as_ptr(),
        x_scales: input.x_scales.as_ptr(),
        topk_ids: input.topk_ids.as_ptr(),
        topk_weights: input.topk_weights.as_ptr(),
        w13_packed: input.w13_packed.as_ptr(),
        w13_scales: input.w13_scales.as_ptr(),
        w2_packed: input.w2_packed.as_ptr(),
        w2_scales: input.w2_scales.as_ptr(),
        output: output.as_mut_ptr(),
    };
    let return_code = run_deepseek_megamoe_experts(&request, &mut out);
    summarize(return_code, out, output)
}

fn valid_shape(input: &CudaDeepSeekMegaMoeExpertsInput<'_>) -> bool {
    let Some(shape) = expected_shape(input) else {
        return false;
    };
    input.num_tokens > 0
        && input.hidden_size > 0
        && input.hidden_size % 128 == 0
        && input.intermediate_size > 0
        && input.intermediate_size % 32 == 0
        && input.num_experts > 0
        && input.top_k > 0
        && shape.output_values <= u32::MAX as usize
        && input.x_fp8.len() == shape.x_fp8
        && input.x_scales.len() == shape.x_scales
        && input.topk_ids.len() == shape.topk
        && input.topk_weights.len() == shape.topk
        && input.w13_packed.len() == shape.w13_packed
        && input.w13_scales.len() == shape.w13_scales
        && input.w2_packed.len() == shape.w2_packed
        && input.w2_scales.len() == shape.w2_scales
}

fn output_len(input: &CudaDeepSeekMegaMoeExpertsInput<'_>) -> usize {
    expected_shape(input)
        .map(|shape| shape.output_values)
        .unwrap_or(0)
}

#[derive(Clone, Copy)]
struct ExpectedShape {
    x_fp8: usize,
    x_scales: usize,
    topk: usize,
    w13_packed: usize,
    w13_scales: usize,
    w2_packed: usize,
    w2_scales: usize,
    output_values: usize,
}

fn expected_shape(input: &CudaDeepSeekMegaMoeExpertsInput<'_>) -> Option<ExpectedShape> {
    let tokens = input.num_tokens as usize;
    let hidden = input.hidden_size as usize;
    let intermediate = input.intermediate_size as usize;
    let experts = input.num_experts as usize;
    let top_k = input.top_k as usize;
    if hidden % 128 != 0 || hidden % 32 != 0 || intermediate % 32 != 0 {
        return None;
    }
    let hidden_blocks = hidden.checked_div(128)?;
    let w13_rows = experts.checked_mul(2)?.checked_mul(intermediate)?;
    let w2_rows = experts.checked_mul(hidden)?;
    Some(ExpectedShape {
        x_fp8: tokens.checked_mul(hidden)?,
        x_scales: tokens.checked_mul(hidden_blocks)?,
        topk: tokens.checked_mul(top_k)?,
        w13_packed: w13_rows.checked_mul(hidden.checked_div(2)?)?,
        w13_scales: w13_rows.checked_mul(hidden.checked_div(32)?)?,
        w2_packed: w2_rows.checked_mul(intermediate.checked_div(2)?)?,
        w2_scales: w2_rows.checked_mul(intermediate.checked_div(32)?)?,
        output_values: tokens.checked_mul(hidden)?,
    })
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekMegaMoeExpertsResult,
    output: Vec<f32>,
) -> CudaDeepSeekMegaMoeExpertsSummary {
    let status = if return_code == 0 && out.status == 0 && out.expert_error == 0 {
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
            "CUDA DeepSeek MegaMoE experts failed: return_code={} status={} cuda_error={} expert_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.expert_error, out.device_count
        ))
    };
    CudaDeepSeekMegaMoeExpertsSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        expert_error: out.expert_error,
        num_tokens: out.num_tokens,
        hidden_size: out.hidden_size,
        intermediate_size: out.intermediate_size,
        num_experts: out.num_experts,
        top_k: out.top_k,
        output,
        output_hash: out.output_hash,
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
    input: CudaDeepSeekMegaMoeExpertsInput<'_>,
    output_len: usize,
    error: impl Into<String>,
) -> CudaDeepSeekMegaMoeExpertsSummary {
    CudaDeepSeekMegaMoeExpertsSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        expert_error: -1,
        num_tokens: input.num_tokens,
        hidden_size: input.hidden_size,
        intermediate_size: input.intermediate_size,
        num_experts: input.num_experts,
        top_k: input.top_k,
        output: vec![0.0f32; output_len],
        output_hash: 0,
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
