use crate::deepseek_moe::ffi::{
    NervaCudaDeepSeekMoeForwardRequest, NervaCudaDeepSeekMoeForwardResult, run_deepseek_moe_forward,
};
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekMoeForwardSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub moe_error: i32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub clamp_swiglu: bool,
    pub swiglu_limit: f32,
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
pub struct CudaDeepSeekMoeForwardInput<'a> {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub clamp_swiglu: bool,
    pub swiglu_limit: f32,
    pub input: &'a [f32],
    pub expert_ids: &'a [u32],
    pub expert_weights: &'a [f32],
    pub w_gate: &'a [f32],
    pub w_up: &'a [f32],
    pub w_down: &'a [f32],
}

pub fn deepseek_moe_forward(
    input: CudaDeepSeekMoeForwardInput<'_>,
) -> CudaDeepSeekMoeForwardSummary {
    let output_values = input.hidden_size as usize;
    if !valid_shape(&input) {
        return failed_summary(
            &input,
            vec![0.0; output_values],
            "invalid DeepSeek MoE forward shape",
        );
    }

    let mut output = vec![0.0f32; output_values];
    let mut out = NervaCudaDeepSeekMoeForwardResult::default();
    let request = NervaCudaDeepSeekMoeForwardRequest {
        hidden_size: input.hidden_size,
        intermediate_size: input.intermediate_size,
        num_experts: input.num_experts,
        top_k: input.top_k,
        clamp_swiglu: u32::from(input.clamp_swiglu),
        swiglu_limit: input.swiglu_limit,
        input: input.input.as_ptr(),
        expert_ids: input.expert_ids.as_ptr(),
        expert_weights: input.expert_weights.as_ptr(),
        w_gate: input.w_gate.as_ptr(),
        w_up: input.w_up.as_ptr(),
        w_down: input.w_down.as_ptr(),
        output: output.as_mut_ptr(),
    };
    let return_code = run_deepseek_moe_forward(&request, &mut out);
    summarize(return_code, out, output)
}

fn valid_shape(input: &CudaDeepSeekMoeForwardInput<'_>) -> bool {
    let hidden = input.hidden_size as usize;
    let intermediate = input.intermediate_size as usize;
    let num_experts = input.num_experts as usize;
    let top_k = input.top_k as usize;
    let expert_matrix = num_experts * intermediate * hidden;
    let down_matrix = num_experts * hidden * intermediate;
    input.hidden_size > 0
        && input.intermediate_size > 0
        && input.num_experts > 0
        && input.top_k > 0
        && input.input.len() == hidden
        && input.expert_ids.len() == top_k
        && input
            .expert_ids
            .iter()
            .all(|expert| *expert < input.num_experts)
        && input.expert_weights.len() == top_k
        && input.w_gate.len() == expert_matrix
        && input.w_up.len() == expert_matrix
        && input.w_down.len() == down_matrix
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekMoeForwardResult,
    output: Vec<f32>,
) -> CudaDeepSeekMoeForwardSummary {
    let status = if return_code == 0 && out.status == 0 && out.moe_error == 0 {
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
            "CUDA DeepSeek MoE forward failed: return_code={} status={} cuda_error={} moe_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.moe_error, out.device_count
        ))
    };
    CudaDeepSeekMoeForwardSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        moe_error: out.moe_error,
        hidden_size: out.hidden_size,
        intermediate_size: out.intermediate_size,
        num_experts: out.num_experts,
        top_k: out.top_k,
        clamp_swiglu: out.clamp_swiglu != 0,
        swiglu_limit: out.swiglu_limit,
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
    input: &CudaDeepSeekMoeForwardInput<'_>,
    output: Vec<f32>,
    error: impl Into<String>,
) -> CudaDeepSeekMoeForwardSummary {
    CudaDeepSeekMoeForwardSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        moe_error: -1,
        hidden_size: input.hidden_size,
        intermediate_size: input.intermediate_size,
        num_experts: input.num_experts,
        top_k: input.top_k,
        clamp_swiglu: input.clamp_swiglu,
        swiglu_limit: input.swiglu_limit,
        output,
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
