use std::ptr;

use crate::deepseek_router::ffi::{
    NERVA_CUDA_DEEPSEEK_ROUTER_V3_GROUPED_SIGMOID, NERVA_CUDA_DEEPSEEK_ROUTER_V4_HASH,
    NERVA_CUDA_DEEPSEEK_ROUTER_V4_SQRTSOFTPLUS, NervaCudaDeepSeekRouterRouteRequest,
    NervaCudaDeepSeekRouterRouteResult, run_deepseek_router_route,
};
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekRouterRouteSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub route_error: i32,
    pub router_kind: u32,
    pub num_experts: u32,
    pub num_groups: u32,
    pub top_k_groups: u32,
    pub top_k: u32,
    pub expert_ids: Vec<u32>,
    pub weights: Vec<f32>,
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

#[derive(Clone, Copy, Debug)]
struct RouterShape {
    kind: u32,
    num_experts: u32,
    num_groups: u32,
    top_k_groups: u32,
    top_k: u32,
    norm_topk_prob: u32,
    route_token: u32,
    routed_scaling_factor: f32,
}

pub fn deepseek_router_route_v3_grouped_sigmoid(
    logits: &[f32],
    correction_bias: Option<&[f32]>,
    num_groups: u32,
    top_k_groups: u32,
    top_k: u32,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
) -> CudaDeepSeekRouterRouteSummary {
    route(
        RouterShape {
            kind: NERVA_CUDA_DEEPSEEK_ROUTER_V3_GROUPED_SIGMOID,
            num_experts: logits.len() as u32,
            num_groups,
            top_k_groups,
            top_k,
            norm_topk_prob: u32::from(norm_topk_prob),
            route_token: 0,
            routed_scaling_factor,
        },
        logits,
        correction_bias,
        None,
    )
}

pub fn deepseek_router_route_v4_sqrtsoftplus(
    logits: &[f32],
    correction_bias: Option<&[f32]>,
    top_k: u32,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
) -> CudaDeepSeekRouterRouteSummary {
    route(
        RouterShape {
            kind: NERVA_CUDA_DEEPSEEK_ROUTER_V4_SQRTSOFTPLUS,
            num_experts: logits.len() as u32,
            num_groups: 0,
            top_k_groups: 0,
            top_k,
            norm_topk_prob: u32::from(norm_topk_prob),
            route_token: 0,
            routed_scaling_factor,
        },
        logits,
        correction_bias,
        None,
    )
}

pub fn deepseek_router_route_v4_hash(
    logits: &[f32],
    hash_route_table: &[u32],
    route_token: u32,
    top_k: u32,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
) -> CudaDeepSeekRouterRouteSummary {
    route(
        RouterShape {
            kind: NERVA_CUDA_DEEPSEEK_ROUTER_V4_HASH,
            num_experts: logits.len() as u32,
            num_groups: 0,
            top_k_groups: 0,
            top_k,
            norm_topk_prob: u32::from(norm_topk_prob),
            route_token,
            routed_scaling_factor,
        },
        logits,
        None,
        Some(hash_route_table),
    )
}

fn route(
    shape: RouterShape,
    logits: &[f32],
    correction_bias: Option<&[f32]>,
    hash_route_table: Option<&[u32]>,
) -> CudaDeepSeekRouterRouteSummary {
    let mut expert_ids = vec![0u32; shape.top_k as usize];
    let mut weights = vec![0.0f32; shape.top_k as usize];
    let mut out = NervaCudaDeepSeekRouterRouteResult::default();
    if shape.num_experts as usize != logits.len()
        || correction_bias.is_some_and(|bias| bias.len() != logits.len())
    {
        return failed_summary(
            shape,
            expert_ids,
            weights,
            "invalid DeepSeek router input shape",
        );
    }
    let request = NervaCudaDeepSeekRouterRouteRequest {
        router_kind: shape.kind,
        num_experts: shape.num_experts,
        num_groups: shape.num_groups,
        top_k_groups: shape.top_k_groups,
        top_k: shape.top_k,
        norm_topk_prob: shape.norm_topk_prob,
        route_token: shape.route_token,
        routed_scaling_factor: shape.routed_scaling_factor,
        logits: logits.as_ptr(),
        correction_bias: correction_bias.map_or(ptr::null(), |bias| bias.as_ptr()),
        hash_route_table: hash_route_table.map_or(ptr::null(), |table| table.as_ptr()),
        hash_route_table_len: hash_route_table.map_or(0, |table| table.len() as u32),
        expert_ids: expert_ids.as_mut_ptr(),
        weights: weights.as_mut_ptr(),
    };
    let return_code = run_deepseek_router_route(&request, &mut out);
    let status = if return_code == 0 && out.status == 0 && out.route_error == 0 {
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
            "CUDA DeepSeek router route failed: return_code={} status={} cuda_error={} route_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.route_error, out.device_count
        ))
    };
    CudaDeepSeekRouterRouteSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        route_error: out.route_error,
        router_kind: out.router_kind,
        num_experts: out.num_experts,
        num_groups: out.num_groups,
        top_k_groups: out.top_k_groups,
        top_k: out.top_k,
        expert_ids,
        weights,
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
    shape: RouterShape,
    expert_ids: Vec<u32>,
    weights: Vec<f32>,
    error: impl Into<String>,
) -> CudaDeepSeekRouterRouteSummary {
    CudaDeepSeekRouterRouteSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        route_error: -1,
        router_kind: shape.kind,
        num_experts: shape.num_experts,
        num_groups: shape.num_groups,
        top_k_groups: shape.top_k_groups,
        top_k: shape.top_k,
        expert_ids,
        weights,
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
