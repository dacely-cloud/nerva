use std::os::raw::c_int;

pub(crate) const NERVA_CUDA_DEEPSEEK_ROUTER_V3_GROUPED_SIGMOID: u32 = 1;
pub(crate) const NERVA_CUDA_DEEPSEEK_ROUTER_V4_SQRTSOFTPLUS: u32 = 2;
pub(crate) const NERVA_CUDA_DEEPSEEK_ROUTER_V4_HASH: u32 = 3;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekRouterSmokeResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) v3_num_experts: u32,
    pub(crate) v3_num_groups: u32,
    pub(crate) v3_top_k_groups: u32,
    pub(crate) v3_top_k: u32,
    pub(crate) v4_num_experts: u32,
    pub(crate) v4_top_k: u32,
    pub(crate) v4_hash_top_k: u32,
    pub(crate) v3_expert_ids: [u32; 2],
    pub(crate) v4_expert_ids: [u32; 2],
    pub(crate) v4_hash_expert_ids: [u32; 3],
    pub(crate) v3_weights: [f32; 2],
    pub(crate) v4_weights: [f32; 2],
    pub(crate) v4_hash_weights: [f32; 3],
    pub(crate) v3_output_hash: u64,
    pub(crate) v4_output_hash: u64,
    pub(crate) v4_hash_output_hash: u64,
    pub(crate) v3_mismatches: u64,
    pub(crate) v4_mismatches: u64,
    pub(crate) v4_hash_mismatches: u64,
    pub(crate) v3_max_abs_diff: f32,
    pub(crate) v4_max_abs_diff: f32,
    pub(crate) v4_hash_max_abs_diff: f32,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaDeepSeekRouterRouteRequest {
    pub(crate) router_kind: u32,
    pub(crate) num_experts: u32,
    pub(crate) num_groups: u32,
    pub(crate) top_k_groups: u32,
    pub(crate) top_k: u32,
    pub(crate) norm_topk_prob: u32,
    pub(crate) route_token: u32,
    pub(crate) routed_scaling_factor: f32,
    pub(crate) logits: *const f32,
    pub(crate) correction_bias: *const f32,
    pub(crate) hash_route_table: *const u32,
    pub(crate) hash_route_table_len: u32,
    pub(crate) expert_ids: *mut u32,
    pub(crate) weights: *mut f32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekRouterRouteResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) route_error: i32,
    pub(crate) router_kind: u32,
    pub(crate) num_experts: u32,
    pub(crate) num_groups: u32,
    pub(crate) top_k_groups: u32,
    pub(crate) top_k: u32,
    pub(crate) output_hash: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) h2d_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

unsafe extern "C" {
    fn nerva_cuda_deepseek_router_smoke(out: *mut NervaCudaDeepSeekRouterSmokeResult) -> c_int;
    fn nerva_cuda_deepseek_router_route(
        request: *const NervaCudaDeepSeekRouterRouteRequest,
        out: *mut NervaCudaDeepSeekRouterRouteResult,
    ) -> c_int;
}

pub(crate) fn run_deepseek_router_smoke(out: &mut NervaCudaDeepSeekRouterSmokeResult) -> c_int {
    unsafe { nerva_cuda_deepseek_router_smoke(out) }
}

pub(crate) fn run_deepseek_router_route(
    request: &NervaCudaDeepSeekRouterRouteRequest,
    out: &mut NervaCudaDeepSeekRouterRouteResult,
) -> c_int {
    unsafe { nerva_cuda_deepseek_router_route(request, out) }
}
