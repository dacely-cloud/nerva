use std::os::raw::c_int;

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

unsafe extern "C" {
    fn nerva_cuda_deepseek_router_smoke(out: *mut NervaCudaDeepSeekRouterSmokeResult) -> c_int;
}

pub(crate) fn run_deepseek_router_smoke(out: &mut NervaCudaDeepSeekRouterSmokeResult) -> c_int {
    unsafe { nerva_cuda_deepseek_router_smoke(out) }
}
