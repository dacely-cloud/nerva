use std::os::raw::c_int;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekMoeSmokeResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) hidden_size: u32,
    pub(crate) intermediate_size: u32,
    pub(crate) num_experts: u32,
    pub(crate) top_k: u32,
    pub(crate) swiglu_limit: f32,
    pub(crate) expert_ids: [u32; 2],
    pub(crate) expert_weights: [f32; 2],
    pub(crate) output: [f32; 3],
    pub(crate) output_hash: u64,
    pub(crate) mismatches: u64,
    pub(crate) max_abs_diff: f32,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaDeepSeekMoeForwardRequest {
    pub(crate) hidden_size: u32,
    pub(crate) intermediate_size: u32,
    pub(crate) num_experts: u32,
    pub(crate) top_k: u32,
    pub(crate) clamp_swiglu: u32,
    pub(crate) swiglu_limit: f32,
    pub(crate) input: *const f32,
    pub(crate) expert_ids: *const u32,
    pub(crate) expert_weights: *const f32,
    pub(crate) w_gate: *const f32,
    pub(crate) w_up: *const f32,
    pub(crate) w_down: *const f32,
    pub(crate) output: *mut f32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekMoeForwardResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) moe_error: i32,
    pub(crate) hidden_size: u32,
    pub(crate) intermediate_size: u32,
    pub(crate) num_experts: u32,
    pub(crate) top_k: u32,
    pub(crate) clamp_swiglu: u32,
    pub(crate) swiglu_limit: f32,
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
    fn nerva_cuda_deepseek_moe_smoke(out: *mut NervaCudaDeepSeekMoeSmokeResult) -> c_int;
    fn nerva_cuda_deepseek_moe_forward(
        request: *const NervaCudaDeepSeekMoeForwardRequest,
        out: *mut NervaCudaDeepSeekMoeForwardResult,
    ) -> c_int;
}

pub(crate) fn run_deepseek_moe_smoke(out: &mut NervaCudaDeepSeekMoeSmokeResult) -> c_int {
    unsafe { nerva_cuda_deepseek_moe_smoke(out) }
}

pub(crate) fn run_deepseek_moe_forward(
    request: &NervaCudaDeepSeekMoeForwardRequest,
    out: &mut NervaCudaDeepSeekMoeForwardResult,
) -> c_int {
    unsafe { nerva_cuda_deepseek_moe_forward(request, out) }
}
