use std::os::raw::c_int;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaDeepSeekMhcPreRequest {
    pub(crate) tokens: u32,
    pub(crate) hc_mult: u32,
    pub(crate) hidden_size: u32,
    pub(crate) sinkhorn_repeat: u32,
    pub(crate) rms_eps: f32,
    pub(crate) hc_pre_eps: f32,
    pub(crate) hc_sinkhorn_eps: f32,
    pub(crate) hc_post_mult_value: f32,
    pub(crate) residual: *const f32,
    pub(crate) fn_weights: *const f32,
    pub(crate) hc_scale: *const f32,
    pub(crate) hc_base: *const f32,
    pub(crate) post_mix: *mut f32,
    pub(crate) comb_mix: *mut f32,
    pub(crate) layer_input: *mut f32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekMhcPreResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) mhc_error: i32,
    pub(crate) tokens: u32,
    pub(crate) hc_mult: u32,
    pub(crate) hidden_size: u32,
    pub(crate) sinkhorn_repeat: u32,
    pub(crate) rms_eps: f32,
    pub(crate) hc_pre_eps: f32,
    pub(crate) hc_sinkhorn_eps: f32,
    pub(crate) hc_post_mult_value: f32,
    pub(crate) post_mix_hash: u64,
    pub(crate) comb_mix_hash: u64,
    pub(crate) layer_input_hash: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) h2d_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaDeepSeekMhcPostRequest {
    pub(crate) tokens: u32,
    pub(crate) hc_mult: u32,
    pub(crate) hidden_size: u32,
    pub(crate) x: *const f32,
    pub(crate) residual: *const f32,
    pub(crate) post_layer_mix: *const f32,
    pub(crate) comb_res_mix: *const f32,
    pub(crate) output: *mut f32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekMhcPostResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) mhc_error: i32,
    pub(crate) tokens: u32,
    pub(crate) hc_mult: u32,
    pub(crate) hidden_size: u32,
    pub(crate) output_hash: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) h2d_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaDeepSeekMhcHeadRequest {
    pub(crate) tokens: u32,
    pub(crate) hc_mult: u32,
    pub(crate) hidden_size: u32,
    pub(crate) rms_eps: f32,
    pub(crate) hc_eps: f32,
    pub(crate) hc_scale: f32,
    pub(crate) hidden_states: *const f32,
    pub(crate) fn_weights: *const f32,
    pub(crate) hc_base: *const f32,
    pub(crate) output: *mut f32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekMhcHeadResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) mhc_error: i32,
    pub(crate) tokens: u32,
    pub(crate) hc_mult: u32,
    pub(crate) hidden_size: u32,
    pub(crate) rms_eps: f32,
    pub(crate) hc_eps: f32,
    pub(crate) hc_scale: f32,
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
    fn nerva_cuda_deepseek_mhc_pre(
        request: *const NervaCudaDeepSeekMhcPreRequest,
        out: *mut NervaCudaDeepSeekMhcPreResult,
    ) -> c_int;
    fn nerva_cuda_deepseek_mhc_post(
        request: *const NervaCudaDeepSeekMhcPostRequest,
        out: *mut NervaCudaDeepSeekMhcPostResult,
    ) -> c_int;
    fn nerva_cuda_deepseek_mhc_head(
        request: *const NervaCudaDeepSeekMhcHeadRequest,
        out: *mut NervaCudaDeepSeekMhcHeadResult,
    ) -> c_int;
}

pub(crate) fn run_deepseek_mhc_pre(
    request: &NervaCudaDeepSeekMhcPreRequest,
    out: &mut NervaCudaDeepSeekMhcPreResult,
) -> c_int {
    unsafe { nerva_cuda_deepseek_mhc_pre(request, out) }
}

pub(crate) fn run_deepseek_mhc_post(
    request: &NervaCudaDeepSeekMhcPostRequest,
    out: &mut NervaCudaDeepSeekMhcPostResult,
) -> c_int {
    unsafe { nerva_cuda_deepseek_mhc_post(request, out) }
}

pub(crate) fn run_deepseek_mhc_head(
    request: &NervaCudaDeepSeekMhcHeadRequest,
    out: &mut NervaCudaDeepSeekMhcHeadResult,
) -> c_int {
    unsafe { nerva_cuda_deepseek_mhc_head(request, out) }
}
