use std::os::raw::c_int;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekMlaSmokeResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) heads: u32,
    pub(crate) tokens: u32,
    pub(crate) kv_lora_rank: u32,
    pub(crate) qk_nope_head_dim: u32,
    pub(crate) qk_rope_head_dim: u32,
    pub(crate) v_head_dim: u32,
    pub(crate) softmax_scale: f32,
    pub(crate) output: [f32; 4],
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

unsafe extern "C" {
    fn nerva_cuda_deepseek_mla_smoke(out: *mut NervaCudaDeepSeekMlaSmokeResult) -> c_int;
}

pub(crate) fn run_deepseek_mla_smoke(out: &mut NervaCudaDeepSeekMlaSmokeResult) -> c_int {
    unsafe { nerva_cuda_deepseek_mla_smoke(out) }
}
