use std::os::raw::c_int;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekQuantSmokeResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) fp8_rows: u32,
    pub(crate) fp8_cols: u32,
    pub(crate) fp8_block_rows: u32,
    pub(crate) fp8_block_cols: u32,
    pub(crate) mxfp4_rows: u32,
    pub(crate) mxfp4_packed_cols: u32,
    pub(crate) mxfp4_scale_packed_cols: u32,
    pub(crate) fp8_output_hash: u64,
    pub(crate) mxfp4_output_hash: u64,
    pub(crate) fp8_mismatches: u64,
    pub(crate) mxfp4_mismatches: u64,
    pub(crate) fp8_max_abs_diff: f32,
    pub(crate) mxfp4_max_abs_diff: f32,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) h2d_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

unsafe extern "C" {
    fn nerva_cuda_deepseek_quant_smoke(out: *mut NervaCudaDeepSeekQuantSmokeResult) -> c_int;
}

pub(crate) fn run_deepseek_quant_smoke(out: &mut NervaCudaDeepSeekQuantSmokeResult) -> c_int {
    unsafe { nerva_cuda_deepseek_quant_smoke(out) }
}
