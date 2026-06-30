use std::os::raw::c_int;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaDeepSeekKvFp8DsMlaPackRequest {
    pub(crate) block_size: u32,
    pub(crate) token_index: u32,
    pub(crate) nope_bytes: u32,
    pub(crate) rope_bf16_values: u32,
    pub(crate) scale_dim: u32,
    pub(crate) nope_fp8: *const u8,
    pub(crate) rope_bf16: *const u16,
    pub(crate) scales: *const u8,
    pub(crate) output_block: *mut u8,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekKvFp8DsMlaPackResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) block_size: u32,
    pub(crate) token_index: u32,
    pub(crate) token_stride: u32,
    pub(crate) scale_dim: u32,
    pub(crate) block_bytes: u64,
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
    fn nerva_cuda_deepseek_kv_fp8_ds_mla_pack(
        request: *const NervaCudaDeepSeekKvFp8DsMlaPackRequest,
        out: *mut NervaCudaDeepSeekKvFp8DsMlaPackResult,
    ) -> c_int;
}

pub(crate) fn run_deepseek_kv_fp8_ds_mla_pack(
    request: &NervaCudaDeepSeekKvFp8DsMlaPackRequest,
    out: &mut NervaCudaDeepSeekKvFp8DsMlaPackResult,
) -> c_int {
    unsafe { nerva_cuda_deepseek_kv_fp8_ds_mla_pack(request, out) }
}
