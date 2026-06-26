use std::os::raw::c_int;

pub(crate) const CUDA_ERROR_NO_DEVICE: i32 = 100;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaTinyBlockResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) hidden: u32,
    pub(crate) intermediate: u32,
    pub(crate) output: [u16; 2],
    pub(crate) output_hash: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) hot_path_allocations: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaLoadedTinyBlockResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) hidden: u32,
    pub(crate) intermediate: u32,
    pub(crate) output: [u16; 2],
    pub(crate) output_hash: u64,
    pub(crate) resident_weight_bytes: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) h2d_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

unsafe extern "C" {
    fn nerva_cuda_tiny_block_smoke(out: *mut NervaCudaTinyBlockResult) -> c_int;
    fn nerva_cuda_loaded_tiny_block_smoke(out: *mut NervaCudaLoadedTinyBlockResult) -> c_int;
}

pub(crate) fn run_tiny_block_smoke(out: &mut NervaCudaTinyBlockResult) -> c_int {
    unsafe { nerva_cuda_tiny_block_smoke(out) }
}

pub(crate) fn run_loaded_tiny_block_smoke(out: &mut NervaCudaLoadedTinyBlockResult) -> c_int {
    unsafe { nerva_cuda_loaded_tiny_block_smoke(out) }
}
