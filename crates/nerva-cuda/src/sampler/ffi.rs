use std::os::raw::c_int;

pub(crate) const CUDA_ERROR_NO_DEVICE: i32 = 100;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaGreedySamplerResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) vocab_size: u32,
    pub(crate) token_index: u64,
    pub(crate) token: u32,
    pub(crate) slot_version: u64,
    pub(crate) completion: u32,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) h2d_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

unsafe extern "C" {
    fn nerva_cuda_greedy_sampler_smoke(out: *mut NervaCudaGreedySamplerResult) -> c_int;
}

pub(crate) fn run_greedy_sampler_smoke(out: &mut NervaCudaGreedySamplerResult) -> c_int {
    unsafe { nerva_cuda_greedy_sampler_smoke(out) }
}
