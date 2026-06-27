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

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaHfSamplerRequest {
    pub(crate) dtype: u32,
    pub(crate) hidden: u32,
    pub(crate) vocab_size: u32,
    pub(crate) token_index: u64,
    pub(crate) rms_eps: f32,
    pub(crate) hidden_bits: *const u16,
    pub(crate) final_norm_weight: *const u16,
    pub(crate) lm_head: *const u16,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaHfSamplerResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) dtype: u32,
    pub(crate) hidden: u32,
    pub(crate) vocab_size: u32,
    pub(crate) token_index: u64,
    pub(crate) token: u32,
    pub(crate) slot_version: u64,
    pub(crate) completion: u32,
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
    fn nerva_cuda_greedy_sampler_smoke(out: *mut NervaCudaGreedySamplerResult) -> c_int;
    fn nerva_cuda_hf_sample_u16(
        request: *const NervaCudaHfSamplerRequest,
        out: *mut NervaCudaHfSamplerResult,
    ) -> c_int;
}

pub(crate) fn run_greedy_sampler_smoke(out: &mut NervaCudaGreedySamplerResult) -> c_int {
    unsafe { nerva_cuda_greedy_sampler_smoke(out) }
}

pub(crate) fn run_hf_sample_u16(
    request: &NervaCudaHfSamplerRequest,
    out: &mut NervaCudaHfSamplerResult,
) -> c_int {
    unsafe { nerva_cuda_hf_sample_u16(request, out) }
}
