use std::os::raw::c_int;

pub(crate) const CUDA_ERROR_NO_DEVICE: i32 = 100;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaTieredAttentionResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) hidden: u32,
    pub(crate) heads: u32,
    pub(crate) blocks: u32,
    pub(crate) tokens: u32,
    pub(crate) output: [f32; 2],
    pub(crate) output_hash: u64,
    pub(crate) cpu_block_events: u64,
    pub(crate) device_block_events: u64,
    pub(crate) resident_kv_bytes: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) h2d_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

unsafe extern "C" {
    fn nerva_cuda_tiered_attention_smoke(out: *mut NervaCudaTieredAttentionResult) -> c_int;
}

pub(crate) fn run_tiered_attention_smoke(out: &mut NervaCudaTieredAttentionResult) -> c_int {
    unsafe { nerva_cuda_tiered_attention_smoke(out) }
}
