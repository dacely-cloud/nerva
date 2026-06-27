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

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaBlockForwardRequest {
    pub(crate) dtype: u32,
    pub(crate) hidden: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) intermediate: u32,
    pub(crate) position: u32,
    pub(crate) rms_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) input: *const u16,
    pub(crate) rms_attn_weight: *const u16,
    pub(crate) rms_mlp_weight: *const u16,
    pub(crate) w_q: *const u16,
    pub(crate) w_k: *const u16,
    pub(crate) w_v: *const u16,
    pub(crate) w_o: *const u16,
    pub(crate) q_bias: *const u16,
    pub(crate) k_bias: *const u16,
    pub(crate) v_bias: *const u16,
    pub(crate) o_bias: *const u16,
    pub(crate) w_gate: *const u16,
    pub(crate) w_up: *const u16,
    pub(crate) w_down: *const u16,
    pub(crate) output: *mut u16,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaBlockForwardResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) dtype: u32,
    pub(crate) hidden: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) intermediate: u32,
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
    fn nerva_cuda_block_forward_u16(
        request: *const NervaCudaBlockForwardRequest,
        out: *mut NervaCudaBlockForwardResult,
    ) -> c_int;
}

pub(crate) fn run_tiny_block_smoke(out: &mut NervaCudaTinyBlockResult) -> c_int {
    unsafe { nerva_cuda_tiny_block_smoke(out) }
}

pub(crate) fn run_loaded_tiny_block_smoke(out: &mut NervaCudaLoadedTinyBlockResult) -> c_int {
    unsafe { nerva_cuda_loaded_tiny_block_smoke(out) }
}

pub(crate) fn run_block_forward_u16(
    request: &NervaCudaBlockForwardRequest,
    out: &mut NervaCudaBlockForwardResult,
) -> c_int {
    unsafe { nerva_cuda_block_forward_u16(request, out) }
}
