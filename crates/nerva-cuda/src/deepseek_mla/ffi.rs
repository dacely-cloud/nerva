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

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaDeepSeekMlaDecodeRequest {
    pub(crate) heads: u32,
    pub(crate) tokens: u32,
    pub(crate) kv_lora_rank: u32,
    pub(crate) qk_nope_head_dim: u32,
    pub(crate) qk_rope_head_dim: u32,
    pub(crate) v_head_dim: u32,
    pub(crate) softmax_scale: f32,
    pub(crate) q_nope: *const f32,
    pub(crate) q_pe: *const f32,
    pub(crate) kv_c: *const f32,
    pub(crate) k_pe: *const f32,
    pub(crate) w_uk: *const f32,
    pub(crate) w_uv: *const f32,
    pub(crate) output: *mut f32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekMlaDecodeResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) decode_error: i32,
    pub(crate) heads: u32,
    pub(crate) tokens: u32,
    pub(crate) kv_lora_rank: u32,
    pub(crate) qk_nope_head_dim: u32,
    pub(crate) qk_rope_head_dim: u32,
    pub(crate) v_head_dim: u32,
    pub(crate) softmax_scale: f32,
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
pub(crate) struct NervaCudaDeepSeekQKvRmsNormRequest {
    pub(crate) num_tokens: u32,
    pub(crate) q_size: u32,
    pub(crate) kv_size: u32,
    pub(crate) eps: f32,
    pub(crate) q: *const f32,
    pub(crate) kv: *const f32,
    pub(crate) q_weight: *const f32,
    pub(crate) kv_weight: *const f32,
    pub(crate) q_out: *mut f32,
    pub(crate) kv_out: *mut f32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekQKvRmsNormResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) num_tokens: u32,
    pub(crate) q_size: u32,
    pub(crate) kv_size: u32,
    pub(crate) eps: f32,
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
    fn nerva_cuda_deepseek_mla_smoke(out: *mut NervaCudaDeepSeekMlaSmokeResult) -> c_int;
    fn nerva_cuda_deepseek_mla_decode(
        request: *const NervaCudaDeepSeekMlaDecodeRequest,
        out: *mut NervaCudaDeepSeekMlaDecodeResult,
    ) -> c_int;
    fn nerva_cuda_deepseek_qkv_rmsnorm(
        request: *const NervaCudaDeepSeekQKvRmsNormRequest,
        out: *mut NervaCudaDeepSeekQKvRmsNormResult,
    ) -> c_int;
}

pub(crate) fn run_deepseek_mla_smoke(out: &mut NervaCudaDeepSeekMlaSmokeResult) -> c_int {
    unsafe { nerva_cuda_deepseek_mla_smoke(out) }
}

pub(crate) fn run_deepseek_mla_decode(
    request: &NervaCudaDeepSeekMlaDecodeRequest,
    out: &mut NervaCudaDeepSeekMlaDecodeResult,
) -> c_int {
    unsafe { nerva_cuda_deepseek_mla_decode(request, out) }
}

pub(crate) fn run_deepseek_qkv_rmsnorm(
    request: &NervaCudaDeepSeekQKvRmsNormRequest,
    out: &mut NervaCudaDeepSeekQKvRmsNormResult,
) -> c_int {
    unsafe { nerva_cuda_deepseek_qkv_rmsnorm(request, out) }
}
