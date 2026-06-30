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

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaDeepSeekQuantFp8DequantRequest {
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) block_rows: u32,
    pub(crate) block_cols: u32,
    pub(crate) weights: *const u8,
    pub(crate) scales: *const u8,
    pub(crate) output: *mut f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaDeepSeekQuantMxfp4DequantRequest {
    pub(crate) rows: u32,
    pub(crate) packed_cols: u32,
    pub(crate) scale_packed_cols: u32,
    pub(crate) packed: *const u8,
    pub(crate) scales: *const u8,
    pub(crate) output: *mut f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaDeepSeekQuantFp8F32ScaleMatvecRequest {
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) block_rows: u32,
    pub(crate) block_cols: u32,
    pub(crate) weights: *const u8,
    pub(crate) scales: *const f32,
    pub(crate) input: *const f32,
    pub(crate) output: *mut f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaDeepSeekQuantFp8F32ScaleEncodedMatvecRequest {
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) block_rows: u32,
    pub(crate) block_cols: u32,
    pub(crate) input_dtype: u32,
    pub(crate) weights: *const u8,
    pub(crate) scales: *const f32,
    pub(crate) input: *const u16,
    pub(crate) output: *mut f32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekQuantDequantResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) block_rows: u32,
    pub(crate) block_cols: u32,
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
pub(crate) struct NervaCudaDeepSeekFusedInvRopeFp8QuantRequest {
    pub(crate) num_tokens: u32,
    pub(crate) n_groups: u32,
    pub(crate) heads_per_group: u32,
    pub(crate) head_dim: u32,
    pub(crate) rope_dim: u32,
    pub(crate) quant_group_size: u32,
    pub(crate) cos_sin_stride: u32,
    pub(crate) fp8_max: f32,
    pub(crate) eps: f32,
    pub(crate) input: *const f32,
    pub(crate) positions: *const i64,
    pub(crate) cos_sin_cache: *const f32,
    pub(crate) fp8_output: *mut u8,
    pub(crate) scale_output: *mut f32,
    pub(crate) packed_scale_output: *mut u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaDeepSeekFusedInvRopeFp8QuantResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) num_tokens: u32,
    pub(crate) n_groups: u32,
    pub(crate) heads_per_group: u32,
    pub(crate) head_dim: u32,
    pub(crate) rope_dim: u32,
    pub(crate) quant_group_size: u32,
    pub(crate) scale_blocks: u32,
    pub(crate) fp8_output_hash: u64,
    pub(crate) scale_output_hash: u64,
    pub(crate) packed_scale_output_hash: u64,
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
    fn nerva_cuda_deepseek_quant_fp8_dequant(
        request: *const NervaCudaDeepSeekQuantFp8DequantRequest,
        out: *mut NervaCudaDeepSeekQuantDequantResult,
    ) -> c_int;
    fn nerva_cuda_deepseek_quant_mxfp4_dequant(
        request: *const NervaCudaDeepSeekQuantMxfp4DequantRequest,
        out: *mut NervaCudaDeepSeekQuantDequantResult,
    ) -> c_int;
    fn nerva_cuda_deepseek_quant_fp8_f32_scale_matvec(
        request: *const NervaCudaDeepSeekQuantFp8F32ScaleMatvecRequest,
        out: *mut NervaCudaDeepSeekQuantDequantResult,
    ) -> c_int;
    fn nerva_cuda_deepseek_quant_fp8_f32_scale_encoded_matvec(
        request: *const NervaCudaDeepSeekQuantFp8F32ScaleEncodedMatvecRequest,
        out: *mut NervaCudaDeepSeekQuantDequantResult,
    ) -> c_int;
    fn nerva_cuda_deepseek_fused_inv_rope_fp8_quant(
        request: *const NervaCudaDeepSeekFusedInvRopeFp8QuantRequest,
        out: *mut NervaCudaDeepSeekFusedInvRopeFp8QuantResult,
    ) -> c_int;
}

pub(crate) fn run_deepseek_quant_smoke(out: &mut NervaCudaDeepSeekQuantSmokeResult) -> c_int {
    unsafe { nerva_cuda_deepseek_quant_smoke(out) }
}

pub(crate) fn run_deepseek_quant_fp8_dequant(
    request: &NervaCudaDeepSeekQuantFp8DequantRequest,
    out: &mut NervaCudaDeepSeekQuantDequantResult,
) -> c_int {
    unsafe { nerva_cuda_deepseek_quant_fp8_dequant(request, out) }
}

pub(crate) fn run_deepseek_quant_mxfp4_dequant(
    request: &NervaCudaDeepSeekQuantMxfp4DequantRequest,
    out: &mut NervaCudaDeepSeekQuantDequantResult,
) -> c_int {
    unsafe { nerva_cuda_deepseek_quant_mxfp4_dequant(request, out) }
}

pub(crate) fn run_deepseek_quant_fp8_f32_scale_matvec(
    request: &NervaCudaDeepSeekQuantFp8F32ScaleMatvecRequest,
    out: &mut NervaCudaDeepSeekQuantDequantResult,
) -> c_int {
    unsafe { nerva_cuda_deepseek_quant_fp8_f32_scale_matvec(request, out) }
}

pub(crate) fn run_deepseek_quant_fp8_f32_scale_encoded_matvec(
    request: &NervaCudaDeepSeekQuantFp8F32ScaleEncodedMatvecRequest,
    out: &mut NervaCudaDeepSeekQuantDequantResult,
) -> c_int {
    unsafe { nerva_cuda_deepseek_quant_fp8_f32_scale_encoded_matvec(request, out) }
}

pub(crate) fn run_deepseek_fused_inv_rope_fp8_quant(
    request: &NervaCudaDeepSeekFusedInvRopeFp8QuantRequest,
    out: &mut NervaCudaDeepSeekFusedInvRopeFp8QuantResult,
) -> c_int {
    unsafe { nerva_cuda_deepseek_fused_inv_rope_fp8_quant(request, out) }
}
