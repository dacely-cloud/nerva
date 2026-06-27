use std::os::raw::c_int;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaProjectionBenchRequest {
    pub(crate) dtype: u32,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) iterations: u32,
    pub(crate) warmup_iterations: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaProjectionBenchResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) device_ordinal: i32,
    pub(crate) compute_capability_major: i32,
    pub(crate) compute_capability_minor: i32,
    pub(crate) dtype: u32,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) iterations: u32,
    pub(crate) warmup_iterations: u32,
    pub(crate) matrix_bytes: u64,
    pub(crate) input_bytes: u64,
    pub(crate) output_bytes: u64,
    pub(crate) cublaslt_total_ns: u64,
    pub(crate) cublaslt_avg_ns: u64,
    pub(crate) cublaslt_default_total_ns: u64,
    pub(crate) cublaslt_default_avg_ns: u64,
    pub(crate) cublaslt_heuristic_count: u32,
    pub(crate) cublaslt_best_heuristic_index: u32,
    pub(crate) cublaslt_best_heuristic_total_ns: u64,
    pub(crate) cublaslt_best_heuristic_avg_ns: u64,
    pub(crate) custom_total_ns: u64,
    pub(crate) custom_avg_ns: u64,
    pub(crate) cublaslt_effective_bandwidth_bps: u64,
    pub(crate) custom_effective_bandwidth_bps: u64,
    pub(crate) selected_strategy: u32,
    pub(crate) mismatch_count: u32,
    pub(crate) max_abs_diff: f32,
    pub(crate) kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) device_allocations: u64,
    pub(crate) device_frees: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) hot_path_allocations: u64,
}

unsafe extern "C" {
    fn nerva_cuda_projection_bench(
        request: *const NervaCudaProjectionBenchRequest,
        out: *mut NervaCudaProjectionBenchResult,
    ) -> c_int;
}

pub(crate) fn run_projection_bench(
    request: &NervaCudaProjectionBenchRequest,
    out: &mut NervaCudaProjectionBenchResult,
) -> c_int {
    unsafe { nerva_cuda_projection_bench(request, out) }
}
