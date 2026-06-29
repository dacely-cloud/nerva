use std::os::raw::{c_char, c_int};

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaExperimentalRtCandidateBenchRequest {
    pub(crate) pages: u32,
    pub(crate) page_tokens: u32,
    pub(crate) dims: u32,
    pub(crate) query_count: u32,
    pub(crate) candidates_per_query: u32,
    pub(crate) iterations: u32,
    pub(crate) warmup_iterations: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaExperimentalRtCandidateBenchResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) device_ordinal: i32,
    pub(crate) compute_capability_major: i32,
    pub(crate) compute_capability_minor: i32,
    pub(crate) pages: u32,
    pub(crate) page_tokens: u32,
    pub(crate) dims: u32,
    pub(crate) query_count: u32,
    pub(crate) candidates_per_query: u32,
    pub(crate) iterations: u32,
    pub(crate) warmup_iterations: u32,
    pub(crate) rt_core_capable: u32,
    pub(crate) real_rt_backend_available: u32,
    pub(crate) rt_headers_available: u32,
    pub(crate) optix_headers_available: u32,
    pub(crate) vulkan_headers_available: u32,
    pub(crate) vulkan_shader_compiler_available: u32,
    pub(crate) vulkan_loader_available: u32,
    pub(crate) vulkan_rt_extensions_available: u32,
    pub(crate) vulkan_physical_devices: u32,
    pub(crate) descriptor_bytes: u64,
    pub(crate) query_bytes: u64,
    pub(crate) candidate_id_bytes: u64,
    pub(crate) output_bytes: u64,
    pub(crate) dense_selector_total_ns: u64,
    pub(crate) dense_selector_avg_ns: u64,
    pub(crate) software_selector_total_ns: u64,
    pub(crate) software_selector_avg_ns: u64,
    pub(crate) rerank_total_ns: u64,
    pub(crate) rerank_avg_ns: u64,
    pub(crate) selector_plus_rerank_avg_ns: u64,
    pub(crate) dense_vs_selector_speedup_x1000: u64,
    pub(crate) dense_vs_selector_plus_rerank_speedup_x1000: u64,
    pub(crate) candidate_fraction_ppm: u64,
    pub(crate) selected_hash: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) device_allocations: u64,
    pub(crate) device_frees: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) hot_path_allocations: u64,
    pub(crate) backend: [c_char; 64],
    pub(crate) reason: [c_char; 192],
}

impl Default for NervaCudaExperimentalRtCandidateBenchResult {
    fn default() -> Self {
        Self {
            status: -1,
            cuda_error: 0,
            device_count: 0,
            device_ordinal: -1,
            compute_capability_major: 0,
            compute_capability_minor: 0,
            pages: 0,
            page_tokens: 0,
            dims: 0,
            query_count: 0,
            candidates_per_query: 0,
            iterations: 0,
            warmup_iterations: 0,
            rt_core_capable: 0,
            real_rt_backend_available: 0,
            rt_headers_available: 0,
            optix_headers_available: 0,
            vulkan_headers_available: 0,
            vulkan_shader_compiler_available: 0,
            vulkan_loader_available: 0,
            vulkan_rt_extensions_available: 0,
            vulkan_physical_devices: 0,
            descriptor_bytes: 0,
            query_bytes: 0,
            candidate_id_bytes: 0,
            output_bytes: 0,
            dense_selector_total_ns: 0,
            dense_selector_avg_ns: 0,
            software_selector_total_ns: 0,
            software_selector_avg_ns: 0,
            rerank_total_ns: 0,
            rerank_avg_ns: 0,
            selector_plus_rerank_avg_ns: 0,
            dense_vs_selector_speedup_x1000: 0,
            dense_vs_selector_plus_rerank_speedup_x1000: 0,
            candidate_fraction_ppm: 0,
            selected_hash: 0,
            kernel_launches: 0,
            sync_calls: 0,
            device_allocations: 0,
            device_frees: 0,
            device_arena_bytes: 0,
            hot_path_allocations: 0,
            backend: [0; 64],
            reason: [0; 192],
        }
    }
}

unsafe extern "C" {
    fn nerva_cuda_experimental_rt_candidate_bench(
        request: *const NervaCudaExperimentalRtCandidateBenchRequest,
        out: *mut NervaCudaExperimentalRtCandidateBenchResult,
    ) -> c_int;
}

pub(crate) fn run_experimental_rt_candidate_bench(
    request: &NervaCudaExperimentalRtCandidateBenchRequest,
    out: &mut NervaCudaExperimentalRtCandidateBenchResult,
) -> c_int {
    unsafe { nerva_cuda_experimental_rt_candidate_bench(request, out) }
}
