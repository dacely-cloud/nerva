use crate::experimental_rt::ffi::{
    run_experimental_rt_candidate_bench, NervaCudaExperimentalRtCandidateBenchRequest,
    NervaCudaExperimentalRtCandidateBenchResult,
};
use crate::experimental_rt::summary::CudaExperimentalRtCandidateBenchSummary;
use crate::smoke::ffi::{c_char_array_to_string, CUDA_ERROR_NO_DEVICE};
use crate::smoke::status::SmokeStatus;

pub fn experimental_rt_candidate_bench(
    pages: u32,
    page_tokens: u32,
    dims: u32,
    query_count: u32,
    candidates_per_query: u32,
    iterations: u32,
    warmup_iterations: u32,
) -> CudaExperimentalRtCandidateBenchSummary {
    let request = NervaCudaExperimentalRtCandidateBenchRequest {
        pages,
        page_tokens,
        dims,
        query_count,
        candidates_per_query,
        iterations,
        warmup_iterations,
    };
    let mut out = NervaCudaExperimentalRtCandidateBenchResult::default();
    let return_code = run_experimental_rt_candidate_bench(&request, &mut out);
    if return_code == 0 && out.status == 0 {
        return CudaExperimentalRtCandidateBenchSummary {
            status: SmokeStatus::Ok,
            backend: c_char_array_to_string(&out.backend)
                .unwrap_or_else(|| "software_cuda_candidate_selector".to_string()),
            reason: c_char_array_to_string(&out.reason).unwrap_or_default(),
            pages: out.pages,
            page_tokens: out.page_tokens,
            dims: out.dims,
            query_count: out.query_count,
            candidates_per_query: out.candidates_per_query,
            iterations: out.iterations,
            warmup_iterations: out.warmup_iterations,
            compute_capability_major: (out.compute_capability_major > 0)
                .then_some(out.compute_capability_major),
            compute_capability_minor: (out.compute_capability_major > 0)
                .then_some(out.compute_capability_minor),
            rt_core_capable: out.rt_core_capable != 0,
            real_rt_backend_available: out.real_rt_backend_available != 0,
            rt_headers_available: out.rt_headers_available != 0,
            optix_headers_available: out.optix_headers_available != 0,
            vulkan_headers_available: out.vulkan_headers_available != 0,
            vulkan_shader_compiler_available: out.vulkan_shader_compiler_available != 0,
            vulkan_loader_available: out.vulkan_loader_available != 0,
            vulkan_rt_extensions_available: out.vulkan_rt_extensions_available != 0,
            vulkan_physical_devices: out.vulkan_physical_devices,
            descriptor_bytes: out.descriptor_bytes,
            query_bytes: out.query_bytes,
            candidate_id_bytes: out.candidate_id_bytes,
            output_bytes: out.output_bytes,
            dense_selector_total_ns: out.dense_selector_total_ns,
            dense_selector_avg_ns: out.dense_selector_avg_ns,
            software_selector_total_ns: out.software_selector_total_ns,
            software_selector_avg_ns: out.software_selector_avg_ns,
            rerank_total_ns: out.rerank_total_ns,
            rerank_avg_ns: out.rerank_avg_ns,
            selector_plus_rerank_avg_ns: out.selector_plus_rerank_avg_ns,
            dense_vs_selector_speedup_x1000: out.dense_vs_selector_speedup_x1000,
            dense_vs_selector_plus_rerank_speedup_x1000: out
                .dense_vs_selector_plus_rerank_speedup_x1000,
            candidate_fraction_ppm: out.candidate_fraction_ppm,
            selected_hash: out.selected_hash,
            kernel_launches: out.kernel_launches,
            sync_calls: out.sync_calls,
            device_allocations: out.device_allocations,
            device_frees: out.device_frees,
            device_arena_bytes: out.device_arena_bytes,
            hot_path_allocations: out.hot_path_allocations,
            error: None,
        };
    }

    let reason = format!(
        "CUDA experimental RT candidate bench failed: return_code={} status={} cuda_error={} device_count={} pages={} page_tokens={} dims={} query_count={} candidates_per_query={} iterations={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        pages,
        page_tokens,
        dims,
        query_count,
        candidates_per_query,
        iterations,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaExperimentalRtCandidateBenchSummary::unavailable(
            pages,
            page_tokens,
            dims,
            query_count,
            candidates_per_query,
            iterations,
            warmup_iterations,
            reason,
        )
    } else {
        CudaExperimentalRtCandidateBenchSummary::failed(
            pages,
            page_tokens,
            dims,
            query_count,
            candidates_per_query,
            iterations,
            warmup_iterations,
            reason,
        )
    }
}
