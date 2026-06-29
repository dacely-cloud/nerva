use crate::experimental_rt::ffi::{
    NervaCudaExperimentalRtCandidateBenchRequest, NervaCudaExperimentalRtCandidateBenchResult,
    run_experimental_rt_candidate_bench,
};
use crate::experimental_rt::summary::CudaExperimentalRtCandidateBenchSummary;
use crate::smoke::ffi::{CUDA_ERROR_NO_DEVICE, c_char_array_to_string};
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
            kv_cache_bytes: out.kv_cache_bytes,
            candidate_id_bytes: out.candidate_id_bytes,
            output_bytes: out.output_bytes,
            dense_selector_total_ns: out.dense_selector_total_ns,
            dense_selector_avg_ns: out.dense_selector_avg_ns,
            software_selector_total_ns: out.software_selector_total_ns,
            software_selector_avg_ns: out.software_selector_avg_ns,
            candidate_selector_total_ns: out.candidate_selector_total_ns,
            candidate_selector_avg_ns: out.candidate_selector_avg_ns,
            rerank_total_ns: out.rerank_total_ns,
            rerank_avg_ns: out.rerank_avg_ns,
            selector_plus_rerank_avg_ns: out.selector_plus_rerank_avg_ns,
            dense_vs_selector_speedup_x1000: out.dense_vs_selector_speedup_x1000,
            dense_vs_selector_plus_rerank_speedup_x1000: out
                .dense_vs_selector_plus_rerank_speedup_x1000,
            candidate_fraction_ppm: out.candidate_fraction_ppm,
            candidate_parity_checked: out.candidate_parity_checked != 0,
            candidate_parity_mismatches: out.candidate_parity_mismatches,
            candidate_parity_first_mismatch_index: out.candidate_parity_first_mismatch_index,
            candidate_parity_first_expected: out.candidate_parity_first_expected,
            candidate_parity_first_actual: out.candidate_parity_first_actual,
            candidate_query_hashes_distinct: out.candidate_query_hashes_distinct,
            candidate_query_hash_repeats: out.candidate_query_hash_repeats,
            local_window_tokens: out.local_window_tokens,
            local_attention_total_ns: out.local_attention_total_ns,
            local_attention_avg_ns: out.local_attention_avg_ns,
            kv_page_access_total_ns: out.kv_page_access_total_ns,
            kv_page_access_avg_ns: out.kv_page_access_avg_ns,
            far_sparse_attention_total_ns: out.far_sparse_attention_total_ns,
            far_sparse_attention_avg_ns: out.far_sparse_attention_avg_ns,
            softmax_merge_total_ns: out.softmax_merge_total_ns,
            softmax_merge_avg_ns: out.softmax_merge_avg_ns,
            dense_full_attention_total_ns: out.dense_full_attention_total_ns,
            dense_full_attention_avg_ns: out.dense_full_attention_avg_ns,
            attention_mass_recall_min_ppm: out.attention_mass_recall_min_ppm,
            attention_mass_recall_avg_ppm: out.attention_mass_recall_avg_ppm,
            page_level_attention_mass_recall_min_ppm: out.page_level_attention_mass_recall_min_ppm,
            page_level_attention_mass_recall_avg_ppm: out.page_level_attention_mass_recall_avg_ppm,
            far_oracle_topk_tokens: out.far_oracle_topk_tokens,
            far_oracle_topk_token_recall_min_ppm: out.far_oracle_topk_token_recall_min_ppm,
            far_oracle_topk_token_recall_avg_ppm: out.far_oracle_topk_token_recall_avg_ppm,
            page_level_far_oracle_topk_token_recall_min_ppm: out
                .page_level_far_oracle_topk_token_recall_min_ppm,
            page_level_far_oracle_topk_token_recall_avg_ppm: out
                .page_level_far_oracle_topk_token_recall_avg_ppm,
            far_oracle_topk_importance_scatter_min_pages: out
                .far_oracle_topk_importance_scatter_min_pages,
            far_oracle_topk_importance_scatter_avg_pages_x1000: out
                .far_oracle_topk_importance_scatter_avg_pages_x1000,
            far_oracle_topk_importance_scatter_max_pages: out
                .far_oracle_topk_importance_scatter_max_pages,
            fine_token_projected_topk_tokens: out.fine_token_projected_topk_tokens,
            fine_token_projected_candidate_tokens: out.fine_token_projected_candidate_tokens,
            fine_token_projected_token_recall_min_ppm: out
                .fine_token_projected_token_recall_min_ppm,
            fine_token_projected_token_recall_avg_ppm: out
                .fine_token_projected_token_recall_avg_ppm,
            fine_token_learned_projected_topk_tokens: out.fine_token_learned_projected_topk_tokens,
            fine_token_learned_projected_candidate_tokens: out
                .fine_token_learned_projected_candidate_tokens,
            fine_token_learned_projected_token_recall_min_ppm: out
                .fine_token_learned_projected_token_recall_min_ppm,
            fine_token_learned_projected_token_recall_avg_ppm: out
                .fine_token_learned_projected_token_recall_avg_ppm,
            norm_stress_topk_tokens: out.norm_stress_topk_tokens,
            norm_stress_no_augmentation_token_recall_min_ppm: out
                .norm_stress_no_augmentation_token_recall_min_ppm,
            norm_stress_no_augmentation_token_recall_avg_ppm: out
                .norm_stress_no_augmentation_token_recall_avg_ppm,
            norm_stress_synthetic_norm_augmented_token_recall_min_ppm: out
                .norm_stress_synthetic_norm_augmented_token_recall_min_ppm,
            norm_stress_synthetic_norm_augmented_token_recall_avg_ppm: out
                .norm_stress_synthetic_norm_augmented_token_recall_avg_ppm,
            dense_selector_attention_stage_avg_ns: out.dense_selector_attention_stage_avg_ns,
            rt_selector_attention_stage_avg_ns: out.rt_selector_attention_stage_avg_ns,
            rt_selector_overlapped_attention_stage_avg_ns: out
                .rt_selector_overlapped_attention_stage_avg_ns,
            dense_vs_rt_attention_stage_speedup_x1000: out
                .dense_vs_rt_attention_stage_speedup_x1000,
            dense_vs_rt_overlapped_attention_stage_speedup_x1000: out
                .dense_vs_rt_overlapped_attention_stage_speedup_x1000,
            dense_full_vs_rt_attention_stage_speedup_x1000: out
                .dense_full_vs_rt_attention_stage_speedup_x1000,
            dense_full_vs_rt_overlapped_attention_stage_speedup_x1000: out
                .dense_full_vs_rt_overlapped_attention_stage_speedup_x1000,
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
