use crate::json::{json_opt_i32, json_opt_str};
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaExperimentalRtCandidateBenchSummary {
    pub status: SmokeStatus,
    pub backend: String,
    pub reason: String,
    pub pages: u32,
    pub page_tokens: u32,
    pub dims: u32,
    pub query_count: u32,
    pub candidates_per_query: u32,
    pub iterations: u32,
    pub warmup_iterations: u32,
    pub compute_capability_major: Option<i32>,
    pub compute_capability_minor: Option<i32>,
    pub rt_core_capable: bool,
    pub real_rt_backend_available: bool,
    pub rt_headers_available: bool,
    pub optix_headers_available: bool,
    pub vulkan_headers_available: bool,
    pub vulkan_shader_compiler_available: bool,
    pub vulkan_loader_available: bool,
    pub vulkan_rt_extensions_available: bool,
    pub vulkan_physical_devices: u32,
    pub descriptor_bytes: u64,
    pub query_bytes: u64,
    pub candidate_id_bytes: u64,
    pub output_bytes: u64,
    pub dense_selector_total_ns: u64,
    pub dense_selector_avg_ns: u64,
    pub software_selector_total_ns: u64,
    pub software_selector_avg_ns: u64,
    pub rerank_total_ns: u64,
    pub rerank_avg_ns: u64,
    pub selector_plus_rerank_avg_ns: u64,
    pub dense_vs_selector_speedup_x1000: u64,
    pub dense_vs_selector_plus_rerank_speedup_x1000: u64,
    pub candidate_fraction_ppm: u64,
    pub selected_hash: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub device_allocations: u64,
    pub device_frees: u64,
    pub device_arena_bytes: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaExperimentalRtCandidateBenchSummary {
    pub fn failed(
        pages: u32,
        page_tokens: u32,
        dims: u32,
        query_count: u32,
        candidates_per_query: u32,
        iterations: u32,
        warmup_iterations: u32,
        reason: String,
    ) -> Self {
        Self {
            status: SmokeStatus::Failed,
            backend: "unavailable".to_string(),
            reason: "experimental RT candidate selector bench did not run".to_string(),
            pages,
            page_tokens,
            dims,
            query_count,
            candidates_per_query,
            iterations,
            warmup_iterations,
            compute_capability_major: None,
            compute_capability_minor: None,
            rt_core_capable: false,
            real_rt_backend_available: false,
            rt_headers_available: false,
            optix_headers_available: false,
            vulkan_headers_available: false,
            vulkan_shader_compiler_available: false,
            vulkan_loader_available: false,
            vulkan_rt_extensions_available: false,
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
            error: Some(reason),
        }
    }

    pub fn unavailable(
        pages: u32,
        page_tokens: u32,
        dims: u32,
        query_count: u32,
        candidates_per_query: u32,
        iterations: u32,
        warmup_iterations: u32,
        reason: String,
    ) -> Self {
        Self {
            status: SmokeStatus::Unavailable,
            ..Self::failed(
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

    pub fn passed(&self) -> bool {
        self.status == SmokeStatus::Ok
            && self.pages > 0
            && self.page_tokens > 0
            && self.dims > 0
            && self.query_count > 0
            && self.candidates_per_query > 0
            && self.iterations > 0
            && self.dense_selector_avg_ns > 0
            && self.software_selector_avg_ns > 0
            && self.rerank_avg_ns > 0
            && self.selector_plus_rerank_avg_ns > 0
            && self.device_allocations == self.device_frees
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"backend\":\"{}\",\"reason\":\"{}\",\"pages\":{},\"page_tokens\":{},\"dims\":{},\"query_count\":{},\"candidates_per_query\":{},\"iterations\":{},\"warmup_iterations\":{},\"compute_capability_major\":{},\"compute_capability_minor\":{},\"rt_core_capable\":{},\"real_rt_backend_available\":{},\"rt_headers_available\":{},\"optix_headers_available\":{},\"vulkan_headers_available\":{},\"vulkan_shader_compiler_available\":{},\"vulkan_loader_available\":{},\"vulkan_rt_extensions_available\":{},\"vulkan_physical_devices\":{},\"descriptor_bytes\":{},\"query_bytes\":{},\"candidate_id_bytes\":{},\"output_bytes\":{},\"dense_selector_total_ns\":{},\"dense_selector_avg_ns\":{},\"software_selector_total_ns\":{},\"software_selector_avg_ns\":{},\"rerank_total_ns\":{},\"rerank_avg_ns\":{},\"selector_plus_rerank_avg_ns\":{},\"dense_vs_selector_speedup_x1000\":{},\"dense_vs_selector_plus_rerank_speedup_x1000\":{},\"candidate_fraction_ppm\":{},\"selected_hash\":{},\"kernel_launches\":{},\"sync_calls\":{},\"device_allocations\":{},\"device_frees\":{},\"device_arena_bytes\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            crate::json::escape_json(&self.backend),
            crate::json::escape_json(&self.reason),
            self.pages,
            self.page_tokens,
            self.dims,
            self.query_count,
            self.candidates_per_query,
            self.iterations,
            self.warmup_iterations,
            json_opt_i32(self.compute_capability_major),
            json_opt_i32(self.compute_capability_minor),
            self.rt_core_capable,
            self.real_rt_backend_available,
            self.rt_headers_available,
            self.optix_headers_available,
            self.vulkan_headers_available,
            self.vulkan_shader_compiler_available,
            self.vulkan_loader_available,
            self.vulkan_rt_extensions_available,
            self.vulkan_physical_devices,
            self.descriptor_bytes,
            self.query_bytes,
            self.candidate_id_bytes,
            self.output_bytes,
            self.dense_selector_total_ns,
            self.dense_selector_avg_ns,
            self.software_selector_total_ns,
            self.software_selector_avg_ns,
            self.rerank_total_ns,
            self.rerank_avg_ns,
            self.selector_plus_rerank_avg_ns,
            self.dense_vs_selector_speedup_x1000,
            self.dense_vs_selector_plus_rerank_speedup_x1000,
            self.candidate_fraction_ppm,
            self.selected_hash,
            self.kernel_launches,
            self.sync_calls,
            self.device_allocations,
            self.device_frees,
            self.device_arena_bytes,
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
    }
}
