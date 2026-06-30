use crate::experimental_rt::ffi::{
    NervaCudaExperimentalRtColdKvStagingRequest, NervaCudaExperimentalRtColdKvStagingResult,
    run_experimental_rt_cold_kv_staging_bench,
};
use crate::json::{escape_json, json_opt_i32, json_opt_str};
use crate::smoke::ffi::{CUDA_ERROR_NO_DEVICE, c_char_array_to_string};
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaExperimentalRtColdKvStagingSummary {
    pub status: SmokeStatus,
    pub backend: String,
    pub reason: String,
    pub page_bytes: u64,
    pub pages_per_step: u32,
    pub iterations: u32,
    pub warmup_iterations: u32,
    pub compute_capability_major: Option<i32>,
    pub compute_capability_minor: Option<i32>,
    pub bytes_per_step: u64,
    pub total_h2d_bytes: u64,
    pub h2d_total_ns: u64,
    pub h2d_avg_ns: u64,
    pub h2d_avg_page_ns: u64,
    pub effective_bandwidth_bps: u64,
    pub effective_bandwidth_gbps_x1000: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub device_allocations: u64,
    pub device_frees: u64,
    pub pinned_host_allocations: u64,
    pub pinned_host_frees: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaExperimentalRtColdKvStagingSummary {
    pub fn passed(&self) -> bool {
        self.status == SmokeStatus::Ok
            && self.page_bytes > 0
            && self.pages_per_step > 0
            && self.iterations > 0
            && self.bytes_per_step > 0
            && self.total_h2d_bytes > 0
            && self.h2d_avg_ns > 0
            && self.h2d_avg_page_ns > 0
            && self.effective_bandwidth_bps > 0
            && self.device_allocations == self.device_frees
            && self.pinned_host_allocations == self.pinned_host_frees
            && self.hot_path_allocations == 0
            && self.error.is_none()
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"backend\":\"{}\",\"reason\":\"{}\",\"page_bytes\":{},\"pages_per_step\":{},\"iterations\":{},\"warmup_iterations\":{},\"compute_capability_major\":{},\"compute_capability_minor\":{},\"bytes_per_step\":{},\"total_h2d_bytes\":{},\"h2d_total_ns\":{},\"h2d_avg_ns\":{},\"h2d_avg_page_ns\":{},\"effective_bandwidth_bps\":{},\"effective_bandwidth_gbps_x1000\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"device_allocations\":{},\"device_frees\":{},\"pinned_host_allocations\":{},\"pinned_host_frees\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            escape_json(&self.backend),
            escape_json(&self.reason),
            self.page_bytes,
            self.pages_per_step,
            self.iterations,
            self.warmup_iterations,
            json_opt_i32(self.compute_capability_major),
            json_opt_i32(self.compute_capability_minor),
            self.bytes_per_step,
            self.total_h2d_bytes,
            self.h2d_total_ns,
            self.h2d_avg_ns,
            self.h2d_avg_page_ns,
            self.effective_bandwidth_bps,
            self.effective_bandwidth_gbps_x1000,
            self.device_arena_bytes,
            self.pinned_host_bytes,
            self.device_allocations,
            self.device_frees,
            self.pinned_host_allocations,
            self.pinned_host_frees,
            self.sync_calls,
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
    }
}

pub fn experimental_rt_cold_kv_staging_bench(
    page_bytes: u64,
    pages_per_step: u32,
    iterations: u32,
    warmup_iterations: u32,
) -> CudaExperimentalRtColdKvStagingSummary {
    let request = NervaCudaExperimentalRtColdKvStagingRequest {
        page_bytes,
        pages_per_step,
        iterations,
        warmup_iterations,
    };
    let mut out = NervaCudaExperimentalRtColdKvStagingResult::default();
    let return_code = run_experimental_rt_cold_kv_staging_bench(&request, &mut out);
    if return_code == 0 && out.status == 0 {
        return CudaExperimentalRtColdKvStagingSummary {
            status: SmokeStatus::Ok,
            backend: c_char_array_to_string(&out.backend)
                .unwrap_or_else(|| "cuda_pinned_h2d_cold_kv_staging".to_string()),
            reason: c_char_array_to_string(&out.reason).unwrap_or_default(),
            page_bytes: out.page_bytes,
            pages_per_step: out.pages_per_step,
            iterations: out.iterations,
            warmup_iterations: out.warmup_iterations,
            compute_capability_major: (out.compute_capability_major > 0)
                .then_some(out.compute_capability_major),
            compute_capability_minor: (out.compute_capability_major > 0)
                .then_some(out.compute_capability_minor),
            bytes_per_step: out.bytes_per_step,
            total_h2d_bytes: out.total_h2d_bytes,
            h2d_total_ns: out.h2d_total_ns,
            h2d_avg_ns: out.h2d_avg_ns,
            h2d_avg_page_ns: out.h2d_avg_page_ns,
            effective_bandwidth_bps: out.effective_bandwidth_bps,
            effective_bandwidth_gbps_x1000: gbps_x1000(out.effective_bandwidth_bps),
            device_arena_bytes: out.device_arena_bytes,
            pinned_host_bytes: out.pinned_host_bytes,
            device_allocations: out.device_allocations,
            device_frees: out.device_frees,
            pinned_host_allocations: out.pinned_host_allocations,
            pinned_host_frees: out.pinned_host_frees,
            sync_calls: out.sync_calls,
            hot_path_allocations: out.hot_path_allocations,
            error: None,
        };
    }

    let reason = format!(
        "CUDA cold KV staging bench failed: return_code={} status={} cuda_error={} device_count={} page_bytes={} pages_per_step={} iterations={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        page_bytes,
        pages_per_step,
        iterations,
    );
    CudaExperimentalRtColdKvStagingSummary {
        status: if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
            SmokeStatus::Unavailable
        } else {
            SmokeStatus::Failed
        },
        backend: c_char_array_to_string(&out.backend)
            .unwrap_or_else(|| "cuda_pinned_h2d_cold_kv_staging".to_string()),
        reason,
        page_bytes,
        pages_per_step,
        iterations,
        warmup_iterations,
        compute_capability_major: None,
        compute_capability_minor: None,
        bytes_per_step: out.bytes_per_step,
        total_h2d_bytes: out.total_h2d_bytes,
        h2d_total_ns: out.h2d_total_ns,
        h2d_avg_ns: out.h2d_avg_ns,
        h2d_avg_page_ns: out.h2d_avg_page_ns,
        effective_bandwidth_bps: out.effective_bandwidth_bps,
        effective_bandwidth_gbps_x1000: gbps_x1000(out.effective_bandwidth_bps),
        device_arena_bytes: out.device_arena_bytes,
        pinned_host_bytes: out.pinned_host_bytes,
        device_allocations: out.device_allocations,
        device_frees: out.device_frees,
        pinned_host_allocations: out.pinned_host_allocations,
        pinned_host_frees: out.pinned_host_frees,
        sync_calls: out.sync_calls,
        hot_path_allocations: out.hot_path_allocations,
        error: Some(c_char_array_to_string(&out.reason).unwrap_or_default()),
    }
}

fn gbps_x1000(bytes_per_second: u64) -> u64 {
    bytes_per_second / 1_000_000
}
