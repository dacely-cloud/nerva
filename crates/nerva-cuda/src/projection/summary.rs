use crate::json::{json_opt_i32, json_opt_str};
use crate::smoke::status::SmokeStatus;

pub const PROJECTION_STRATEGY_CUBLASLT: u32 = 1;
pub const PROJECTION_STRATEGY_CUSTOM: u32 = 2;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaProjectionBenchSummary {
    pub status: SmokeStatus,
    pub dtype: u32,
    pub rows: u32,
    pub cols: u32,
    pub iterations: u32,
    pub warmup_iterations: u32,
    pub compute_capability_major: Option<i32>,
    pub compute_capability_minor: Option<i32>,
    pub matrix_bytes: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub cublaslt_total_ns: u64,
    pub cublaslt_avg_ns: u64,
    pub custom_total_ns: u64,
    pub custom_avg_ns: u64,
    pub cublaslt_effective_bandwidth_bps: u64,
    pub custom_effective_bandwidth_bps: u64,
    pub selected_strategy: u32,
    pub mismatch_count: u32,
    pub max_abs_diff: f32,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub device_allocations: u64,
    pub device_frees: u64,
    pub device_arena_bytes: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaProjectionBenchSummary {
    pub fn failed(
        dtype: u32,
        rows: u32,
        cols: u32,
        iterations: u32,
        warmup_iterations: u32,
        reason: String,
    ) -> Self {
        Self {
            status: SmokeStatus::Failed,
            dtype,
            rows,
            cols,
            iterations,
            warmup_iterations,
            compute_capability_major: None,
            compute_capability_minor: None,
            matrix_bytes: 0,
            input_bytes: 0,
            output_bytes: 0,
            cublaslt_total_ns: 0,
            cublaslt_avg_ns: 0,
            custom_total_ns: 0,
            custom_avg_ns: 0,
            cublaslt_effective_bandwidth_bps: 0,
            custom_effective_bandwidth_bps: 0,
            selected_strategy: 0,
            mismatch_count: 0,
            max_abs_diff: 0.0,
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
        dtype: u32,
        rows: u32,
        cols: u32,
        iterations: u32,
        warmup_iterations: u32,
        reason: String,
    ) -> Self {
        Self {
            status: SmokeStatus::Unavailable,
            ..Self::failed(dtype, rows, cols, iterations, warmup_iterations, reason)
        }
    }

    pub fn passed(&self) -> bool {
        self.status == SmokeStatus::Ok
            && self.rows > 0
            && self.cols > 0
            && self.iterations > 0
            && self.cublaslt_avg_ns > 0
            && self.custom_avg_ns > 0
            && matches!(
                self.selected_strategy,
                PROJECTION_STRATEGY_CUBLASLT | PROJECTION_STRATEGY_CUSTOM
            )
            && self.mismatch_count == 0
            && self.max_abs_diff.is_finite()
            && self.device_allocations == self.device_frees
            && self.hot_path_allocations == 0
    }

    pub fn selected_strategy_name(&self) -> &'static str {
        match self.selected_strategy {
            PROJECTION_STRATEGY_CUBLASLT => "cublaslt",
            PROJECTION_STRATEGY_CUSTOM => "custom_row_major",
            _ => "none",
        }
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"dtype\":{},\"rows\":{},\"cols\":{},\"iterations\":{},\"warmup_iterations\":{},\"compute_capability_major\":{},\"compute_capability_minor\":{},\"matrix_bytes\":{},\"input_bytes\":{},\"output_bytes\":{},\"cublaslt_total_ns\":{},\"cublaslt_avg_ns\":{},\"custom_total_ns\":{},\"custom_avg_ns\":{},\"cublaslt_effective_bandwidth_bps\":{},\"custom_effective_bandwidth_bps\":{},\"selected_strategy\":\"{}\",\"selected_strategy_id\":{},\"mismatch_count\":{},\"max_abs_diff\":{},\"kernel_launches\":{},\"sync_calls\":{},\"device_allocations\":{},\"device_frees\":{},\"device_arena_bytes\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.dtype,
            self.rows,
            self.cols,
            self.iterations,
            self.warmup_iterations,
            json_opt_i32(self.compute_capability_major),
            json_opt_i32(self.compute_capability_minor),
            self.matrix_bytes,
            self.input_bytes,
            self.output_bytes,
            self.cublaslt_total_ns,
            self.cublaslt_avg_ns,
            self.custom_total_ns,
            self.custom_avg_ns,
            self.cublaslt_effective_bandwidth_bps,
            self.custom_effective_bandwidth_bps,
            self.selected_strategy_name(),
            self.selected_strategy,
            self.mismatch_count,
            self.max_abs_diff,
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
