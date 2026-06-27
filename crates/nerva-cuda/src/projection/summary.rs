use crate::projection::summary_json::summary_to_json;
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
    pub cublaslt_default_total_ns: u64,
    pub cublaslt_default_avg_ns: u64,
    pub cublaslt_heuristic_count: u32,
    pub cublaslt_best_heuristic_index: u32,
    pub cublaslt_best_heuristic_total_ns: u64,
    pub cublaslt_best_heuristic_avg_ns: u64,
    pub custom_total_ns: u64,
    pub custom_avg_ns: u64,
    pub cublaslt_graph_total_ns: u64,
    pub cublaslt_graph_avg_ns: u64,
    pub cublaslt_default_graph_total_ns: u64,
    pub cublaslt_default_graph_avg_ns: u64,
    pub cublaslt_best_heuristic_graph_total_ns: u64,
    pub cublaslt_best_heuristic_graph_avg_ns: u64,
    pub custom_graph_total_ns: u64,
    pub custom_graph_avg_ns: u64,
    pub cublaslt_graph_nodes: u64,
    pub custom_graph_nodes: u64,
    pub graph_replays: u64,
    pub graph_captures: u64,
    pub selected_graph_strategy: u32,
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
            cublaslt_default_total_ns: 0,
            cublaslt_default_avg_ns: 0,
            cublaslt_heuristic_count: 0,
            cublaslt_best_heuristic_index: 0,
            cublaslt_best_heuristic_total_ns: 0,
            cublaslt_best_heuristic_avg_ns: 0,
            custom_total_ns: 0,
            custom_avg_ns: 0,
            cublaslt_graph_total_ns: 0,
            cublaslt_graph_avg_ns: 0,
            cublaslt_default_graph_total_ns: 0,
            cublaslt_default_graph_avg_ns: 0,
            cublaslt_best_heuristic_graph_total_ns: 0,
            cublaslt_best_heuristic_graph_avg_ns: 0,
            custom_graph_total_ns: 0,
            custom_graph_avg_ns: 0,
            cublaslt_graph_nodes: 0,
            custom_graph_nodes: 0,
            graph_replays: 0,
            graph_captures: 0,
            selected_graph_strategy: 0,
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
            && self.cublaslt_graph_avg_ns > 0
            && self.custom_graph_avg_ns > 0
            && self.cublaslt_graph_nodes > 0
            && self.custom_graph_nodes > 0
            && self.graph_replays > 0
            && self.graph_captures > 0
            && matches!(
                self.selected_strategy,
                PROJECTION_STRATEGY_CUBLASLT | PROJECTION_STRATEGY_CUSTOM
            )
            && matches!(
                self.selected_graph_strategy,
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

    pub fn selected_graph_strategy_name(&self) -> &'static str {
        match self.selected_graph_strategy {
            PROJECTION_STRATEGY_CUBLASLT => "cublaslt",
            PROJECTION_STRATEGY_CUSTOM => "custom_row_major",
            _ => "none",
        }
    }

    pub fn to_json(&self) -> String {
        summary_to_json(self)
    }
}
