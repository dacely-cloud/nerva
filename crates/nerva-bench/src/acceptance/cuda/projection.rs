use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_projection_benchmark(report: &mut AcceptanceReport) {
    let summary = nerva_cuda::projection::probe::projection_bench(1, 64, 128, 8, 1, 1);
    report.push(
        "cuda_projection_benchmark",
        summary.passed(),
        format!(
            "status={:?} rows={} cols={} dtype={} iterations={} cublaslt_avg_ns={} cublaslt_default_avg_ns={} cublaslt_heuristics={} cublaslt_best_heuristic_index={} cublaslt_best_heuristic_avg_ns={} custom_avg_ns={} cublaslt_graph_avg_ns={} cublaslt_default_graph_avg_ns={} cublaslt_best_heuristic_graph_avg_ns={} custom_graph_avg_ns={} cublaslt_graph_nodes={} custom_graph_nodes={} graph_replays={} graph_captures={} selected_strategy={} selected_graph_strategy={} cublaslt_bandwidth_bps={} custom_bandwidth_bps={} mismatches={} max_abs_diff={} device_allocations={} device_frees={} hot_path_allocations={} error={}",
            summary.status,
            summary.rows,
            summary.cols,
            summary.dtype,
            summary.iterations,
            summary.cublaslt_avg_ns,
            summary.cublaslt_default_avg_ns,
            summary.cublaslt_heuristic_count,
            summary.cublaslt_best_heuristic_index,
            summary.cublaslt_best_heuristic_avg_ns,
            summary.custom_avg_ns,
            summary.cublaslt_graph_avg_ns,
            summary.cublaslt_default_graph_avg_ns,
            summary.cublaslt_best_heuristic_graph_avg_ns,
            summary.custom_graph_avg_ns,
            summary.cublaslt_graph_nodes,
            summary.custom_graph_nodes,
            summary.graph_replays,
            summary.graph_captures,
            summary.selected_strategy_name(),
            summary.selected_graph_strategy_name(),
            summary.cublaslt_effective_bandwidth_bps,
            summary.custom_effective_bandwidth_bps,
            summary.mismatch_count,
            summary.max_abs_diff,
            summary.device_allocations,
            summary.device_frees,
            summary.hot_path_allocations,
            summary.error.as_deref().unwrap_or("none"),
        ),
    );
}
