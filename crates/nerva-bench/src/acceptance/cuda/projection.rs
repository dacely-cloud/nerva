use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_projection_benchmark(report: &mut AcceptanceReport) {
    let summary = nerva_cuda::projection::probe::projection_bench(1, 64, 128, 8, 1);
    report.push(
        "cuda_projection_benchmark",
        summary.passed(),
        format!(
            "status={:?} rows={} cols={} dtype={} iterations={} cublaslt_avg_ns={} custom_avg_ns={} selected_strategy={} cublaslt_bandwidth_bps={} custom_bandwidth_bps={} mismatches={} max_abs_diff={} device_allocations={} device_frees={} hot_path_allocations={} error={}",
            summary.status,
            summary.rows,
            summary.cols,
            summary.dtype,
            summary.iterations,
            summary.cublaslt_avg_ns,
            summary.custom_avg_ns,
            summary.selected_strategy_name(),
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
