use crate::projection::ffi::{
    run_projection_bench, NervaCudaProjectionBenchRequest, NervaCudaProjectionBenchResult,
};
use crate::projection::summary::CudaProjectionBenchSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

pub fn projection_bench(
    dtype: u32,
    rows: u32,
    cols: u32,
    iterations: u32,
    warmup_iterations: u32,
    block_tokens: u32,
) -> CudaProjectionBenchSummary {
    let request = NervaCudaProjectionBenchRequest {
        dtype,
        rows,
        cols,
        iterations,
        warmup_iterations,
        block_tokens,
    };
    let mut out = NervaCudaProjectionBenchResult::default();
    let return_code = run_projection_bench(&request, &mut out);
    if return_code == 0 && out.status == 0 {
        return CudaProjectionBenchSummary {
            status: SmokeStatus::Ok,
            dtype: out.dtype,
            rows: out.rows,
            cols: out.cols,
            block_tokens: out.block_tokens,
            iterations: out.iterations,
            warmup_iterations: out.warmup_iterations,
            compute_capability_major: (out.compute_capability_major > 0)
                .then_some(out.compute_capability_major),
            compute_capability_minor: (out.compute_capability_major > 0)
                .then_some(out.compute_capability_minor),
            matrix_bytes: out.matrix_bytes,
            input_bytes: out.input_bytes,
            output_bytes: out.output_bytes,
            cublaslt_total_ns: out.cublaslt_total_ns,
            cublaslt_avg_ns: out.cublaslt_avg_ns,
            cublaslt_default_total_ns: out.cublaslt_default_total_ns,
            cublaslt_default_avg_ns: out.cublaslt_default_avg_ns,
            cublaslt_heuristic_count: out.cublaslt_heuristic_count,
            cublaslt_best_heuristic_index: out.cublaslt_best_heuristic_index,
            cublaslt_best_heuristic_total_ns: out.cublaslt_best_heuristic_total_ns,
            cublaslt_best_heuristic_avg_ns: out.cublaslt_best_heuristic_avg_ns,
            custom_total_ns: out.custom_total_ns,
            custom_avg_ns: out.custom_avg_ns,
            cublaslt_graph_total_ns: out.cublaslt_graph_total_ns,
            cublaslt_graph_avg_ns: out.cublaslt_graph_avg_ns,
            cublaslt_default_graph_total_ns: out.cublaslt_default_graph_total_ns,
            cublaslt_default_graph_avg_ns: out.cublaslt_default_graph_avg_ns,
            cublaslt_best_heuristic_graph_total_ns: out.cublaslt_best_heuristic_graph_total_ns,
            cublaslt_best_heuristic_graph_avg_ns: out.cublaslt_best_heuristic_graph_avg_ns,
            custom_graph_total_ns: out.custom_graph_total_ns,
            custom_graph_avg_ns: out.custom_graph_avg_ns,
            cublaslt_graph_nodes: out.cublaslt_graph_nodes,
            custom_graph_nodes: out.custom_graph_nodes,
            graph_replays: out.graph_replays,
            graph_captures: out.graph_captures,
            selected_graph_strategy: out.selected_graph_strategy,
            cublaslt_effective_bandwidth_bps: out.cublaslt_effective_bandwidth_bps,
            custom_effective_bandwidth_bps: out.custom_effective_bandwidth_bps,
            selected_strategy: out.selected_strategy,
            mismatch_count: out.mismatch_count,
            max_abs_diff: out.max_abs_diff,
            kernel_launches: out.kernel_launches,
            sync_calls: out.sync_calls,
            device_allocations: out.device_allocations,
            device_frees: out.device_frees,
            device_arena_bytes: out.device_arena_bytes,
            hot_path_allocations: out.hot_path_allocations,
            block_cublaslt_total_ns: out.block_cublaslt_total_ns,
            block_cublaslt_avg_ns: out.block_cublaslt_avg_ns,
            block_cublaslt_per_token_ns: out.block_cublaslt_per_token_ns,
            block_cublaslt_graph_total_ns: out.block_cublaslt_graph_total_ns,
            block_cublaslt_graph_avg_ns: out.block_cublaslt_graph_avg_ns,
            block_cublaslt_graph_per_token_ns: out.block_cublaslt_graph_per_token_ns,
            block_cublaslt_graph_nodes: out.block_cublaslt_graph_nodes,
            block_cublaslt_speedup_x1000: out.block_cublaslt_speedup_x1000,
            block_cublaslt_graph_speedup_x1000: out.block_cublaslt_graph_speedup_x1000,
            block_cublaslt_effective_bandwidth_bps: out.block_cublaslt_effective_bandwidth_bps,
            error: None,
        };
    }

    let reason = format!(
        "CUDA projection bench failed: return_code={} status={} cuda_error={} device_count={} rows={} cols={} dtype={} iterations={} cublaslt_avg_ns={} custom_avg_ns={} mismatches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.rows,
        out.cols,
        out.dtype,
        out.iterations,
        out.cublaslt_avg_ns,
        out.custom_avg_ns,
        out.mismatch_count,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaProjectionBenchSummary::unavailable(
            dtype,
            rows,
            cols,
            iterations,
            warmup_iterations,
            reason,
        )
    } else {
        CudaProjectionBenchSummary::failed(dtype, rows, cols, iterations, warmup_iterations, reason)
    }
}
