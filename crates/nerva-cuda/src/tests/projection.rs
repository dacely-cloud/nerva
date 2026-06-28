use crate::projection::summary::{
    CudaProjectionBenchSummary, PROJECTION_STRATEGY_CUBLASLT, PROJECTION_STRATEGY_CUSTOM,
};

#[test]
fn projection_bench_summary_reports_selected_strategy() {
    let summary = CudaProjectionBenchSummary {
        status: crate::smoke::status::SmokeStatus::Ok,
        dtype: 1,
        rows: 64,
        cols: 128,
        block_tokens: 4,
        iterations: 4,
        warmup_iterations: 1,
        compute_capability_major: Some(12),
        compute_capability_minor: Some(0),
        matrix_bytes: 16_384,
        input_bytes: 256,
        output_bytes: 256,
        cublaslt_total_ns: 8_000,
        cublaslt_avg_ns: 2_000,
        cublaslt_default_total_ns: 9_000,
        cublaslt_default_avg_ns: 2_250,
        cublaslt_heuristic_count: 4,
        cublaslt_best_heuristic_index: 1,
        cublaslt_best_heuristic_total_ns: 8_000,
        cublaslt_best_heuristic_avg_ns: 2_000,
        custom_total_ns: 4_000,
        custom_avg_ns: 1_000,
        cublaslt_graph_total_ns: 7_200,
        cublaslt_graph_avg_ns: 1_800,
        cublaslt_default_graph_total_ns: 8_000,
        cublaslt_default_graph_avg_ns: 2_000,
        cublaslt_best_heuristic_graph_total_ns: 7_200,
        cublaslt_best_heuristic_graph_avg_ns: 1_800,
        custom_graph_total_ns: 4_400,
        custom_graph_avg_ns: 1_100,
        cublaslt_graph_nodes: 1,
        custom_graph_nodes: 1,
        graph_replays: 15,
        graph_captures: 3,
        selected_graph_strategy: PROJECTION_STRATEGY_CUSTOM,
        cublaslt_effective_bandwidth_bps: 8_000_000,
        custom_effective_bandwidth_bps: 16_000_000,
        selected_strategy: PROJECTION_STRATEGY_CUSTOM,
        mismatch_count: 0,
        max_abs_diff: 0.0,
        kernel_launches: 10,
        sync_calls: 5,
        device_allocations: 7,
        device_frees: 7,
        device_arena_bytes: 1_000_000,
        hot_path_allocations: 0,
        block_cublaslt_total_ns: 4_000,
        block_cublaslt_avg_ns: 1_000,
        block_cublaslt_per_token_ns: 250,
        block_cublaslt_graph_total_ns: 3_600,
        block_cublaslt_graph_avg_ns: 900,
        block_cublaslt_graph_per_token_ns: 225,
        block_cublaslt_graph_nodes: 1,
        block_cublaslt_speedup_x1000: 8_000,
        block_cublaslt_graph_speedup_x1000: 8_000,
        block_cublaslt_effective_bandwidth_bps: 32_000_000,
        error: None,
    };

    assert!(summary.passed());
    assert_eq!(summary.selected_strategy_name(), "custom_row_major");
    assert_eq!(summary.selected_graph_strategy_name(), "custom_row_major");
    assert!(
        summary
            .to_json()
            .contains("\"selected_strategy\":\"custom_row_major\"")
    );
    assert!(
        summary
            .to_json()
            .contains("\"selected_graph_strategy\":\"custom_row_major\"")
    );
}

#[test]
fn projection_bench_summary_rejects_mismatches_and_unfreed_allocations() {
    let mut summary = CudaProjectionBenchSummary {
        status: crate::smoke::status::SmokeStatus::Ok,
        dtype: 1,
        rows: 64,
        cols: 128,
        block_tokens: 1,
        iterations: 4,
        warmup_iterations: 1,
        compute_capability_major: Some(12),
        compute_capability_minor: Some(0),
        matrix_bytes: 16_384,
        input_bytes: 256,
        output_bytes: 256,
        cublaslt_total_ns: 8_000,
        cublaslt_avg_ns: 2_000,
        cublaslt_default_total_ns: 8_000,
        cublaslt_default_avg_ns: 2_000,
        cublaslt_heuristic_count: 0,
        cublaslt_best_heuristic_index: 0,
        cublaslt_best_heuristic_total_ns: 0,
        cublaslt_best_heuristic_avg_ns: 0,
        custom_total_ns: 12_000,
        custom_avg_ns: 3_000,
        cublaslt_graph_total_ns: 8_000,
        cublaslt_graph_avg_ns: 2_000,
        cublaslt_default_graph_total_ns: 8_000,
        cublaslt_default_graph_avg_ns: 2_000,
        cublaslt_best_heuristic_graph_total_ns: 0,
        cublaslt_best_heuristic_graph_avg_ns: 0,
        custom_graph_total_ns: 12_000,
        custom_graph_avg_ns: 3_000,
        cublaslt_graph_nodes: 1,
        custom_graph_nodes: 1,
        graph_replays: 10,
        graph_captures: 2,
        selected_graph_strategy: PROJECTION_STRATEGY_CUBLASLT,
        cublaslt_effective_bandwidth_bps: 8_000_000,
        custom_effective_bandwidth_bps: 5_000_000,
        selected_strategy: PROJECTION_STRATEGY_CUBLASLT,
        mismatch_count: 1,
        max_abs_diff: 0.0,
        kernel_launches: 10,
        sync_calls: 5,
        device_allocations: 7,
        device_frees: 7,
        device_arena_bytes: 1_000_000,
        hot_path_allocations: 0,
        block_cublaslt_total_ns: 0,
        block_cublaslt_avg_ns: 0,
        block_cublaslt_per_token_ns: 0,
        block_cublaslt_graph_total_ns: 0,
        block_cublaslt_graph_avg_ns: 0,
        block_cublaslt_graph_per_token_ns: 0,
        block_cublaslt_graph_nodes: 0,
        block_cublaslt_speedup_x1000: 0,
        block_cublaslt_graph_speedup_x1000: 0,
        block_cublaslt_effective_bandwidth_bps: 0,
        error: None,
    };
    assert!(!summary.passed());

    summary.mismatch_count = 0;
    summary.device_frees = 6;
    assert!(!summary.passed());
}
