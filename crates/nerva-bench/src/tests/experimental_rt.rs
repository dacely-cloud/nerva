use nerva_cuda::experimental_rt::summary::CudaExperimentalRtCandidateBenchSummary;

use crate::cli::cuda_rt::{
    RtArgs, RtMatrixConfig, RtMatrixPoint, candidate_sweep_values, decode_latency_estimate_json,
    matrix_json, sweep_json,
};

#[test]
fn candidate_sweep_values_include_bounds_and_powers() {
    assert_eq!(candidate_sweep_values(100, 3, 20), vec![3, 4, 8, 16, 20]);
    assert_eq!(candidate_sweep_values(5, 1, 999), vec![1, 2, 4, 5]);
}

#[test]
fn sweep_json_reports_candidate_points() {
    let config = RtArgs {
        context_tokens: 4096,
        query_count: 1,
        candidates_per_query: 4,
        iterations: 1,
        page_tokens: 64,
        layer_count: 36,
    };
    let one = failed_summary(1);
    let four = failed_summary(4);
    let json = sweep_json(&config, &[one, four]);

    assert!(json.contains("\"mode\":\"experimental_rt_candidate_sweep\""));
    assert!(json.contains("\"scope\":\"recall_candidate_size_synthetic\""));
    assert!(json.contains("\"candidates_per_query\":1"));
    assert!(json.contains("\"candidates_per_query\":4"));
    assert!(json.contains("\"page_level_equal_bytes_baseline\":"));
    assert!(json.contains("\"decode_latency_estimate\":"));
    assert!(json.contains("\"full_decode_latency_measured\":false"));
    assert!(json.contains("\"layer_count\":36"));
    assert!(json.contains("\"scope\":\"synthetic_equal_bytes_measured\""));
    assert!(json.contains("\"selector\":\"synthetic_page_level_descriptor_topk\""));
    assert!(json.contains("\"quest_reproduction_status\":\"pending\""));
    assert!(json.contains("\"attention_mass_recall_min_ppm\":0"));
    assert!(json.contains("\"far_oracle_topk_token_recall_min_ppm\":0"));
    assert!(json.contains("\"far_oracle_topk_importance_scatter_avg_pages_x1000\":0"));
    assert!(json.contains("\"points\":["));
}

#[test]
fn decode_latency_estimate_reports_layer_scaled_savings() {
    let mut summary = failed_summary(4);
    summary.dense_selector_attention_stage_avg_ns = 1_000;
    summary.rt_selector_attention_stage_avg_ns = 700;
    summary.rt_selector_overlapped_attention_stage_avg_ns = 600;
    summary.dense_full_attention_avg_ns = 2_000;

    let json = decode_latency_estimate_json(&summary, 36);
    assert!(json.contains("\"scope\":\"attention_stage_derived\""));
    assert!(json.contains("\"rt_vs_dense_selector_saved_ns_per_layer\":300"));
    assert!(json.contains("\"rt_vs_dense_selector_saved_ns_per_token\":10800"));
    assert!(json.contains("\"rt_overlapped_vs_dense_full_saved_ns_per_token\":50400"));
}

#[test]
fn matrix_json_reports_context_query_candidate_grid() {
    let config = RtMatrixConfig {
        iterations: 1,
        page_tokens: 64,
        layer_count: 36,
    };
    let points = vec![
        RtMatrixPoint {
            context_tokens: 128 * 1024,
            query_count: 1,
            candidates_per_query: 128,
            summary: failed_summary(128),
        },
        RtMatrixPoint {
            context_tokens: 1024 * 1024,
            query_count: 32,
            candidates_per_query: 1024,
            summary: failed_summary(1024),
        },
    ];

    let json = matrix_json(&config, &points);
    assert!(json.contains("\"mode\":\"experimental_rt_matrix\""));
    assert!(json.contains("\"scope\":\"attention_decode_latency_synthetic_matrix\""));
    assert!(json.contains("\"context_tokens\":[131072,262144,524288,1048576]"));
    assert!(json.contains("\"query_counts\":[1,8,32]"));
    assert!(json.contains("\"candidate_pages\":[128,256,512,1024]"));
    assert!(json.contains("\"context_tokens\":1048576"));
    assert!(json.contains("\"query_count\":32"));
    assert!(json.contains("\"decode_latency_estimate\":"));
}

fn failed_summary(candidates_per_query: u32) -> CudaExperimentalRtCandidateBenchSummary {
    CudaExperimentalRtCandidateBenchSummary::failed(
        64,
        64,
        16,
        1,
        candidates_per_query,
        1,
        8,
        "not run".to_string(),
    )
}
