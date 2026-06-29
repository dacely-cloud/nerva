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
    let mut one = failed_summary(1);
    one.fine_token_projected_candidate_tokens = 1;
    one.fine_token_learned_projected_candidate_tokens = 1;
    let mut four = failed_summary(4);
    four.fine_token_projected_candidate_tokens = 4;
    four.fine_token_learned_projected_candidate_tokens = 4;
    let json = sweep_json(&config, &[one, four]);

    assert!(json.contains("\"mode\":\"experimental_rt_candidate_sweep\""));
    assert!(json.contains("\"scope\":\"recall_candidate_size_synthetic\""));
    assert!(json.contains("\"recall_curve\":"));
    assert!(json.contains("\"scope\":\"recall_vs_candidate_bytes_synthetic\""));
    assert!(json.contains("\"oracle\":\"dense_far_token_topk\""));
    assert!(json.contains("\"bytes_axis\":\"estimated_attention_kv_bytes_per_query\""));
    assert!(json.contains("\"candidates_per_query\":1"));
    assert!(json.contains("\"candidates_per_query\":4"));
    assert!(json.contains("\"far_oracle\":"));
    assert!(json.contains("\"selectors\":["));
    assert!(json.contains("\"selector\":\"synthetic_fine_token_projected_3d\""));
    assert!(json.contains("\"reduction\":\"synthetic_random_projection_3d\""));
    assert!(json.contains("\"selector\":\"synthetic_fine_token_learned_projected_4d\""));
    assert!(json.contains("\"reduction\":\"synthetic_learned_projection_4d\""));
    assert!(json.contains("\"selector\":\"synthetic_page_level_descriptor_topk\""));
    assert!(json.contains("\"norm_sensitivity\":"));
    assert!(json.contains("\"synthetic_norm_augmentation_gain_min_ppm\":0"));
    assert!(json.contains("\"page_level_equal_bytes_baseline\":"));
    assert!(json.contains("\"fine_token_projected_page_byte_bracket\":"));
    assert!(json.contains("\"fine_token_learned_projected_page_byte_bracket\":"));
    assert!(json.contains("\"decode_latency_estimate\":"));
    assert!(json.contains("\"full_decode_latency_measured\":false"));
    assert!(json.contains("\"layer_count\":36"));
    assert!(json.contains("\"scope\":\"synthetic_equal_bytes_measured\""));
    assert!(json.contains("\"selector\":\"synthetic_page_level_descriptor_topk\""));
    assert!(json.contains("\"quest_reproduction_status\":\"pending\""));
    assert!(json.contains("\"attention_mass_recall_min_ppm\":0"));
    assert!(json.contains("\"far_oracle_topk_token_recall_min_ppm\":0"));
    assert!(json.contains("\"far_oracle_topk_importance_scatter_avg_pages_x1000\":0"));
    assert!(json.contains("\"synthetic_far_oracle_topk_page_overfetch_avg_x1000\":0"));
    assert!(json.contains("\"fine_token_projected_token_recall_min_ppm\":0"));
    assert!(json.contains("\"fine_token_learned_projected_token_recall_min_ppm\":0"));
    assert!(
        json.contains("\"synthetic_fine_token_projected_vs_page_candidate_kv_fraction_ppm\":15625")
    );
    assert!(json.contains(
        "\"synthetic_fine_token_learned_projected_vs_page_candidate_kv_fraction_ppm\":15625"
    ));
    assert!(json.contains("\"scope\":\"synthetic_fine_token_page_byte_bracket\""));
    assert!(json.contains("\"scope\":\"synthetic_fine_token_learned_page_byte_bracket\""));
    assert!(json.contains("\"fine_token_candidate_tokens_per_query\":1"));
    assert!(json.contains("\"floor\":{\"candidate_pages_per_query\":0"));
    assert!(json.contains("\"ceil\":{\"candidate_pages_per_query\":1"));
    assert!(json.contains("\"measured\":true"));
    assert!(json.contains("\"norm_stress_no_augmentation_token_recall_min_ppm\":0"));
    assert!(json.contains("\"norm_stress_synthetic_norm_augmented_token_recall_min_ppm\":0"));
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
    assert!(json.contains("\"scope\":\"recall_context_candidate_synthetic_matrix\""));
    assert!(json.contains("\"context_tokens\":[131072,262144,524288,1048576]"));
    assert!(json.contains("\"query_counts\":[1,8,32]"));
    assert!(json.contains("\"candidate_pages\":[128,256,512,1024]"));
    assert!(json.contains("\"recall_curves\":["));
    assert!(json.contains("\"context_tokens\":131072,\"query_count\":1,\"recall_curve\":"));
    assert!(json.contains("\"context_tokens\":1048576"));
    assert!(json.contains("\"query_count\":32"));
    assert!(json.contains("\"scope\":\"recall_vs_candidate_bytes_synthetic\""));
    assert!(json.contains("\"selector\":\"synthetic_fine_token_projected_3d\""));
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
