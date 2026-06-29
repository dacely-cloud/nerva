use crate::experimental_rt::summary::CudaExperimentalRtCandidateBenchSummary;

#[test]
fn experimental_rt_json_reports_synthetic_kv_byte_estimates() {
    let mut summary = CudaExperimentalRtCandidateBenchSummary::failed(
        4,
        8,
        16,
        1,
        2,
        1,
        0,
        "not run".to_string(),
    );
    summary.local_window_tokens = 16;
    summary.far_oracle_topk_tokens = 4;
    summary.far_oracle_topk_importance_scatter_min_pages = 1;
    summary.far_oracle_topk_importance_scatter_avg_pages_x1000 = 1500;
    summary.far_oracle_topk_importance_scatter_max_pages = 2;
    summary.fine_token_projected_topk_tokens = 4;
    summary.fine_token_projected_candidate_tokens = 4;
    summary.fine_token_projected_token_recall_min_ppm = 1_000_000;
    summary.fine_token_projected_token_recall_avg_ppm = 1_000_000;
    summary.fine_token_learned_projected_topk_tokens = 4;
    summary.fine_token_learned_projected_candidate_tokens = 4;
    summary.fine_token_learned_projected_token_recall_min_ppm = 1_000_000;
    summary.fine_token_learned_projected_token_recall_avg_ppm = 1_000_000;

    let json = summary.to_json();

    assert!(json.contains("\"synthetic_kv_bytes_per_token\":128"));
    assert!(json.contains("\"synthetic_dense_full_kv_bytes_per_query\":4096"));
    assert!(json.contains("\"synthetic_local_window_kv_bytes_per_query\":2048"));
    assert!(json.contains("\"synthetic_candidate_page_kv_bytes_per_query\":2048"));
    assert!(json.contains("\"synthetic_estimated_rt_attention_kv_bytes_per_query\":4096"));
    assert!(json.contains("\"synthetic_estimated_rt_vs_dense_kv_fraction_ppm\":1000000"));
    assert!(json.contains("\"synthetic_far_oracle_topk_token_kv_bytes_per_query\":512"));
    assert!(
        json.contains("\"synthetic_far_oracle_topk_scatter_page_kv_bytes_min_per_query\":1024")
    );
    assert!(
        json.contains("\"synthetic_far_oracle_topk_scatter_page_kv_bytes_avg_per_query\":1536")
    );
    assert!(
        json.contains("\"synthetic_far_oracle_topk_scatter_page_kv_bytes_max_per_query\":2048")
    );
    assert!(json.contains("\"synthetic_far_oracle_topk_page_overfetch_avg_x1000\":3000"));
    assert!(json.contains("\"synthetic_fine_token_projected_candidate_kv_bytes_per_query\":512"));
    assert!(
        json.contains(
            "\"synthetic_fine_token_projected_vs_page_candidate_kv_fraction_ppm\":250000"
        )
    );
    assert!(
        json.contains(
            "\"synthetic_fine_token_learned_projected_candidate_kv_bytes_per_query\":512"
        )
    );
    assert!(json.contains(
        "\"synthetic_fine_token_learned_projected_vs_page_candidate_kv_fraction_ppm\":250000"
    ));
    assert!(json.contains("\"page_level_attention_mass_recall_min_ppm\":0"));
    assert!(json.contains("\"page_level_attention_mass_recall_avg_ppm\":0"));
    assert!(json.contains("\"far_oracle_topk_tokens\":4"));
    assert!(json.contains("\"far_oracle_topk_token_recall_min_ppm\":0"));
    assert!(json.contains("\"far_oracle_topk_token_recall_avg_ppm\":0"));
    assert!(json.contains("\"page_level_far_oracle_topk_token_recall_min_ppm\":0"));
    assert!(json.contains("\"page_level_far_oracle_topk_token_recall_avg_ppm\":0"));
    assert!(json.contains("\"far_oracle_topk_importance_scatter_min_pages\":1"));
    assert!(json.contains("\"far_oracle_topk_importance_scatter_avg_pages_x1000\":1500"));
    assert!(json.contains("\"far_oracle_topk_importance_scatter_max_pages\":2"));
    assert!(json.contains("\"fine_token_projected_topk_tokens\":4"));
    assert!(json.contains("\"fine_token_projected_candidate_tokens\":4"));
    assert!(json.contains("\"fine_token_projected_token_recall_min_ppm\":1000000"));
    assert!(json.contains("\"fine_token_projected_token_recall_avg_ppm\":1000000"));
    assert!(json.contains("\"fine_token_learned_projected_topk_tokens\":4"));
    assert!(json.contains("\"fine_token_learned_projected_candidate_tokens\":4"));
    assert!(json.contains("\"fine_token_learned_projected_token_recall_min_ppm\":1000000"));
    assert!(json.contains("\"fine_token_learned_projected_token_recall_avg_ppm\":1000000"));
    assert!(json.contains("\"norm_stress_topk_tokens\":0"));
    assert!(json.contains("\"norm_stress_no_augmentation_token_recall_min_ppm\":0"));
    assert!(json.contains("\"norm_stress_no_augmentation_token_recall_avg_ppm\":0"));
    assert!(json.contains("\"norm_stress_synthetic_norm_augmented_token_recall_min_ppm\":0"));
    assert!(json.contains("\"norm_stress_synthetic_norm_augmented_token_recall_avg_ppm\":0"));
}
