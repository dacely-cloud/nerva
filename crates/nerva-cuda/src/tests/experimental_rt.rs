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

    let json = summary.to_json();

    assert!(json.contains("\"synthetic_kv_bytes_per_token\":128"));
    assert!(json.contains("\"synthetic_dense_full_kv_bytes_per_query\":4096"));
    assert!(json.contains("\"synthetic_local_window_kv_bytes_per_query\":2048"));
    assert!(json.contains("\"synthetic_candidate_page_kv_bytes_per_query\":2048"));
    assert!(json.contains("\"synthetic_estimated_rt_attention_kv_bytes_per_query\":4096"));
    assert!(json.contains("\"synthetic_estimated_rt_vs_dense_kv_fraction_ppm\":1000000"));
}
