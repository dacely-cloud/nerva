use crate::artifact::run::run_artifact;
use crate::perf::run::{compare_perf_baseline, perf_baseline_json_from_args};

#[test]
fn perf_baseline_rejects_slow_nerva_claims() {
    let summary = compare_perf_baseline(
        "qwen3_8b_bf16_decode".to_string(),
        "single_gpu_resident_external_baseline_required".to_string(),
        85.07,
        89.33,
        88.0,
        11.99,
        11.66,
        11.80,
    )
    .unwrap();

    assert!(!summary.beats_vllm);
    assert!(!summary.beats_rvllm);
    assert!(!summary.claim_allowed);
    assert!(summary.throughput_speedup_vs_best_baseline < 1.0);
    assert!(summary.p99_ratio_vs_best_baseline > 1.0);
}

#[test]
fn perf_baseline_allows_only_faster_and_lower_tail_results() {
    let json = perf_baseline_json_from_args(&[
        "tiered_kv_advantage_case".to_string(),
        "claimed_advantage_zone".to_string(),
        "55".to_string(),
        "40".to_string(),
        "42".to_string(),
        "45".to_string(),
        "60".to_string(),
        "58".to_string(),
    ])
    .unwrap();

    assert!(json.contains("\"schema\":\"nerva-perf-baseline-v1\""));
    assert!(json.contains("\"beats_vllm\":true"));
    assert!(json.contains("\"beats_rvllm\":true"));
    assert!(json.contains("\"claim_allowed\":true"));
}

#[test]
fn perf_baseline_artifact_wraps_comparison_evidence() {
    let json = run_artifact(
        Some("perf-baseline".to_string()),
        vec![
            "larger_than_vram_decode".to_string(),
            "claimed_advantage_zone".to_string(),
            "60".to_string(),
            "40".to_string(),
            "45".to_string(),
            "50".to_string(),
            "70".to_string(),
            "65".to_string(),
        ],
    )
    .unwrap();

    assert!(json.contains("\"artifact_schema\":\"nerva-bench-v1\""));
    assert!(json.contains("\"command\":\"perf-baseline\""));
    assert!(json.contains("\"summary\""));
    assert!(json.contains("\"claim_allowed\":true"));
}
