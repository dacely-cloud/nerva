use crate::artifact::run::run_artifact;
use crate::perf::measurement::PerfMeasurement;
use crate::perf::run::{compare_perf_baseline, perf_baseline_json_from_args};
use crate::probes::projection::{
    run_projection_batch_advance_probe, run_projection_batch_exec_probe, run_projection_batch_plan,
};

const NERVA_QWEN3_8B_PERF: &str =
    include_str!("../../../../docs/source/perf/qwen3_8b_nerva_cuda_generate.json");
const VLLM_QWEN3_8B_PERF: &str =
    include_str!("../../../../docs/source/perf/qwen3_8b_vllm_latency.json");

#[test]
fn perf_baseline_rejects_claims_without_beating_all_baselines() {
    let summary = compare_perf_baseline(
        "qwen3_8b_bf16_decode".to_string(),
        "single_gpu_resident_external_baseline_required".to_string(),
        97.07,
        89.33,
        100.0,
        10.35,
        11.66,
        9.90,
    )
    .unwrap();

    assert!(summary.beats_vllm);
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

#[test]
fn external_baseline_artifact_records_unmeasured_rvllm_status() {
    let json = run_artifact(
        Some("external-baseline".to_string()),
        vec![
            "rvllm".to_string(),
            "qwen3_8b_bf16_decode".to_string(),
            "single_gpu_resident_external_baseline_required".to_string(),
            "unsupported_workload".to_string(),
            "patched rvllm-bench exits before inference with unsupported architecture: Qwen3ForCausalLM".to_string(),
        ],
    )
    .unwrap();

    assert!(json.contains("\"artifact_schema\":\"nerva-bench-v1\""));
    assert!(json.contains("\"schema\":\"nerva-external-baseline-v1\""));
    assert!(json.contains("\"engine\":\"rvllm\""));
    assert!(json.contains("\"baseline_status\":\"unsupported_workload\""));
    assert!(json.contains("\"tokens_per_second\":null"));
    assert!(json.contains("\"claim_blocked\":true"));
}

#[test]
fn perf_measurement_artifacts_prove_current_qwen_vllm_comparison() {
    let nerva = PerfMeasurement::parse("nerva", NERVA_QWEN3_8B_PERF).unwrap();
    let vllm = PerfMeasurement::parse("vllm", VLLM_QWEN3_8B_PERF).unwrap();

    assert_eq!(nerva.engine, "nerva");
    assert_eq!(vllm.engine, "vllm");
    assert!(nerva.matches_workload(&vllm));
    assert!(nerva.beats(&vllm));
    assert_eq!(
        nerva.measurement_id,
        "nerva-qwen3-8b-cuda-generate-2026-06-27"
    );
}

#[test]
fn perf_measurement_rejects_non_positive_or_mismatched_artifacts() {
    let bad = r#"{
        "schema":"nerva-perf-measurement-v1",
        "engine":"nerva",
        "workload":"qwen3_8b_bf16_decode",
        "scope":"single_gpu_resident_external_baseline_required",
        "measurement_id":"bad",
        "tokens_per_second":0,
        "p99_ms":1
    }"#;
    assert!(PerfMeasurement::parse("bad", bad).is_err());

    let mut other = PerfMeasurement::parse("vllm", VLLM_QWEN3_8B_PERF).unwrap();
    other.scope = "different_scope".to_string();
    let nerva = PerfMeasurement::parse("nerva", NERVA_QWEN3_8B_PERF).unwrap();
    assert!(!nerva.matches_workload(&other));
}

#[test]
fn projection_batch_plan_reports_exact_block_reuse() {
    let json = run_projection_batch_plan(8, 8, 8, 2).unwrap();

    assert!(json.contains("\"schema\":\"nerva-projection-batch-plan-v1\""));
    assert!(json.contains("\"plan_reason\":\"ready\""));
    assert!(json.contains("\"exact\":true"));
    assert!(json.contains("\"block_tokens\":8"));
    assert!(json.contains("\"selected_request_ids\":[0,1,2,3,4,5,6,7]"));
    assert!(json.contains("\"ideal_projection_weight_stream_reuse_x1000\":8000"));
    assert!(json.contains("\"executor_status\":\"planner_only\""));
}

#[test]
fn projection_batch_plan_requires_compatible_weight_hash_group() {
    let json = run_projection_batch_plan(8, 1, 8, 2).unwrap();

    assert!(json.contains("\"plan_reason\":\"insufficient_compatible_ready\""));
    assert!(json.contains("\"exact\":false"));
    assert!(json.contains("\"block_tokens\":0"));
    assert!(json.contains("\"ideal_projection_weight_stream_reuse_x1000\":0"));
}

#[test]
fn projection_batch_exec_probe_skips_unproven_batch_before_cuda() {
    let json = run_projection_batch_exec_probe(8, 1, 64, 128, 1, 4, 1, 8, 2).unwrap();

    assert!(json.contains("\"schema\":\"nerva-projection-batch-exec-probe-v1\""));
    assert!(json.contains("\"status\":\"skipped\""));
    assert!(json.contains("\"reason\":\"insufficient_compatible_ready\""));
    assert!(json.contains("\"exact\":false"));
    assert!(json.contains("\"executor_status\":\"not_executed\""));
}

#[test]
fn projection_batch_advance_probe_skips_unproven_batch_before_cuda() {
    let json = run_projection_batch_advance_probe(8, 1, 8, 2).unwrap();

    assert!(json.contains("\"schema\":\"nerva-projection-batch-advance-probe-v1\""));
    assert!(json.contains("\"status\":\"skipped\""));
    assert!(json.contains("\"reason\":\"insufficient_compatible_ready\""));
    assert!(json.contains("\"exact\":false"));
    assert!(json.contains("\"executor_status\":\"not_executed\""));
}
