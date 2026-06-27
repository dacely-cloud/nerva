use crate::acceptance::report::AcceptanceReport;
use crate::artifact::run::run_artifact;
use crate::perf::run::compare_perf_baseline;

pub(crate) fn push_perf_claim_gate(report: &mut AcceptanceReport) {
    let rvllm_status = "compile_failed";
    let rvllm_evidence = "rvllm-bench build failed at /root/rvllm commit 17b1c85dff7cea3cc6259f19fce394d6cfea002e with CUDA_HOME=/usr/local/cuda-13.1: rvllm-loader missing Gemma4LayerWeights and Gemma4LoadedModel fields";
    push_rvllm_baseline_status(report, rvllm_status, rvllm_evidence);
    let current_qwen_nerva_tps = 97.09;
    let current_qwen_nerva_p99_ms = 10.35;
    let current_qwen_vllm_tps = 89.33;
    let current_qwen_vllm_p99_ms = 11.66;
    let current_qwen_beats_vllm = current_qwen_nerva_tps > current_qwen_vllm_tps
        && current_qwen_nerva_p99_ms < current_qwen_vllm_p99_ms;
    let allowed = compare_perf_baseline(
        "tiered_kv_advantage_case".to_string(),
        "claimed_advantage_zone".to_string(),
        55.0,
        40.0,
        42.0,
        45.0,
        60.0,
        58.0,
    );
    match allowed {
        Ok(allowed) => report.push(
            "perf_claim_gate",
            allowed.claim_allowed,
            format!(
                "current_qwen_claim_allowed=false current_qwen_nerva_tps={:.2} current_qwen_nerva_p99_ms={:.2} current_qwen_vllm_tps={:.2} current_qwen_vllm_p99_ms={:.2} current_qwen_beats_vllm={} current_qwen_rvllm_baseline_status={} advantage_claim_allowed={} advantage_speedup_vs_best={:.3} advantage_p99_ratio_vs_best={:.3} requires_vllm_and_rvllm=true",
                current_qwen_nerva_tps,
                current_qwen_nerva_p99_ms,
                current_qwen_vllm_tps,
                current_qwen_vllm_p99_ms,
                current_qwen_beats_vllm,
                rvllm_status,
                allowed.claim_allowed,
                allowed.throughput_speedup_vs_best_baseline,
                allowed.p99_ratio_vs_best_baseline,
            ),
        ),
        Err(err) => report.push("perf_claim_gate", false, err),
    }
}

fn push_rvllm_baseline_status(
    report: &mut AcceptanceReport,
    baseline_status: &str,
    evidence: &str,
) {
    let artifact = run_artifact(
        Some("external-baseline".to_string()),
        vec![
            "rvllm".to_string(),
            "qwen3_8b_bf16_decode".to_string(),
            "single_gpu_resident_external_baseline_required".to_string(),
            baseline_status.to_string(),
            evidence.to_string(),
        ],
    );
    match artifact {
        Ok(json) => report.push(
            "rvllm_external_baseline",
            json.contains("\"engine\":\"rvllm\"")
                && json.contains("\"baseline_status\":\"compile_failed\"")
                && json.contains("\"claim_blocked\":true"),
            format!(
                "status={} artifact_schema={} claim_blocked={} evidence_recorded={}",
                baseline_status,
                json.contains("\"artifact_schema\":\"nerva-bench-v1\""),
                json.contains("\"claim_blocked\":true"),
                json.contains("rvllm-loader missing Gemma4LayerWeights"),
            ),
        ),
        Err(err) => report.push("rvllm_external_baseline", false, err),
    }
}
