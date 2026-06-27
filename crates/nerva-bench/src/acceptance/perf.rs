use crate::acceptance::report::AcceptanceReport;
use crate::artifact::run::run_artifact;
use crate::perf::measurement::PerfMeasurement;

const NERVA_QWEN3_8B_PERF: &str =
    include_str!("../../../../docs/source/perf/qwen3_8b_nerva_cuda_generate.json");
const VLLM_QWEN3_8B_PERF: &str =
    include_str!("../../../../docs/source/perf/qwen3_8b_vllm_latency.json");

pub(crate) fn push_perf_claim_gate(report: &mut AcceptanceReport) {
    let rvllm_status = "unsupported_workload";
    let rvllm_evidence = "rvllm commit 17b1c85dff7cea3cc6259f19fce394d6cfea002e does not provide a comparable Qwen3-8B baseline: unpatched rvllm-bench fails to compile from stale Gemma4 PLE initializers; a temporary compatibility-patched checkout builds, then rvllm-bench exits before inference with unsupported architecture: Qwen3ForCausalLM";
    push_rvllm_baseline_status(report, rvllm_status, rvllm_evidence);
    match qwen_claim_gate_details(rvllm_status) {
        Ok((passed, details)) => report.push("perf_claim_gate", passed, details),
        Err(err) => report.push("perf_claim_gate", false, err),
    }
}

fn qwen_claim_gate_details(rvllm_status: &str) -> Result<(bool, String), String> {
    let nerva = PerfMeasurement::parse("nerva_qwen3_8b", NERVA_QWEN3_8B_PERF)?;
    let vllm = PerfMeasurement::parse("vllm_qwen3_8b", VLLM_QWEN3_8B_PERF)?;
    if !nerva.matches_workload(&vllm) {
        return Err("NERVA and vLLM perf artifacts describe different workloads".to_string());
    }
    let beats_vllm = nerva.beats(&vllm);
    let rvllm_comparable = rvllm_status == "measured";
    let current_claim_allowed = beats_vllm && rvllm_comparable;
    let passed = beats_vllm && !current_claim_allowed;
    Ok((
        passed,
        format!(
            "qwen_artifacts_loaded=true current_qwen_claim_allowed={} current_qwen_nerva_artifact={} current_qwen_vllm_artifact={} current_qwen_nerva_tps={:.2} current_qwen_nerva_p99_ms={:.2} current_qwen_vllm_tps={:.2} current_qwen_vllm_p99_ms={:.2} current_qwen_beats_vllm={} current_qwen_rvllm_baseline_status={} current_qwen_rvllm_comparable={} requires_vllm_and_rvllm=true",
            current_claim_allowed,
            nerva.measurement_id,
            vllm.measurement_id,
            nerva.tokens_per_second,
            nerva.p99_ms,
            vllm.tokens_per_second,
            vllm.p99_ms,
            beats_vllm,
            rvllm_status,
            rvllm_comparable,
        ),
    ))
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
                && json.contains("\"baseline_status\":\"unsupported_workload\"")
                && json.contains("\"claim_blocked\":true"),
            format!(
                "status={} artifact_schema={} claim_blocked={} evidence_recorded={}",
                baseline_status,
                json.contains("\"artifact_schema\":\"nerva-bench-v1\""),
                json.contains("\"claim_blocked\":true"),
                json.contains("unsupported architecture: Qwen3ForCausalLM"),
            ),
        ),
        Err(err) => report.push("rvllm_external_baseline", false, err),
    }
}
