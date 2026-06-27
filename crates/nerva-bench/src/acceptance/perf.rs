use crate::acceptance::report::AcceptanceReport;
use crate::perf::run::compare_perf_baseline;

pub(crate) fn push_perf_claim_gate(report: &mut AcceptanceReport) {
    let rejected = compare_perf_baseline(
        "qwen3_8b_bf16_decode".to_string(),
        "single_gpu_resident_current".to_string(),
        0.39,
        40.0,
        35.0,
        2584.0,
        80.0,
        95.0,
    );
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
    match (rejected, allowed) {
        (Ok(rejected), Ok(allowed)) => report.push(
            "perf_claim_gate",
            !rejected.claim_allowed && allowed.claim_allowed,
            format!(
                "current_qwen_claim_allowed={} current_speedup_vs_best={:.6} current_p99_ratio_vs_best={:.3} advantage_claim_allowed={} advantage_speedup_vs_best={:.3} advantage_p99_ratio_vs_best={:.3} requires_vllm_and_rvllm=true",
                rejected.claim_allowed,
                rejected.throughput_speedup_vs_best_baseline,
                rejected.p99_ratio_vs_best_baseline,
                allowed.claim_allowed,
                allowed.throughput_speedup_vs_best_baseline,
                allowed.p99_ratio_vs_best_baseline,
            ),
        ),
        (Err(err), _) | (_, Err(err)) => report.push("perf_claim_gate", false, err),
    }
}
