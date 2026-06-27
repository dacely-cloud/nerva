use crate::acceptance::report::AcceptanceReport;
use crate::perf::run::compare_perf_baseline;

pub(crate) fn push_perf_claim_gate(report: &mut AcceptanceReport) {
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
                "current_qwen_claim_allowed=false current_qwen_nerva_tps=71.76 current_qwen_nerva_p99_ms=14.10 current_qwen_external_baseline_status=missing advantage_claim_allowed={} advantage_speedup_vs_best={:.3} advantage_p99_ratio_vs_best={:.3} requires_vllm_and_rvllm=true",
                allowed.claim_allowed,
                allowed.throughput_speedup_vs_best_baseline,
                allowed.p99_ratio_vs_best_baseline,
            ),
        ),
        Err(err) => report.push("perf_claim_gate", false, err),
    }
}
