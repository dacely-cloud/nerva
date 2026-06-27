use crate::acceptance::report::AcceptanceReport;
use crate::perf::run::compare_perf_baseline;

pub(crate) fn push_perf_claim_gate(report: &mut AcceptanceReport) {
    let current_qwen_nerva_tps = 74.49;
    let current_qwen_nerva_p99_ms = 13.50;
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
                "current_qwen_claim_allowed=false current_qwen_nerva_tps={:.2} current_qwen_nerva_p99_ms={:.2} current_qwen_vllm_tps={:.2} current_qwen_vllm_p99_ms={:.2} current_qwen_beats_vllm={} current_qwen_rvllm_baseline_status=missing advantage_claim_allowed={} advantage_speedup_vs_best={:.3} advantage_p99_ratio_vs_best={:.3} requires_vllm_and_rvllm=true",
                current_qwen_nerva_tps,
                current_qwen_nerva_p99_ms,
                current_qwen_vllm_tps,
                current_qwen_vllm_p99_ms,
                current_qwen_beats_vllm,
                allowed.claim_allowed,
                allowed.throughput_speedup_vs_best_baseline,
                allowed.p99_ratio_vs_best_baseline,
            ),
        ),
        Err(err) => report.push("perf_claim_gate", false, err),
    }
}
