use crate::json::json_escape;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PerfBaselineSummary {
    pub workload: String,
    pub scope: String,
    pub nerva_tokens_per_second: f64,
    pub vllm_tokens_per_second: f64,
    pub rvllm_tokens_per_second: f64,
    pub nerva_p99_ms: f64,
    pub vllm_p99_ms: f64,
    pub rvllm_p99_ms: f64,
    pub throughput_speedup_vs_best_baseline: f64,
    pub p99_ratio_vs_best_baseline: f64,
    pub beats_vllm: bool,
    pub beats_rvllm: bool,
    pub claim_allowed: bool,
}

impl PerfBaselineSummary {
    pub(crate) fn to_json(&self) -> String {
        format!(
            "{{\"status\":\"ok\",\"schema\":\"nerva-perf-baseline-v1\",\"workload\":\"{}\",\"scope\":\"{}\",\"nerva_tokens_per_second\":{},\"vllm_tokens_per_second\":{},\"rvllm_tokens_per_second\":{},\"nerva_p99_ms\":{},\"vllm_p99_ms\":{},\"rvllm_p99_ms\":{},\"throughput_speedup_vs_best_baseline\":{},\"p99_ratio_vs_best_baseline\":{},\"beats_vllm\":{},\"beats_rvllm\":{},\"claim_allowed\":{}}}",
            json_escape(&self.workload),
            json_escape(&self.scope),
            json_number(self.nerva_tokens_per_second),
            json_number(self.vllm_tokens_per_second),
            json_number(self.rvllm_tokens_per_second),
            json_number(self.nerva_p99_ms),
            json_number(self.vllm_p99_ms),
            json_number(self.rvllm_p99_ms),
            json_number(self.throughput_speedup_vs_best_baseline),
            json_number(self.p99_ratio_vs_best_baseline),
            self.beats_vllm,
            self.beats_rvllm,
            self.claim_allowed,
        )
    }
}

fn json_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}
