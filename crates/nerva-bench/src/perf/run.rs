use crate::parse::parse_required_f64;
use crate::perf::summary::PerfBaselineSummary;

pub(crate) fn perf_baseline_json_from_args(args: &[String]) -> Result<String, String> {
    let workload = args
        .first()
        .cloned()
        .ok_or_else(|| "workload is required".to_string())?;
    let scope = args
        .get(1)
        .cloned()
        .ok_or_else(|| "scope is required".to_string())?;
    perf_baseline_json(
        workload,
        scope,
        parse_required_f64(args.get(2).cloned(), "nerva_tokens_per_second")?,
        parse_required_f64(args.get(3).cloned(), "vllm_tokens_per_second")?,
        parse_required_f64(args.get(4).cloned(), "rvllm_tokens_per_second")?,
        parse_required_f64(args.get(5).cloned(), "nerva_p99_ms")?,
        parse_required_f64(args.get(6).cloned(), "vllm_p99_ms")?,
        parse_required_f64(args.get(7).cloned(), "rvllm_p99_ms")?,
    )
}

pub(crate) fn perf_baseline_json(
    workload: String,
    scope: String,
    nerva_tokens_per_second: f64,
    vllm_tokens_per_second: f64,
    rvllm_tokens_per_second: f64,
    nerva_p99_ms: f64,
    vllm_p99_ms: f64,
    rvllm_p99_ms: f64,
) -> Result<String, String> {
    let summary = compare_perf_baseline(
        workload,
        scope,
        nerva_tokens_per_second,
        vllm_tokens_per_second,
        rvllm_tokens_per_second,
        nerva_p99_ms,
        vllm_p99_ms,
        rvllm_p99_ms,
    )?;
    Ok(summary.to_json())
}

pub(crate) fn compare_perf_baseline(
    workload: String,
    scope: String,
    nerva_tokens_per_second: f64,
    vllm_tokens_per_second: f64,
    rvllm_tokens_per_second: f64,
    nerva_p99_ms: f64,
    vllm_p99_ms: f64,
    rvllm_p99_ms: f64,
) -> Result<PerfBaselineSummary, String> {
    require_positive("nerva_tokens_per_second", nerva_tokens_per_second)?;
    require_positive("vllm_tokens_per_second", vllm_tokens_per_second)?;
    require_positive("rvllm_tokens_per_second", rvllm_tokens_per_second)?;
    require_positive("nerva_p99_ms", nerva_p99_ms)?;
    require_positive("vllm_p99_ms", vllm_p99_ms)?;
    require_positive("rvllm_p99_ms", rvllm_p99_ms)?;
    let best_baseline_tps = vllm_tokens_per_second.max(rvllm_tokens_per_second);
    let best_baseline_p99 = vllm_p99_ms.min(rvllm_p99_ms);
    let beats_vllm = nerva_tokens_per_second > vllm_tokens_per_second && nerva_p99_ms < vllm_p99_ms;
    let beats_rvllm =
        nerva_tokens_per_second > rvllm_tokens_per_second && nerva_p99_ms < rvllm_p99_ms;
    Ok(PerfBaselineSummary {
        workload,
        scope,
        nerva_tokens_per_second,
        vllm_tokens_per_second,
        rvllm_tokens_per_second,
        nerva_p99_ms,
        vllm_p99_ms,
        rvllm_p99_ms,
        throughput_speedup_vs_best_baseline: nerva_tokens_per_second / best_baseline_tps,
        p99_ratio_vs_best_baseline: nerva_p99_ms / best_baseline_p99,
        beats_vllm,
        beats_rvllm,
        claim_allowed: beats_vllm && beats_rvllm,
    })
}

fn require_positive(label: &'static str, value: f64) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!("{label} must be a finite positive number"))
    }
}
