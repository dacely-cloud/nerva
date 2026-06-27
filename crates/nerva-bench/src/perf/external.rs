use crate::json::json_escape;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExternalBaselineSummary {
    pub engine: String,
    pub workload: String,
    pub scope: String,
    pub baseline_status: String,
    pub tokens_per_second: Option<f64>,
    pub p99_ms: Option<f64>,
    pub comparable: bool,
    pub claim_blocked: bool,
    pub evidence: String,
}

impl ExternalBaselineSummary {
    pub(crate) fn to_json(&self) -> String {
        format!(
            "{{\"status\":\"ok\",\"schema\":\"nerva-external-baseline-v1\",\"engine\":\"{}\",\"workload\":\"{}\",\"scope\":\"{}\",\"baseline_status\":\"{}\",\"tokens_per_second\":{},\"p99_ms\":{},\"comparable\":{},\"claim_blocked\":{},\"evidence\":\"{}\"}}",
            json_escape(&self.engine),
            json_escape(&self.workload),
            json_escape(&self.scope),
            json_escape(&self.baseline_status),
            optional_number(self.tokens_per_second),
            optional_number(self.p99_ms),
            self.comparable,
            self.claim_blocked,
            json_escape(&self.evidence),
        )
    }
}

pub(crate) fn external_baseline_json_from_args(args: &[String]) -> Result<String, String> {
    let engine = required(args, 0, "engine")?;
    let workload = required(args, 1, "workload")?;
    let scope = required(args, 2, "scope")?;
    let baseline_status = required(args, 3, "baseline_status")?;
    let evidence = required(args, 4, "evidence")?;
    let comparable = baseline_status == "measured";
    let claim_blocked = baseline_status != "measured";
    Ok(ExternalBaselineSummary {
        engine,
        workload,
        scope,
        baseline_status,
        tokens_per_second: None,
        p99_ms: None,
        comparable,
        claim_blocked,
        evidence,
    }
    .to_json())
}

fn required(args: &[String], index: usize, name: &str) -> Result<String, String> {
    args.get(index)
        .cloned()
        .ok_or_else(|| format!("{name} is required"))
}

fn optional_number(value: Option<f64>) -> String {
    match value {
        Some(value) if value.fract() == 0.0 => format!("{value:.1}"),
        Some(value) => value.to_string(),
        None => "null".to_string(),
    }
}
