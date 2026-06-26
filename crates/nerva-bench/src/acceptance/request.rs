use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_request_state(report: &mut AcceptanceReport) {
    match nerva_runtime::request::probe::run_request_state_probe() {
        Ok(summary) => report.push(
            "request_state_machine",
            summary.passed(),
            format!(
                "prompt_tokens={} generated_tokens={} host_observed_tokens={} seed_from_prompt={} device_generated_edges={} device_without_host={} max_host_lag={} stop_reason={} duplicate_rejections={} missing_rejections={} post_completion_rejections={} ledger_count={} device_events={} hot_path_allocations={}",
                summary.prompt_tokens.len(),
                summary.generated_tokens.len(),
                summary.host_observed_tokens.len(),
                summary.seed_from_prompt,
                summary.device_generated_edges,
                summary.device_steps_without_host_observation,
                summary.max_host_visibility_lag,
                summary.stop_reason.as_str(),
                summary.duplicate_row_rejections,
                summary.missing_row_rejections,
                summary.post_completion_rejections,
                summary.ledger_count,
                summary.device_events,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("request_state_machine", false, format!("{err:?}")),
    }
}
