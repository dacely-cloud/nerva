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

pub(crate) fn push_request_scheduler(report: &mut AcceptanceReport) {
    match nerva_runtime::request::scheduler::probe::run_request_scheduler_probe() {
        Ok(summary) => report.push(
            "request_scheduler_admission",
            summary.passed(),
            format!(
                "capacity={} admitted={} active={} completed={} full_rejections={} duplicate_rejections={} missing_request_rejections={} iterations={} max_active={} generated_tokens={} host_observed_tokens={} token_ledgers={} critical_path_reports={} graph_replay_events={} device_activity_events={} copy_events={} soft_visibility_syncs={} host_event_wait_ns={} gpu_idle_ns={} estimated_events={} runtime_timestamp_events={} unclassified_syncs={} bounded_slots={} unbounded_queue_ops={} host_wait_gpu_idle_separated={} hot_path_allocations={}",
                summary.capacity,
                summary.admitted_requests,
                summary.active_requests,
                summary.completed_requests,
                summary.full_rejections,
                summary.duplicate_rejections,
                summary.missing_request_rejections,
                summary.scheduler_iterations,
                summary.max_active_requests,
                summary.generated_tokens,
                summary.host_observed_tokens,
                summary.token_ledgers,
                summary.critical_path_reports,
                summary.graph_replay_events,
                summary.device_activity_events,
                summary.copy_events,
                summary.soft_visibility_syncs,
                summary.host_event_wait_ns,
                summary.gpu_idle_ns,
                summary.estimated_events,
                summary.runtime_timestamp_events,
                summary.unclassified_syncs,
                summary.bounded_slots,
                summary.unbounded_queue_ops,
                summary.host_wait_gpu_idle_separated,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("request_scheduler_admission", false, format!("{err:?}")),
    }
}
