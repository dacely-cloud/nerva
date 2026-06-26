use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::token::policy::summary::TokenPolicyStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_token_policy(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_token_policy_probe() {
        Ok(summary) => report.push(
            "token_policy_paths",
            matches!(summary.status, TokenPolicyStatus::Ok)
                && summary.passed()
                && summary.host_policy_steps == 1
                && summary.policy_syncs == 1
                && summary.host_causality_edges == summary.host_policy_steps
                && summary.device_fast_host_dependencies == 0
                && summary.soft_visibility_syncs == summary.steps
                && summary.hot_path_allocations == 0,
            format!(
                "steps={} device_fast_steps={} host_policy_steps={} hybrid_validation_steps={} seed_edges={} device_ring_edges={} host_causality_edges={} policy_syncs={} soft_visibility_syncs={} host_visibility_hard_dependencies={} device_fast_host_dependencies={} graph_replays={} observed_tokens={} mismatched_tokens={} hot_path_allocations={}",
                summary.steps,
                summary.device_fast_steps,
                summary.host_policy_steps,
                summary.hybrid_validation_steps,
                summary.seed_edges,
                summary.device_ring_edges,
                summary.host_causality_edges,
                summary.policy_syncs,
                summary.soft_visibility_syncs,
                summary.host_visibility_hard_dependencies,
                summary.device_fast_host_dependencies,
                summary.graph_replays,
                summary.observed_tokens,
                summary.mismatched_tokens,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("token_policy_paths", false, format!("{err:?}")),
    }
}
