use nerva_runtime::engine::runtime::Runtime;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_phase_handoff(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_phase_handoff_probe() {
        Ok(summary) => report.push(
            "phase_handoff_ownership",
            summary.passed()
                && summary.planned_handoffs == 3
                && summary.applied_handoffs == 3
                && summary.rejected_handoffs == 4
                && summary.owner_mismatch_rejections == 1
                && summary.stale_version_rejections == 1
                && summary.unready_rejections == 1
                && summary.illegal_transition_rejections == 1
                && summary.phase_handoff_syncs == summary.applied_handoffs
                && summary.version_publications == summary.applied_handoffs
                && summary.hot_path_allocations == 0,
            format!(
                "planned_handoffs={} applied_handoffs={} rejected_handoffs={} owner_mismatch_rejections={} stale_version_rejections={} unready_rejections={} illegal_transition_rejections={} phase_handoff_syncs={} version_publications={} final_max_version={} hot_path_allocations={}",
                summary.planned_handoffs,
                summary.applied_handoffs,
                summary.rejected_handoffs,
                summary.owner_mismatch_rejections,
                summary.stale_version_rejections,
                summary.unready_rejections,
                summary.illegal_transition_rejections,
                summary.phase_handoff_syncs,
                summary.version_publications,
                summary.final_max_version,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("phase_handoff_ownership", false, format!("{err:?}")),
    }
}
