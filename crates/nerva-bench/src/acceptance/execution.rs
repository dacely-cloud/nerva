use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::execution::summary::ExecutionTransactionStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_transaction_planner(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_execution_transaction_probe() {
        Ok(summary) => report.push(
            "execution_transaction_planner",
            matches!(summary.status, ExecutionTransactionStatus::Ok)
                && summary.passed()
                && summary.block_version_dependencies == summary.block_uses
                && summary.execution_decisions == summary.operations
                && summary.hard_syncs > 0
                && summary.soft_visibility_syncs > 0
                && summary.phase_handoff_syncs > 0
                && summary.debug_syncs == 0
                && summary.hot_path_allocations == 0
                && summary.stale_dependencies == 0
                && summary.unclassified_syncs == 0,
            format!(
                "operations={} graph_capturable={} block_uses={} block_version_dependencies={} execution_decisions={} hard_syncs={} soft_visibility_syncs={} phase_handoff_syncs={} device_active_ns={} host_event_wait_ns={} hot_path_allocations={} stale_dependencies={} unclassified_syncs={}",
                summary.operations,
                summary.graph_capturable_operations,
                summary.block_uses,
                summary.block_version_dependencies,
                summary.execution_decisions,
                summary.hard_syncs,
                summary.soft_visibility_syncs,
                summary.phase_handoff_syncs,
                summary.device_active_ns,
                summary.host_event_wait_ns,
                summary.hot_path_allocations,
                summary.stale_dependencies,
                summary.unclassified_syncs,
            ),
        ),
        Err(err) => report.push("execution_transaction_planner", false, format!("{err:?}")),
    }
}
