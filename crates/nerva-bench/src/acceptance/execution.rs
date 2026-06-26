use nerva_runtime::engine::compute_near_data::config::ComputeNearDataProbeConfig;
use nerva_runtime::engine::compute_near_data::summary::ComputeNearDataProbeStatus;
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

pub(crate) fn push_compute_near_data(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_compute_near_data_probe(ComputeNearDataProbeConfig::default()) {
        Ok(summary) => report.push(
            "compute_near_data_resident_blocks",
            matches!(summary.status, ComputeNearDataProbeStatus::Ok)
                && summary.parity
                && summary.blocks == 2
                && summary.dram_blocks == 1
                && summary.vram_blocks == 1
                && summary.execution_decisions == 2
                && summary.block_version_dependencies == 2
                && summary.cpu_events == 1
                && summary.device_events == 1
                && summary.copy_events == 1
                && summary.merge_bytes == 8
                && summary.hot_path_allocations == 0,
            format!(
                "rows={} cols={} split_row={} blocks={} dram_blocks={} vram_blocks={} parity={} max_abs_error={} execution_decisions={} block_version_dependencies={} cpu_events={} device_events={} copy_events={} merge_bytes={} hot_path_allocations={}",
                summary.rows,
                summary.cols,
                summary.split_row,
                summary.blocks,
                summary.dram_blocks,
                summary.vram_blocks,
                summary.parity,
                summary.max_abs_error,
                summary.execution_decisions,
                summary.block_version_dependencies,
                summary.cpu_events,
                summary.device_events,
                summary.copy_events,
                summary.merge_bytes,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("compute_near_data_resident_blocks", false, format!("{err:?}")),
    }
}
