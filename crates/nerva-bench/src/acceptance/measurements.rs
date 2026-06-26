use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::measurements::planner::summary::MeasuredPlannerStatus;
use nerva_runtime::measurements::summary::MeasurementTableStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_measurement_table(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_measurement_table_probe() {
        Ok(summary) => report.push(
            "measurement_table_bootstrap",
            matches!(summary.status, MeasurementTableStatus::Ok)
                && summary.passed()
                && summary.measured_entries >= 5
                && summary.estimated_entries == 0
                && summary.runtime_timestamp_entries == summary.measured_entries
                && summary.cpu_copy_entries > 0
                && summary.cpu_kernel_entries > 0
                && summary.merge_entries > 0
                && summary.queue_entries > 0
                && summary.sync_entries > 0
                && summary.total_latency_ns > 0
                && summary.min_effective_bandwidth_bps > 0
                && summary.all_nonzero_latency
                && summary.all_measured
                && summary.hot_path_allocations == 0,
            format!(
                "measured_entries={} estimated_entries={} runtime_timestamp_entries={} cpu_copy_entries={} cpu_kernel_entries={} merge_entries={} queue_entries={} sync_entries={} total_latency_ns={} min_effective_bandwidth_bps={} all_nonzero_latency={} all_measured={} hot_path_allocations={}",
                summary.measured_entries,
                summary.estimated_entries,
                summary.runtime_timestamp_entries,
                summary.cpu_copy_entries,
                summary.cpu_kernel_entries,
                summary.merge_entries,
                summary.queue_entries,
                summary.sync_entries,
                summary.total_latency_ns,
                summary.min_effective_bandwidth_bps,
                summary.all_nonzero_latency,
                summary.all_measured,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push(
            "measurement_table_bootstrap",
            false,
            format!("measurement table probe failed: {err:?}"),
        ),
    }
}

pub(crate) fn push_measured_planner(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_measured_planner_probe() {
        Ok(summary) => report.push(
            "measured_planner_decision",
            matches!(summary.status, MeasuredPlannerStatus::Ok)
                && summary.passed()
                && summary.source_measurements >= 5
                && summary.runtime_timestamp_entries == summary.source_measurements
                && summary.execution_decisions == 1
                && summary.candidate_count >= 3
                && summary.measured_candidate_costs == summary.candidate_count
                && summary.estimated_candidate_costs == 0
                && summary.predicted_visible_ns > 0
                && summary.actual_visible_ns == summary.predicted_visible_ns
                && summary.decision_metric_source == "runtime_timestamp"
                && summary.all_candidates_measured
                && summary.selected_cost_measured
                && summary.hot_path_allocations == 0,
            format!(
                "source_measurements={} runtime_timestamp_entries={} execution_decisions={} candidate_count={} measured_candidate_costs={} estimated_candidate_costs={} selected_label={} selected_executor={} predicted_visible_ns={} actual_visible_ns={} decision_metric_source={} all_candidates_measured={} selected_cost_measured={} hot_path_allocations={}",
                summary.source_measurements,
                summary.runtime_timestamp_entries,
                summary.execution_decisions,
                summary.candidate_count,
                summary.measured_candidate_costs,
                summary.estimated_candidate_costs,
                summary.selected_label,
                summary.selected_executor,
                summary.predicted_visible_ns,
                summary.actual_visible_ns,
                summary.decision_metric_source,
                summary.all_candidates_measured,
                summary.selected_cost_measured,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push(
            "measured_planner_decision",
            false,
            format!("measured planner probe failed: {err:?}"),
        ),
    }
}
