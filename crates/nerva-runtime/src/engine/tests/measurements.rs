use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::measurements::planner::summary::MeasuredPlannerStatus;
use crate::measurements::summary::MeasurementTableStatus;

#[test]
fn measurement_table_probe_records_runtime_timestamp_entries() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_measurement_table_probe().unwrap();

    assert_eq!(summary.status, MeasurementTableStatus::Ok);
    assert!(summary.passed());
    assert_eq!(summary.measured_entries, 5);
    assert_eq!(summary.estimated_entries, 0);
    assert_eq!(summary.runtime_timestamp_entries, summary.measured_entries);
    assert_eq!(summary.cpu_copy_entries, 1);
    assert_eq!(summary.cpu_kernel_entries, 1);
    assert_eq!(summary.merge_entries, 1);
    assert_eq!(summary.queue_entries, 1);
    assert_eq!(summary.sync_entries, 1);
    assert!(summary.total_latency_ns > 0);
    assert!(summary.min_effective_bandwidth_bps > 0);
    assert!(summary.all_nonzero_latency);
    assert!(summary.all_measured);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(
        summary
            .to_json()
            .contains("\"metric_source\":\"runtime_timestamp\"")
    );
    assert!(summary.to_json().contains("\"estimated_entries\":0"));
}

#[test]
fn measured_planner_records_measured_execution_decision() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_measured_planner_probe().unwrap();

    assert_eq!(summary.status, MeasuredPlannerStatus::Ok);
    assert!(summary.passed());
    assert!(summary.source_measurements >= 5);
    assert_eq!(
        summary.runtime_timestamp_entries,
        summary.source_measurements
    );
    assert_eq!(summary.execution_decisions, 1);
    assert!(summary.candidate_count >= 3);
    assert_eq!(summary.measured_candidate_costs, summary.candidate_count);
    assert_eq!(summary.estimated_candidate_costs, 0);
    assert!(summary.predicted_visible_ns > 0);
    assert_eq!(summary.actual_visible_ns, summary.predicted_visible_ns);
    assert_eq!(summary.decision_metric_source, "runtime_timestamp");
    assert!(summary.all_candidates_measured);
    assert!(summary.selected_cost_measured);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"measured_candidate_costs\":"));
    assert!(
        summary
            .to_json()
            .contains("\"estimated_candidate_costs\":0")
    );
}
