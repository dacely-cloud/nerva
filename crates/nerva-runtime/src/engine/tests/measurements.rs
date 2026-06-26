use crate::engine::runtime::{Runtime, RuntimeConfig};
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
