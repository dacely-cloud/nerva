use nerva_runtime::engine::runtime::Runtime;
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
