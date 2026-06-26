use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::transport::measured::summary::MeasuredTransportSelectorStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_measured_transport_selector(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_measured_transport_selector_probe() {
        Ok(summary) => report.push(
            "measured_transport_selector",
            matches!(summary.status, MeasuredTransportSelectorStatus::Ok)
                && summary.passed()
                && summary.request_bytes == 32 * 1024
                && summary.source_entries >= 3
                && summary.runtime_timestamp_events > 0
                && summary.execution_decisions == 1
                && summary.candidate_count >= 3
                && summary.measured_candidate_costs == summary.candidate_count
                && summary.estimated_candidate_costs == 0
                && summary.selected_bucket_payload_bytes == summary.request_bytes
                && summary.selected_measured_p95_ns > 0
                && summary.selected_visible_ns >= summary.selected_measured_p95_ns
                && summary.selected_bandwidth_bps > 0
                && summary.decision_metric_source == "runtime_timestamp"
                && summary.selected_cost_measured
                && summary.all_candidates_measured
                && summary.packet_loss == 0
                && summary.checksum_failures == 0
                && summary.hot_path_allocations == 0,
            format!(
                "request_bytes={} source_entries={} runtime_timestamp_events={} execution_decisions={} candidate_count={} measured_candidate_costs={} estimated_candidate_costs={} selected_label={} selected_bucket_payload_bytes={} selected_p95_ns={} selected_visible_ns={} selected_bandwidth_bps={} decision_metric_source={} selected_cost_measured={} all_candidates_measured={} packet_loss={} checksum_failures={} hot_path_allocations={}",
                summary.request_bytes,
                summary.source_entries,
                summary.runtime_timestamp_events,
                summary.execution_decisions,
                summary.candidate_count,
                summary.measured_candidate_costs,
                summary.estimated_candidate_costs,
                summary.selected_label,
                summary.selected_bucket_payload_bytes,
                summary.selected_measured_p95_ns,
                summary.selected_visible_ns,
                summary.selected_bandwidth_bps,
                summary.decision_metric_source,
                summary.selected_cost_measured,
                summary.all_candidates_measured,
                summary.packet_loss,
                summary.checksum_failures,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("measured_transport_selector", false, format!("{err:?}")),
    }
}
