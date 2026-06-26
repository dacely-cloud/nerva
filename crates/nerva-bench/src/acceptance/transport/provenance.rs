use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::transport::provenance::summary::TransportMetricProvenanceStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_transport_metric_provenance(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_transport_metric_provenance_probe() {
        Ok(summary) => report.push(
            "transport_metric_provenance",
            matches!(summary.status, TransportMetricProvenanceStatus::Ok)
                && summary.passed()
                && summary.compared_sizes >= 3
                && summary.runtime_timestamp_events == summary.compared_sizes
                && summary.estimated_model_events == summary.compared_sizes
                && summary.transport_events == summary.compared_sizes * 2
                && summary.measured_event_mislabels == 0
                && summary.estimated_event_mislabels == 0
                && !summary.estimated_presented_as_measured
                && summary.sources_separated
                && summary.total_measured_p95_ns > 0
                && summary.total_estimated_visible_ns > 0
                && summary.packet_loss == 0
                && summary.checksum_failures == 0
                && summary.hot_path_allocations == 0,
            format!(
                "compared_sizes={} measured_matrix_entries={} estimated_matrix_entries={} runtime_timestamp_events={} estimated_model_events={} transport_events={} measured_event_mislabels={} estimated_event_mislabels={} estimated_presented_as_measured={} sources_separated={} total_measured_p95_ns={} total_estimated_visible_ns={} min_ratio_per_mille={} max_ratio_per_mille={} packet_loss={} checksum_failures={} hot_path_allocations={}",
                summary.compared_sizes,
                summary.measured_matrix_entries,
                summary.estimated_matrix_entries,
                summary.runtime_timestamp_events,
                summary.estimated_model_events,
                summary.transport_events,
                summary.measured_event_mislabels,
                summary.estimated_event_mislabels,
                summary.estimated_presented_as_measured,
                summary.sources_separated,
                summary.total_measured_p95_ns,
                summary.total_estimated_visible_ns,
                summary.min_ratio_per_mille,
                summary.max_ratio_per_mille,
                summary.packet_loss,
                summary.checksum_failures,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("transport_metric_provenance", false, format!("{err:?}")),
    }
}
