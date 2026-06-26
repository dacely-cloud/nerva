use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::provenance::summary::TransportMetricProvenanceStatus;

#[test]
fn transport_metric_provenance_keeps_measured_and_estimated_sources_separate() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_transport_metric_provenance_probe().unwrap();

    assert_eq!(summary.status, TransportMetricProvenanceStatus::Ok);
    assert!(summary.passed());
    assert!(summary.compared_sizes >= 3);
    assert_eq!(summary.runtime_timestamp_events, summary.compared_sizes);
    assert_eq!(summary.estimated_model_events, summary.compared_sizes);
    assert_eq!(summary.transport_events, summary.compared_sizes * 2);
    assert_eq!(summary.measured_event_mislabels, 0);
    assert_eq!(summary.estimated_event_mislabels, 0);
    assert!(!summary.estimated_presented_as_measured);
    assert!(summary.sources_separated);
    assert!(summary.total_measured_p95_ns > 0);
    assert!(summary.total_estimated_visible_ns > 0);
    assert_eq!(summary.packet_loss, 0);
    assert_eq!(summary.checksum_failures, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(
        summary
            .to_json()
            .contains("\"measured_source\":\"runtime_timestamp\"")
    );
    assert!(
        summary
            .to_json()
            .contains("\"estimated_source\":\"estimated_model\"")
    );
}
