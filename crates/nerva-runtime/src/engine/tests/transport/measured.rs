use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::measured::summary::MeasuredTransportSelectorStatus;

#[test]
fn measured_transport_selector_records_runtime_timestamp_decision() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_measured_transport_selector_probe().unwrap();

    assert_eq!(summary.status, MeasuredTransportSelectorStatus::Ok);
    assert!(summary.passed());
    assert_eq!(summary.request_bytes, 32 * 1024);
    assert!(summary.source_entries >= 3);
    assert!(summary.runtime_timestamp_events > 0);
    assert_eq!(summary.execution_decisions, 1);
    assert!(summary.candidate_count >= 3);
    assert_eq!(summary.measured_candidate_costs, summary.candidate_count);
    assert_eq!(summary.estimated_candidate_costs, 0);
    assert_eq!(summary.selected_bucket_payload_bytes, summary.request_bytes);
    assert!(summary.selected_measured_p95_ns > 0);
    assert!(summary.selected_visible_ns >= summary.selected_measured_p95_ns);
    assert!(summary.selected_bandwidth_bps > 0);
    assert_eq!(summary.decision_metric_source, "runtime_timestamp");
    assert!(summary.selected_cost_measured);
    assert!(summary.all_candidates_measured);
    assert_eq!(summary.packet_loss, 0);
    assert_eq!(summary.checksum_failures, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(
        summary
            .to_json()
            .contains("\"selected_label\":\"kernel_udp_measured_bucket_32k\"")
    );
}
