use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::tcp_control::config::TcpControlProbeConfig;
use crate::transport::tcp_control::summary::TcpControlStatus;

#[test]
fn tcp_control_probe_reports_control_only_loopback() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_tcp_control_probe(TcpControlProbeConfig::reference_handshake())
        .unwrap();

    assert_eq!(summary.status, TcpControlStatus::Ok);
    assert_eq!(summary.backend, "tcp_control_only");
    assert!(summary.passed());
    assert_eq!(summary.control_bytes_sent, 64);
    assert!(summary.control_bytes_received > 0);
    assert_eq!(summary.tensor_payload_bytes, 0);
    assert!(summary.control_plane_only);
    assert!(summary.debug_only);
    assert!(!summary.production_tensor_data_plane);
    assert_eq!(summary.runtime_timestamp_events, 1);
    assert_eq!(summary.transport_events, 1);
    assert_eq!(summary.pageable_copies, 0);
    assert_eq!(summary.per_token_registrations, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(
        summary
            .to_json()
            .contains("\"backend\":\"tcp_control_only\"")
    );
}

#[test]
fn tcp_control_probe_rejects_tensor_sized_control_payloads() {
    let mut config = TcpControlProbeConfig::reference_handshake();
    config.control_bytes = config.max_control_bytes + 1;

    assert!(crate::transport::tcp_control::run::run_tcp_control_probe(config).is_err());
}
