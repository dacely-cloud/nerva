use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::kernel_udp::config::KernelUdpProbeConfig;
use crate::transport::kernel_udp::matrix::summary::KernelUdpBaselineMatrixStatus;
use crate::transport::kernel_udp::summary::KernelUdpBaselineStatus;

#[test]
fn kernel_udp_probe_transfers_validated_loopback_chunks() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_kernel_udp_baseline_probe(KernelUdpProbeConfig::reference_decode_activation())
        .unwrap();

    assert_eq!(summary.status, KernelUdpBaselineStatus::Ok);
    assert!(summary.passed());
    assert_eq!(summary.backend, "kernel_udp_test");
    assert_eq!(summary.payload_bytes, 32 * 1024);
    assert_eq!(summary.chunk_payload_bytes, 4 * 1024);
    assert_eq!(summary.chunks, 8);
    assert_eq!(summary.packets_sent, summary.chunks);
    assert_eq!(summary.packets_received, summary.chunks);
    assert_eq!(summary.validated_packets, summary.chunks);
    assert_eq!(summary.bytes_received, summary.payload_bytes);
    assert_eq!(summary.packet_loss, 0);
    assert_eq!(summary.checksum_failures, 0);
    assert!(summary.p50_completion_latency_ns > 0);
    assert!(summary.p95_completion_latency_ns >= summary.p50_completion_latency_ns);
    assert!(summary.p99_completion_latency_ns >= summary.p95_completion_latency_ns);
    assert!(summary.effective_payload_bandwidth_bps > 0);
    assert_eq!(summary.runtime_timestamp_events, summary.chunks);
    assert_eq!(summary.transport_events, summary.chunks);
    assert!(summary.baseline_only);
    assert!(!summary.production_tensor_data_plane);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(
        summary
            .to_json()
            .contains("\"backend\":\"kernel_udp_test\"")
    );
}

#[test]
fn kernel_udp_probe_rejects_oversized_datagrams() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut config = KernelUdpProbeConfig::reference_decode_activation();
    config.chunk_payload_bytes = 64 * 1024;

    assert!(runtime.run_kernel_udp_baseline_probe(config).is_err());
}

#[test]
fn kernel_udp_matrix_probe_measures_decode_sizes() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_kernel_udp_baseline_matrix_probe().unwrap();

    assert_eq!(summary.status, KernelUdpBaselineMatrixStatus::Ok);
    assert!(summary.passed());
    assert_eq!(summary.backend, "kernel_udp_test");
    assert_eq!(summary.measured_sizes, 3);
    assert_eq!(summary.entries.len(), 3);
    assert_eq!(summary.entries[0].payload_bytes, 32 * 1024);
    assert_eq!(summary.entries[1].payload_bytes, 256 * 1024);
    assert_eq!(summary.entries[2].payload_bytes, 1024 * 1024);
    assert!(summary.total_payload_bytes > 0);
    assert!(summary.total_wire_bytes > summary.total_payload_bytes);
    assert!(summary.total_runtime_timestamp_events > 0);
    assert_eq!(
        summary.total_transport_events,
        summary.total_runtime_timestamp_events
    );
    assert!(summary.p50_max_ns > 0);
    assert!(summary.p95_max_ns >= summary.p50_max_ns);
    assert!(summary.p99_max_ns >= summary.p95_max_ns);
    assert!(summary.min_effective_payload_bandwidth_bps > 0);
    assert_eq!(summary.packet_loss, 0);
    assert_eq!(summary.checksum_failures, 0);
    assert!(summary.baseline_only);
    assert!(!summary.production_tensor_data_plane);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"measured_sizes\":3"));
}
