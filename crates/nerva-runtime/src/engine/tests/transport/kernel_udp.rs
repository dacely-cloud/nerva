use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::kernel_udp::config::KernelUdpProbeConfig;
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
