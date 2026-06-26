use crate::capabilities::snapshot::CapabilityState;
use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::dpdk_udp::config::DpdkUdpProbeConfig;
use crate::transport::dpdk_udp::protocol::{DpdkUdpMemoryPath, plan_dpdk_udp_protocol};
use crate::transport::dpdk_udp::summary::DpdkUdpProtocolStatus;

#[test]
fn dpdk_udp_protocol_plans_bounded_decode_activation_chunks() {
    let config = DpdkUdpProbeConfig::reference_decode_activation();
    let plan = plan_dpdk_udp_protocol(
        config,
        CapabilityState::DegradedToPinnedHost,
        CapabilityState::SupportedUnverified,
    )
    .unwrap();

    assert_eq!(plan.selected_path, DpdkUdpMemoryPath::PinnedHostBuffer);
    assert_eq!(
        plan.capability_result,
        CapabilityState::DegradedToPinnedHost
    );
    assert!(!plan.direct_gpu_memory_claimed);
    assert!(plan.pinned_host_required);
    assert_eq!(plan.chunk_count, 8);
    assert_eq!(plan.preposted_receives, 8);
    assert_eq!(plan.nack_ranges, 1);
    assert_eq!(plan.selective_retransmits, 1);
    assert_eq!(plan.ack_packets, 0);
    assert_eq!(plan.credit_stalls, 0);
    assert_eq!(plan.credit_stall_ns, 0);
    assert!(plan.total_wire_bytes > config.payload_bytes);
    assert!(plan.chunks.iter().all(|chunk| chunk.retained_by_sender));
    assert_eq!(
        plan.chunks.iter().filter(|chunk| chunk.needs_nack).count(),
        1
    );
}

#[test]
fn dpdk_udp_protocol_rejects_insufficient_sender_retention() {
    let mut config = DpdkUdpProbeConfig::reference_decode_activation();
    config.sender_retention_chunks = 4;

    assert!(
        plan_dpdk_udp_protocol(
            config,
            CapabilityState::DegradedToPinnedHost,
            CapabilityState::SupportedUnverified,
        )
        .is_err()
    );
}

#[test]
fn dpdk_udp_protocol_reports_credit_pressure_windows() {
    let config = DpdkUdpProbeConfig::credit_pressure_decode_activation();
    let plan = plan_dpdk_udp_protocol(
        config,
        CapabilityState::DegradedToPinnedHost,
        CapabilityState::SupportedUnverified,
    )
    .unwrap();

    assert_eq!(plan.chunk_count, 8);
    assert_eq!(plan.credit_windows, 3);
    assert_eq!(plan.credit_stalls, 2);
    assert_eq!(plan.credit_stall_ns, 1_500);
    assert_eq!(plan.preposted_receives, 8);
    assert_eq!(plan.nack_ranges, 0);
    assert_eq!(plan.selective_retransmits, 0);
}

#[test]
fn dpdk_udp_probe_reports_pinned_host_fallback_without_hot_allocations() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_dpdk_udp_protocol_probe(DpdkUdpProbeConfig::reference_decode_activation())
        .unwrap();

    assert_eq!(summary.status, DpdkUdpProtocolStatus::Ok);
    assert_eq!(summary.selected_path, DpdkUdpMemoryPath::PinnedHostBuffer);
    assert!(summary.passed());
    assert_eq!(summary.payload_bytes, 32 * 1024);
    assert_eq!(summary.chunks, 8);
    assert_eq!(summary.preposted_receives, summary.chunks);
    assert_eq!(summary.nack_ranges, 1);
    assert_eq!(summary.selective_retransmits, 1);
    assert_eq!(summary.ack_packets, 0);
    assert_eq!(summary.credit_stall_ns, 0);
    assert_eq!(summary.pageable_copies, 0);
    assert_eq!(summary.per_token_registrations, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(
        summary
            .to_json()
            .contains("\"selected_path\":\"dpdk_udp_pinned_host\"")
    );
}
