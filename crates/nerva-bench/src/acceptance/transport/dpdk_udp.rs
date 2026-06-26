use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::transport::dpdk_udp::config::DpdkUdpProbeConfig;
use nerva_runtime::transport::dpdk_udp::protocol::DpdkUdpMemoryPath;
use nerva_runtime::transport::dpdk_udp::summary::DpdkUdpProtocolStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_dpdk_udp_protocol(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_dpdk_udp_protocol_probe(DpdkUdpProbeConfig::reference_decode_activation()) {
        Ok(summary) => report.push(
            "dpdk_udp_activation_protocol",
            matches!(summary.status, DpdkUdpProtocolStatus::Ok)
                && summary.selected_path == DpdkUdpMemoryPath::PinnedHostBuffer
                && summary.passed()
                && summary.payload_bytes == 32 * 1024
                && summary.protocol_version == 1
                && summary.request_id > 0
                && summary.sequence_id > 0
                && summary.block_id > 0
                && summary.block_version > 0
                && summary.chunks == 8
                && summary.nack_ranges == 1
                && summary.selective_retransmits == 1
                && summary.ack_packets == 0
                && !summary.direct_gpu_memory_claimed
                && summary.pinned_host_required
                && summary.pageable_copies == 0
                && summary.per_token_registrations == 0
                && summary.hot_path_allocations == 0,
            format!(
                "version={} request={} sequence={} block={} block_version={} path={} capability={:?} payload_bytes={} chunks={} wire_bytes={} preposted_receives={} credit_window={} credit_stalls={} sender_retention={} bitmap_words={} nack_ranges={} retransmits={} ack_packets={} mbufs={} rings={} direct_gpu_memory_claimed={} pinned_host_required={} fallback_decisions={} transport_events={} phase_handoff_syncs={} pageable_copies={} per_token_registrations={} hot_path_allocations={}",
                summary.protocol_version,
                summary.request_id,
                summary.sequence_id,
                summary.block_id,
                summary.block_version,
                summary.selected_path.as_str(),
                summary.capability_result,
                summary.payload_bytes,
                summary.chunks,
                summary.total_wire_bytes,
                summary.preposted_receives,
                summary.credit_window_chunks,
                summary.credit_stalls,
                summary.sender_retention_chunks,
                summary.receiver_bitmap_words,
                summary.nack_ranges,
                summary.selective_retransmits,
                summary.ack_packets,
                summary.mbufs_preallocated,
                summary.rings_preallocated,
                summary.direct_gpu_memory_claimed,
                summary.pinned_host_required,
                summary.fallback_decisions,
                summary.transport_events,
                summary.phase_handoff_syncs,
                summary.pageable_copies,
                summary.per_token_registrations,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("dpdk_udp_activation_protocol", false, format!("{err:?}")),
    }
}

pub(crate) fn push_dpdk_udp_credit_pressure(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime
        .run_dpdk_udp_protocol_probe(DpdkUdpProbeConfig::credit_pressure_decode_activation())
    {
        Ok(summary) => report.push(
            "dpdk_udp_credit_pressure",
            matches!(summary.status, DpdkUdpProtocolStatus::Ok)
                && summary.selected_path == DpdkUdpMemoryPath::PinnedHostBuffer
                && summary.passed()
                && summary.chunks == 8
                && summary.credit_window_chunks == 3
                && summary.credit_windows == 3
                && summary.credit_stalls == 2
                && summary.nack_ranges == 0
                && summary.selective_retransmits == 0
                && summary.preposted_receives == summary.chunks
                && summary.pageable_copies == 0
                && summary.per_token_registrations == 0
                && summary.hot_path_allocations == 0,
            format!(
                "chunks={} credit_window={} credit_windows={} credit_stalls={} preposted_receives={} nack_ranges={} retransmits={} fallback_decisions={} pageable_copies={} per_token_registrations={} hot_path_allocations={}",
                summary.chunks,
                summary.credit_window_chunks,
                summary.credit_windows,
                summary.credit_stalls,
                summary.preposted_receives,
                summary.nack_ranges,
                summary.selective_retransmits,
                summary.fallback_decisions,
                summary.pageable_copies,
                summary.per_token_registrations,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("dpdk_udp_credit_pressure", false, format!("{err:?}")),
    }
}
