use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::transport::kernel_udp::config::KernelUdpProbeConfig;
use nerva_runtime::transport::kernel_udp::matrix::summary::KernelUdpBaselineMatrixStatus;
use nerva_runtime::transport::kernel_udp::summary::KernelUdpBaselineStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_kernel_udp_baseline(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_kernel_udp_baseline_probe(KernelUdpProbeConfig::reference_decode_activation())
    {
        Ok(summary) => report.push(
            "kernel_udp_loopback_baseline",
            matches!(summary.status, KernelUdpBaselineStatus::Ok)
                && summary.passed()
                && summary.backend == "kernel_udp_test"
                && summary.payload_bytes == 32 * 1024
                && summary.chunk_payload_bytes == 4 * 1024
                && summary.chunks == 8
                && summary.packets_sent == summary.chunks
                && summary.packets_received == summary.chunks
                && summary.validated_packets == summary.chunks
                && summary.bytes_received == summary.payload_bytes
                && summary.p50_completion_latency_ns > 0
                && summary.p95_completion_latency_ns >= summary.p50_completion_latency_ns
                && summary.p99_completion_latency_ns >= summary.p95_completion_latency_ns
                && summary.effective_payload_bandwidth_bps > 0
                && summary.runtime_timestamp_events == summary.chunks
                && summary.transport_events == summary.chunks
                && summary.packet_loss == 0
                && summary.checksum_failures == 0
                && summary.baseline_only
                && !summary.production_tensor_data_plane
                && summary.pageable_copies == 0
                && summary.per_token_registrations == 0
                && summary.hot_path_allocations == 0,
            format!(
                "backend={} version={} request={} sequence={} block={} block_version={} payload_bytes={} chunk_payload_bytes={} chunks={} wire_bytes={} sent={} received={} validated={} bytes_received={} p50_ns={} p95_ns={} p99_ns={} total_ns={} bandwidth_bps={} runtime_timestamp_events={} transport_events={} packet_loss={} checksum_failures={} baseline_only={} production_tensor_data_plane={} pageable_copies={} per_token_registrations={} hot_path_allocations={}",
                summary.backend,
                summary.protocol_version,
                summary.request_id,
                summary.sequence_id,
                summary.block_id,
                summary.block_version,
                summary.payload_bytes,
                summary.chunk_payload_bytes,
                summary.chunks,
                summary.total_wire_bytes,
                summary.packets_sent,
                summary.packets_received,
                summary.validated_packets,
                summary.bytes_received,
                summary.p50_completion_latency_ns,
                summary.p95_completion_latency_ns,
                summary.p99_completion_latency_ns,
                summary.total_completion_latency_ns,
                summary.effective_payload_bandwidth_bps,
                summary.runtime_timestamp_events,
                summary.transport_events,
                summary.packet_loss,
                summary.checksum_failures,
                summary.baseline_only,
                summary.production_tensor_data_plane,
                summary.pageable_copies,
                summary.per_token_registrations,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("kernel_udp_loopback_baseline", false, format!("{err:?}")),
    }
}

pub(crate) fn push_kernel_udp_matrix(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_kernel_udp_baseline_matrix_probe() {
        Ok(summary) => report.push(
            "kernel_udp_measured_matrix",
            matches!(summary.status, KernelUdpBaselineMatrixStatus::Ok)
                && summary.passed()
                && summary.backend == "kernel_udp_test"
                && summary.measured_sizes == 3
                && summary.entries.len() == 3
                && summary.entries[0].payload_bytes == 32 * 1024
                && summary.entries[1].payload_bytes == 256 * 1024
                && summary.entries[2].payload_bytes == 1024 * 1024
                && summary.total_payload_bytes > 0
                && summary.total_wire_bytes > summary.total_payload_bytes
                && summary.total_runtime_timestamp_events > 0
                && summary.total_transport_events == summary.total_runtime_timestamp_events
                && summary.p50_max_ns > 0
                && summary.p95_max_ns >= summary.p50_max_ns
                && summary.p99_max_ns >= summary.p95_max_ns
                && summary.min_effective_payload_bandwidth_bps > 0
                && summary.packet_loss == 0
                && summary.checksum_failures == 0
                && summary.baseline_only
                && !summary.production_tensor_data_plane
                && summary.pageable_copies == 0
                && summary.per_token_registrations == 0
                && summary.hot_path_allocations == 0,
            format!(
                "backend={} measured_sizes={} total_payload_bytes={} total_wire_bytes={} runtime_timestamp_events={} transport_events={} p50_max_ns={} p95_max_ns={} p99_max_ns={} min_bandwidth_bps={} packet_loss={} checksum_failures={} baseline_only={} production_tensor_data_plane={} pageable_copies={} per_token_registrations={} hot_path_allocations={}",
                summary.backend,
                summary.measured_sizes,
                summary.total_payload_bytes,
                summary.total_wire_bytes,
                summary.total_runtime_timestamp_events,
                summary.total_transport_events,
                summary.p50_max_ns,
                summary.p95_max_ns,
                summary.p99_max_ns,
                summary.min_effective_payload_bandwidth_bps,
                summary.packet_loss,
                summary.checksum_failures,
                summary.baseline_only,
                summary.production_tensor_data_plane,
                summary.pageable_copies,
                summary.per_token_registrations,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("kernel_udp_measured_matrix", false, format!("{err:?}")),
    }
}
