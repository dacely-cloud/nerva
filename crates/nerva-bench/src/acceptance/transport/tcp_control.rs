use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::transport::tcp_control::config::TcpControlProbeConfig;
use nerva_runtime::transport::tcp_control::summary::TcpControlStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_tcp_control(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_tcp_control_probe(TcpControlProbeConfig::reference_handshake()) {
        Ok(summary) => report.push(
            "tcp_control_debug_only",
            matches!(summary.status, TcpControlStatus::Ok)
                && summary.passed()
                && summary.backend == "tcp_control_only"
                && summary.tensor_payload_bytes == 0
                && summary.control_plane_only
                && summary.debug_only
                && !summary.production_tensor_data_plane
                && summary.runtime_timestamp_events == 1
                && summary.transport_events == 1
                && summary.pageable_copies == 0
                && summary.per_token_registrations == 0
                && summary.hot_path_allocations == 0,
            format!(
                "backend={} control_bytes_sent={} control_bytes_received={} tensor_payload_bytes={} total_wire_bytes={} connection_count={} control_messages={} completion_latency_ns={} control_plane_only={} debug_only={} production_tensor_data_plane={} runtime_timestamp_events={} transport_events={} pageable_copies={} per_token_registrations={} hot_path_allocations={}",
                summary.backend,
                summary.control_bytes_sent,
                summary.control_bytes_received,
                summary.tensor_payload_bytes,
                summary.total_wire_bytes,
                summary.connection_count,
                summary.control_messages,
                summary.completion_latency_ns,
                summary.control_plane_only,
                summary.debug_only,
                summary.production_tensor_data_plane,
                summary.runtime_timestamp_events,
                summary.transport_events,
                summary.pageable_copies,
                summary.per_token_registrations,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("tcp_control_debug_only", false, format!("{err:?}")),
    }
}
