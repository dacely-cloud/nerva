use nerva_core::types::error::{NervaError, Result};

use crate::transport::kernel_udp::matrix::summary::KernelUdpBaselineMatrixSummary;
use crate::transport::measured::candidate::MeasuredTransportCandidate;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MeasuredTransportSource {
    pub request_bytes: usize,
    pub source_entries: u64,
    pub runtime_timestamp_events: u64,
    pub packet_loss: u64,
    pub checksum_failures: u64,
    pub candidates: Vec<MeasuredTransportCandidate>,
}

impl MeasuredTransportSource {
    pub(crate) fn from_kernel_udp_matrix(
        request_bytes: usize,
        matrix: &KernelUdpBaselineMatrixSummary,
    ) -> Result<Self> {
        if request_bytes == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "measured transport request bytes must be nonzero".to_string(),
            });
        }
        let mut candidates = Vec::new();
        for entry in &matrix.entries {
            if entry.payload_bytes >= request_bytes {
                candidates.push(MeasuredTransportCandidate {
                    label: bucket_label(entry.payload_bytes)?,
                    bucket_payload_bytes: entry.payload_bytes,
                    measured_p95_ns: entry.p95_completion_latency_ns,
                    effective_payload_bandwidth_bps: entry.effective_payload_bandwidth_bps,
                    visible_ns: measured_visible_ns(
                        request_bytes,
                        entry.payload_bytes,
                        entry.p95_completion_latency_ns,
                        entry.effective_payload_bandwidth_bps,
                    ),
                });
            }
        }
        if candidates.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "measured transport matrix has no bucket for requested payload".to_string(),
            });
        }
        Ok(Self {
            request_bytes,
            source_entries: matrix.entries.len() as u64,
            runtime_timestamp_events: matrix.total_runtime_timestamp_events,
            packet_loss: matrix.packet_loss,
            checksum_failures: matrix.checksum_failures,
            candidates,
        })
    }
}

fn measured_visible_ns(
    request_bytes: usize,
    bucket_payload_bytes: usize,
    measured_p95_ns: u64,
    effective_payload_bandwidth_bps: u64,
) -> u64 {
    let excess_bytes = bucket_payload_bytes.saturating_sub(request_bytes);
    measured_p95_ns.saturating_add(bytes_to_ns(
        excess_bytes,
        effective_payload_bandwidth_bps.max(1),
    ))
}

fn bytes_to_ns(bytes: usize, bandwidth_bps: u64) -> u64 {
    ((bytes as u128)
        .saturating_mul(1_000_000_000)
        .saturating_div(bandwidth_bps as u128))
    .min(u64::MAX as u128) as u64
}

fn bucket_label(payload_bytes: usize) -> Result<&'static str> {
    match payload_bytes {
        32768 => Ok("kernel_udp_measured_bucket_32k"),
        262144 => Ok("kernel_udp_measured_bucket_256k"),
        1048576 => Ok("kernel_udp_measured_bucket_1m"),
        _ => Err(NervaError::InvalidArgument {
            reason: format!("unsupported measured transport bucket size {payload_bytes}"),
        }),
    }
}
