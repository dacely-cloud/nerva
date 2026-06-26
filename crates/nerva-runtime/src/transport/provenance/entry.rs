use crate::transport::kernel_udp::matrix::summary::KernelUdpBaselineMatrixSummary;
use crate::transport::matrix::types::{
    TransportCapabilityMatrixEntry, TransportCapabilityMatrixSummary, TransportMatrixRequestedPath,
};
use crate::transport::path::types::TransferMode;
use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::metric::MetricSource;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportMetricProvenanceEntry {
    pub payload_bytes: usize,
    pub measured_p95_ns: u64,
    pub estimated_visible_ns: u64,
    pub ratio_per_mille: u64,
    pub measured_source: MetricSource,
    pub estimated_source: MetricSource,
    pub requested_path: TransportMatrixRequestedPath,
    pub mode: TransferMode,
}

impl TransportMetricProvenanceEntry {
    pub(crate) fn from_pair(
        measured_payload_bytes: usize,
        measured_p95_ns: u64,
        estimated: &TransportCapabilityMatrixEntry,
    ) -> Self {
        Self {
            payload_bytes: measured_payload_bytes,
            measured_p95_ns,
            estimated_visible_ns: estimated.estimated_visible_ns,
            ratio_per_mille: ratio_per_mille(measured_p95_ns, estimated.estimated_visible_ns),
            measured_source: MetricSource::RuntimeTimestamp,
            estimated_source: MetricSource::EstimatedModel,
            requested_path: estimated.requested_path,
            mode: estimated.mode,
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"payload_bytes\":{},\"measured_p95_ns\":{},\"estimated_visible_ns\":{},\"ratio_per_mille\":{},\"measured_source\":\"{}\",\"estimated_source\":\"{}\",\"requested_path\":\"{}\",\"mode\":\"{}\"}}",
            self.payload_bytes,
            self.measured_p95_ns,
            self.estimated_visible_ns,
            self.ratio_per_mille,
            self.measured_source.as_str(),
            self.estimated_source.as_str(),
            requested_path_label(self.requested_path),
            transfer_mode_label(self.mode),
        )
    }
}

pub(crate) fn build_transport_metric_provenance_entries(
    measured: &KernelUdpBaselineMatrixSummary,
    estimated: &TransportCapabilityMatrixSummary,
) -> Result<Vec<TransportMetricProvenanceEntry>> {
    let mut entries = Vec::with_capacity(measured.entries.len());
    for measured_entry in &measured.entries {
        let estimated_entry = find_estimated_decode_pinned_host_entry(
            measured_entry.payload_bytes,
            &estimated.entries,
        )?;
        entries.push(TransportMetricProvenanceEntry::from_pair(
            measured_entry.payload_bytes,
            measured_entry.p95_completion_latency_ns,
            estimated_entry,
        ));
    }
    Ok(entries)
}

fn find_estimated_decode_pinned_host_entry(
    payload_bytes: usize,
    entries: &[TransportCapabilityMatrixEntry],
) -> Result<&TransportCapabilityMatrixEntry> {
    entries
        .iter()
        .find(|entry| {
            entry.size_bytes == payload_bytes
                && entry.mode == TransferMode::Decode
                && entry.requested_path == TransportMatrixRequestedPath::PinnedHostBounce
        })
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!(
                "missing estimated pinned-host decode transport entry for {} bytes",
                payload_bytes
            ),
        })
}

fn ratio_per_mille(measured_ns: u64, estimated_ns: u64) -> u64 {
    if estimated_ns == 0 {
        0
    } else {
        measured_ns.saturating_mul(1000) / estimated_ns
    }
}

fn requested_path_label(path: TransportMatrixRequestedPath) -> &'static str {
    match path {
        TransportMatrixRequestedPath::GpuDirectRdma => "gpu_direct_rdma",
        TransportMatrixRequestedPath::PinnedHostBounce => "pinned_host_bounce",
        TransportMatrixRequestedPath::CpuProducedBoundary => "cpu_produced_boundary",
        TransportMatrixRequestedPath::MappedPinnedWrite => "mapped_pinned_write",
    }
}

fn transfer_mode_label(mode: TransferMode) -> &'static str {
    match mode {
        TransferMode::Decode => "decode",
        TransferMode::Prefill => "prefill",
    }
}
