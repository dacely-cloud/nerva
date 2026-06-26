use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::transport::kernel_udp::matrix::summary::KernelUdpBaselineMatrixSummary;
use crate::transport::matrix::types::{
    TransportCapabilityMatrixSummary, TransportMatrixRequestedPath,
};
use crate::transport::path::types::TransferMode;
use crate::transport::provenance::entry::TransportMetricProvenanceEntry;
use crate::transport::provenance::ledger::{ESTIMATED_LABEL, MEASURED_LABEL};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportMetricProvenanceStatus {
    Ok,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportMetricProvenanceSummary {
    pub status: TransportMetricProvenanceStatus,
    pub compared_sizes: u64,
    pub entries: Vec<TransportMetricProvenanceEntry>,
    pub measured_matrix_entries: u64,
    pub estimated_matrix_entries: u64,
    pub runtime_timestamp_events: u64,
    pub estimated_model_events: u64,
    pub transport_events: u64,
    pub measured_event_mislabels: u64,
    pub estimated_event_mislabels: u64,
    pub estimated_presented_as_measured: bool,
    pub sources_separated: bool,
    pub total_measured_p95_ns: u64,
    pub total_estimated_visible_ns: u64,
    pub min_ratio_per_mille: u64,
    pub max_ratio_per_mille: u64,
    pub packet_loss: u64,
    pub checksum_failures: u64,
    pub hot_path_allocations: u64,
}

impl TransportMetricProvenanceSummary {
    pub(crate) fn from_parts(
        measured: &KernelUdpBaselineMatrixSummary,
        estimated: &TransportCapabilityMatrixSummary,
        entries: Vec<TransportMetricProvenanceEntry>,
        ledger: &TokenLedger,
    ) -> Self {
        let measured_event_mislabels = ledger
            .events
            .iter()
            .filter(|event| {
                event.label == MEASURED_LABEL
                    && event.metric_source != MetricSource::RuntimeTimestamp
            })
            .count() as u64;
        let estimated_event_mislabels = ledger
            .events
            .iter()
            .filter(|event| {
                event.label == ESTIMATED_LABEL
                    && event.metric_source != MetricSource::EstimatedModel
            })
            .count() as u64;
        let runtime_timestamp_events =
            ledger.event_count_for_source(MetricSource::RuntimeTimestamp);
        let estimated_model_events = ledger.event_count_for_source(MetricSource::EstimatedModel);
        let compared_sizes = entries.len() as u64;
        let sources_separated = runtime_timestamp_events == compared_sizes
            && estimated_model_events == compared_sizes
            && measured_event_mislabels == 0
            && estimated_event_mislabels == 0;

        Self {
            status: TransportMetricProvenanceStatus::Ok,
            compared_sizes,
            measured_matrix_entries: measured.entries.len() as u64,
            estimated_matrix_entries: estimated.entries.len() as u64,
            runtime_timestamp_events,
            estimated_model_events,
            transport_events: ledger.event_count(LedgerEventKind::Transport),
            measured_event_mislabels,
            estimated_event_mislabels,
            estimated_presented_as_measured: estimated_event_mislabels > 0,
            sources_separated,
            total_measured_p95_ns: entries.iter().map(|entry| entry.measured_p95_ns).sum(),
            total_estimated_visible_ns: entries
                .iter()
                .map(|entry| entry.estimated_visible_ns)
                .sum(),
            min_ratio_per_mille: entries
                .iter()
                .map(|entry| entry.ratio_per_mille)
                .min()
                .unwrap_or(0),
            max_ratio_per_mille: entries
                .iter()
                .map(|entry| entry.ratio_per_mille)
                .max()
                .unwrap_or(0),
            packet_loss: measured.packet_loss,
            checksum_failures: measured.checksum_failures,
            hot_path_allocations: ledger.hot_path_allocations,
            entries,
        }
    }

    pub fn passed(&self) -> bool {
        matches!(self.status, TransportMetricProvenanceStatus::Ok)
            && self.compared_sizes >= 3
            && self.entries.len() as u64 == self.compared_sizes
            && self.measured_matrix_entries >= self.compared_sizes
            && self.estimated_matrix_entries >= self.compared_sizes
            && self.runtime_timestamp_events == self.compared_sizes
            && self.estimated_model_events == self.compared_sizes
            && self.transport_events == self.compared_sizes.saturating_mul(2)
            && self.measured_event_mislabels == 0
            && self.estimated_event_mislabels == 0
            && !self.estimated_presented_as_measured
            && self.sources_separated
            && self.total_measured_p95_ns > 0
            && self.total_estimated_visible_ns > 0
            && self.max_ratio_per_mille >= self.min_ratio_per_mille
            && self.packet_loss == 0
            && self.checksum_failures == 0
            && self.hot_path_allocations == 0
            && self.entries.iter().all(entry_has_separate_sources)
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            TransportMetricProvenanceStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"compared_sizes\":{},\"measured_matrix_entries\":{},\"estimated_matrix_entries\":{},\"runtime_timestamp_events\":{},\"estimated_model_events\":{},\"transport_events\":{},\"measured_event_mislabels\":{},\"estimated_event_mislabels\":{},\"estimated_presented_as_measured\":{},\"sources_separated\":{},\"total_measured_p95_ns\":{},\"total_estimated_visible_ns\":{},\"min_ratio_per_mille\":{},\"max_ratio_per_mille\":{},\"packet_loss\":{},\"checksum_failures\":{},\"hot_path_allocations\":{},\"entries\":{}}}",
            status,
            self.compared_sizes,
            self.measured_matrix_entries,
            self.estimated_matrix_entries,
            self.runtime_timestamp_events,
            self.estimated_model_events,
            self.transport_events,
            self.measured_event_mislabels,
            self.estimated_event_mislabels,
            self.estimated_presented_as_measured,
            self.sources_separated,
            self.total_measured_p95_ns,
            self.total_estimated_visible_ns,
            self.min_ratio_per_mille,
            self.max_ratio_per_mille,
            self.packet_loss,
            self.checksum_failures,
            self.hot_path_allocations,
            entries_json(&self.entries),
        )
    }
}

fn entry_has_separate_sources(entry: &TransportMetricProvenanceEntry) -> bool {
    entry.payload_bytes > 0
        && entry.measured_p95_ns > 0
        && entry.estimated_visible_ns > 0
        && entry.measured_source == MetricSource::RuntimeTimestamp
        && entry.estimated_source == MetricSource::EstimatedModel
        && entry.requested_path == TransportMatrixRequestedPath::PinnedHostBounce
        && entry.mode == TransferMode::Decode
}

fn entries_json(entries: &[TransportMetricProvenanceEntry]) -> String {
    let items = entries
        .iter()
        .map(TransportMetricProvenanceEntry::to_json)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}
