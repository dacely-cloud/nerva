use nerva_core::types::cost::CostSource;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::transport::measured::decision::MeasuredTransportDecision;
use crate::transport::measured::source::MeasuredTransportSource;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MeasuredTransportSelectorStatus {
    Ok,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasuredTransportSelectorSummary {
    pub status: MeasuredTransportSelectorStatus,
    pub request_bytes: usize,
    pub source_entries: u64,
    pub runtime_timestamp_events: u64,
    pub execution_decisions: u64,
    pub candidate_count: u64,
    pub measured_candidate_costs: u64,
    pub estimated_candidate_costs: u64,
    pub selected_label: &'static str,
    pub selected_bucket_payload_bytes: usize,
    pub selected_measured_p95_ns: u64,
    pub selected_visible_ns: u64,
    pub selected_bandwidth_bps: u64,
    pub decision_metric_source: &'static str,
    pub selected_cost_measured: bool,
    pub all_candidates_measured: bool,
    pub packet_loss: u64,
    pub checksum_failures: u64,
    pub hot_path_allocations: u64,
}

impl MeasuredTransportSelectorSummary {
    pub(crate) fn from_ledger(
        source: &MeasuredTransportSource,
        decision: MeasuredTransportDecision,
        ledger: &TokenLedger,
    ) -> Self {
        let selected_cost_measured = ledger.execution_decisions.iter().any(|entry| {
            entry.operation == "measured_transport_bucket_selection"
                && entry.metric_source == MetricSource::RuntimeTimestamp
                && entry.candidate_costs.iter().any(|cost| {
                    cost.label == decision.selected.label && cost.source == CostSource::Measured
                })
        });
        Self {
            status: MeasuredTransportSelectorStatus::Ok,
            request_bytes: source.request_bytes,
            source_entries: source.source_entries,
            runtime_timestamp_events: source.runtime_timestamp_events,
            execution_decisions: ledger.execution_decisions.len() as u64,
            candidate_count: decision.candidate_count,
            measured_candidate_costs: decision.measured_candidate_costs,
            estimated_candidate_costs: decision.estimated_candidate_costs,
            selected_label: decision.selected.label,
            selected_bucket_payload_bytes: decision.selected.bucket_payload_bytes,
            selected_measured_p95_ns: decision.selected.measured_p95_ns,
            selected_visible_ns: decision.selected.visible_ns,
            selected_bandwidth_bps: decision.selected.effective_payload_bandwidth_bps,
            decision_metric_source: MetricSource::RuntimeTimestamp.as_str(),
            selected_cost_measured,
            all_candidates_measured: decision.candidate_count == decision.measured_candidate_costs
                && decision.estimated_candidate_costs == 0,
            packet_loss: source.packet_loss,
            checksum_failures: source.checksum_failures,
            hot_path_allocations: ledger.hot_path_allocations,
        }
    }

    pub fn passed(&self) -> bool {
        matches!(self.status, MeasuredTransportSelectorStatus::Ok)
            && self.request_bytes == 32 * 1024
            && self.source_entries >= 3
            && self.runtime_timestamp_events > 0
            && self.execution_decisions == 1
            && self.candidate_count >= 3
            && self.measured_candidate_costs == self.candidate_count
            && self.estimated_candidate_costs == 0
            && self.selected_bucket_payload_bytes == self.request_bytes
            && self.selected_measured_p95_ns > 0
            && self.selected_visible_ns >= self.selected_measured_p95_ns
            && self.selected_bandwidth_bps > 0
            && self.decision_metric_source == "runtime_timestamp"
            && self.selected_cost_measured
            && self.all_candidates_measured
            && self.packet_loss == 0
            && self.checksum_failures == 0
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            MeasuredTransportSelectorStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"request_bytes\":{},\"source_entries\":{},\"runtime_timestamp_events\":{},\"execution_decisions\":{},\"candidate_count\":{},\"measured_candidate_costs\":{},\"estimated_candidate_costs\":{},\"selected_label\":\"{}\",\"selected_bucket_payload_bytes\":{},\"selected_measured_p95_ns\":{},\"selected_visible_ns\":{},\"selected_bandwidth_bps\":{},\"decision_metric_source\":\"{}\",\"selected_cost_measured\":{},\"all_candidates_measured\":{},\"packet_loss\":{},\"checksum_failures\":{},\"hot_path_allocations\":{}}}",
            status,
            self.request_bytes,
            self.source_entries,
            self.runtime_timestamp_events,
            self.execution_decisions,
            self.candidate_count,
            self.measured_candidate_costs,
            self.estimated_candidate_costs,
            self.selected_label,
            self.selected_bucket_payload_bytes,
            self.selected_measured_p95_ns,
            self.selected_visible_ns,
            self.selected_bandwidth_bps,
            self.decision_metric_source,
            self.selected_cost_measured,
            self.all_candidates_measured,
            self.packet_loss,
            self.checksum_failures,
            self.hot_path_allocations,
        )
    }
}
