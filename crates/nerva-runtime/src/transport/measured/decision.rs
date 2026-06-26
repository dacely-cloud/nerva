use nerva_core::types::cost::CostSource;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::TransportDeviceId;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::decision::ExecutionDecision;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::transport::measured::candidate::MeasuredTransportCandidate;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct MeasuredTransportDecision {
    pub selected: MeasuredTransportCandidate,
    pub candidate_count: u64,
    pub measured_candidate_costs: u64,
    pub estimated_candidate_costs: u64,
}

pub(crate) fn record_measured_transport_decision(
    ledger: &mut TokenLedger,
    candidates: &[MeasuredTransportCandidate],
) -> Result<MeasuredTransportDecision> {
    let selected = candidates
        .iter()
        .copied()
        .min_by_key(|candidate| candidate.visible_ns)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "measured transport selector requires at least one candidate".to_string(),
        })?;
    let costs = candidates
        .iter()
        .copied()
        .map(MeasuredTransportCandidate::as_cost)
        .collect::<Vec<_>>();
    let measured_candidate_costs = costs
        .iter()
        .filter(|cost| cost.source == CostSource::Measured)
        .count() as u64;
    let estimated_candidate_costs = costs
        .iter()
        .filter(|cost| cost.source == CostSource::Estimated)
        .count() as u64;

    ledger.record_execution_decision(ExecutionDecision {
        operation: "measured_transport_bucket_selection",
        executor_selected: ExecutionOwner::Nic(TransportDeviceId(0)),
        candidate_costs: costs,
        reason: "lowest measured p95 visible cost for the requested transport payload",
        predicted_visible_ns: selected.visible_ns,
        actual_visible_ns: Some(selected.visible_ns),
        metric_source: MetricSource::RuntimeTimestamp,
    });

    Ok(MeasuredTransportDecision {
        selected,
        candidate_count: candidates.len() as u64,
        measured_candidate_costs,
        estimated_candidate_costs,
    })
}
