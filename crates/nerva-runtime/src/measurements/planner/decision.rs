use nerva_core::types::cost::source::CostSource;
use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::decision::ExecutionDecision;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::measurements::planner::candidate::MeasuredPlannerCandidate;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct MeasuredPlannerDecision {
    pub selected: MeasuredPlannerCandidate,
    pub candidate_count: u64,
    pub measured_candidate_costs: u64,
    pub estimated_candidate_costs: u64,
}

pub(crate) fn record_measured_planner_decision(
    ledger: &mut TokenLedger,
    candidates: &[MeasuredPlannerCandidate],
) -> Result<MeasuredPlannerDecision> {
    let selected = select_lowest_visible_cost(candidates)?;
    let costs = candidates
        .iter()
        .copied()
        .map(MeasuredPlannerCandidate::as_cost)
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
        operation: "measured_exact_matvec_placement",
        executor_selected: selected.executor,
        candidate_costs: costs,
        reason: "lowest measured visible critical path cost",
        predicted_visible_ns: selected.visible_ns,
        actual_visible_ns: Some(selected.visible_ns),
        metric_source: MetricSource::RuntimeTimestamp,
    });

    Ok(MeasuredPlannerDecision {
        selected,
        candidate_count: candidates.len() as u64,
        measured_candidate_costs,
        estimated_candidate_costs,
    })
}

fn select_lowest_visible_cost(
    candidates: &[MeasuredPlannerCandidate],
) -> Result<MeasuredPlannerCandidate> {
    candidates
        .iter()
        .copied()
        .min_by_key(|candidate| candidate.visible_ns)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "measured planner requires at least one candidate".to_string(),
        })
}
