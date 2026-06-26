use nerva_core::types::cost::source::CostSource;
use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::warm_compute::probe::candidate::run_warm_compute_candidate;
use crate::warm_compute::probe::fixture::WarmComputeFixture;
use crate::warm_compute::probe::selection::candidate_visible_ns;
use crate::warm_compute::strategy::WarmComputeStrategy;
use crate::warm_compute::summary::{WarmComputeProbeStatus, WarmComputeProbeSummary};

pub fn warm_compute_probe() -> Result<WarmComputeProbeSummary> {
    let fixture = WarmComputeFixture::default();
    let mut ledger = TokenLedger::new(0);
    let mut candidates = Vec::new();

    for strategy in [
        WarmComputeStrategy::CpuDram,
        WarmComputeStrategy::GpuResident,
        WarmComputeStrategy::GpuStaged,
        WarmComputeStrategy::HybridSplit,
    ] {
        candidates.push(run_warm_compute_candidate(
            strategy,
            fixture.rows,
            fixture.cols,
            &fixture.matrix,
            &fixture.input,
            &mut ledger,
        )?);
    }

    let output_hash = candidates
        .first()
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "warm compute probe produced no candidates".to_string(),
        })?
        .output_hash;
    let parity = candidates
        .iter()
        .all(|candidate| candidate.output_hash == output_hash);
    if !parity {
        return Err(NervaError::InvalidArgument {
            reason: "warm compute candidate parity failed".to_string(),
        });
    }

    let selected = candidates
        .iter()
        .min_by_key(|candidate| candidate.visible_ns)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "warm compute candidate selection failed".to_string(),
        })?;
    let selected_strategy = selected.strategy;
    let selected_visible_ns = selected.visible_ns;
    let cpu_visible = candidate_visible_ns(&candidates, WarmComputeStrategy::CpuDram)?;
    let staged_visible = candidate_visible_ns(&candidates, WarmComputeStrategy::GpuStaged)?;

    ledger.record_execution_decision(ExecutionDecision {
        operation: "dense_matvec",
        executor_selected: selected_strategy.executor(),
        candidate_costs: candidate_costs(&candidates),
        reason: "select exact candidate with lowest modeled visible cost and preserve measured probe cost",
        predicted_visible_ns: selected_visible_ns,
        actual_visible_ns: Some(selected.measured_ns),
        metric_source: MetricSource::RuntimeTimestamp,
    });
    ledger.require_zero_hot_path_allocations()?;

    let copy_bytes = ledger
        .events
        .iter()
        .filter(|event| event.kind == LedgerEventKind::Copy)
        .map(|event| event.bytes)
        .sum();

    Ok(WarmComputeProbeSummary {
        status: WarmComputeProbeStatus::Ok,
        rows: fixture.rows,
        cols: fixture.cols,
        candidates,
        selected_strategy,
        parity,
        cpu_beats_staged: cpu_visible < staged_visible,
        execution_decisions: ledger.execution_decisions.len() as u64,
        runtime_timestamp_decisions: decisions_with_source(&ledger, MetricSource::RuntimeTimestamp),
        measured_candidate_costs: candidate_costs_with_source(&ledger, CostSource::Measured),
        estimated_candidate_costs: candidate_costs_with_source(&ledger, CostSource::Estimated),
        cpu_events: ledger.event_count(LedgerEventKind::CpuActivity),
        device_events: ledger.event_count(LedgerEventKind::DeviceActivity),
        copy_events: ledger.event_count(LedgerEventKind::Copy),
        copy_bytes,
        total_latency_ns: ledger.total_latency_ns(),
        hot_path_allocations: ledger.hot_path_allocations,
        output_hash,
    })
}

fn candidate_costs(
    candidates: &[crate::warm_compute::summary::WarmComputeCandidate],
) -> Vec<CandidateCost> {
    let mut costs = Vec::with_capacity(candidates.len() * 2);
    for candidate in candidates {
        costs.push(CandidateCost::estimated(
            candidate.strategy.label(),
            candidate.visible_ns,
        ));
        costs.push(CandidateCost::measured(
            candidate.strategy.label(),
            candidate.measured_ns,
        ));
    }
    costs
}

fn decisions_with_source(ledger: &TokenLedger, source: MetricSource) -> u64 {
    ledger
        .execution_decisions
        .iter()
        .filter(|decision| decision.metric_source == source)
        .count() as u64
}

fn candidate_costs_with_source(ledger: &TokenLedger, source: CostSource) -> u64 {
    ledger
        .execution_decisions
        .iter()
        .flat_map(|decision| decision.candidate_costs.iter())
        .filter(|cost| cost.source == source)
        .count() as u64
}
