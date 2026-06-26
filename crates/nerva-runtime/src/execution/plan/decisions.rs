use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::execution::types::TransactionOperation;

pub fn record_execution_decision(ledger: &mut TokenLedger, operation: &TransactionOperation) {
    ledger.record_execution_decision(ExecutionDecision {
        operation: operation.name,
        executor_selected: operation.executor,
        candidate_costs: vec![
            CandidateCost::estimated("selected", operation.predicted_visible_ns),
            CandidateCost::estimated("host-roundtrip", operation.predicted_visible_ns * 4),
        ],
        reason: "transaction critical-path plan",
        predicted_visible_ns: operation.predicted_visible_ns,
        actual_visible_ns: None,
        metric_source: MetricSource::EstimatedModel,
    });
}
