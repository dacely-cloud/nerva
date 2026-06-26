use nerva_core::types::cost::CostSource;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::measurements::planner::decision::MeasuredPlannerDecision;
use crate::measurements::planner::sources::PlannerMeasurementSources;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MeasuredPlannerStatus {
    Ok,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasuredPlannerSummary {
    pub status: MeasuredPlannerStatus,
    pub source_measurements: u64,
    pub runtime_timestamp_entries: u64,
    pub execution_decisions: u64,
    pub candidate_count: u64,
    pub measured_candidate_costs: u64,
    pub estimated_candidate_costs: u64,
    pub selected_label: &'static str,
    pub selected_executor: &'static str,
    pub predicted_visible_ns: u64,
    pub actual_visible_ns: u64,
    pub decision_metric_source: &'static str,
    pub all_candidates_measured: bool,
    pub selected_cost_measured: bool,
    pub hot_path_allocations: u64,
}

impl MeasuredPlannerSummary {
    pub(crate) fn from_ledger(
        sources: &PlannerMeasurementSources,
        decision: MeasuredPlannerDecision,
        ledger: &TokenLedger,
    ) -> Self {
        let selected_cost_measured = ledger.execution_decisions.iter().any(|entry| {
            entry.operation == "measured_exact_matvec_placement"
                && entry.candidate_costs.iter().any(|cost| {
                    cost.label == decision.selected.label && cost.source == CostSource::Measured
                })
                && entry.metric_source == MetricSource::RuntimeTimestamp
        });

        Self {
            status: MeasuredPlannerStatus::Ok,
            source_measurements: sources.source_measurements,
            runtime_timestamp_entries: sources.runtime_timestamp_entries,
            execution_decisions: ledger.execution_decisions.len() as u64,
            candidate_count: decision.candidate_count,
            measured_candidate_costs: decision.measured_candidate_costs,
            estimated_candidate_costs: decision.estimated_candidate_costs,
            selected_label: decision.selected.label,
            selected_executor: execution_owner_name(decision.selected.executor),
            predicted_visible_ns: decision.selected.visible_ns,
            actual_visible_ns: decision.selected.visible_ns,
            decision_metric_source: MetricSource::RuntimeTimestamp.as_str(),
            all_candidates_measured: decision.candidate_count == decision.measured_candidate_costs
                && decision.estimated_candidate_costs == 0,
            selected_cost_measured,
            hot_path_allocations: ledger.hot_path_allocations,
        }
    }

    pub fn passed(&self) -> bool {
        matches!(self.status, MeasuredPlannerStatus::Ok)
            && self.source_measurements >= 5
            && self.runtime_timestamp_entries == self.source_measurements
            && self.execution_decisions == 1
            && self.candidate_count >= 3
            && self.measured_candidate_costs == self.candidate_count
            && self.estimated_candidate_costs == 0
            && self.predicted_visible_ns > 0
            && self.actual_visible_ns == self.predicted_visible_ns
            && self.decision_metric_source == "runtime_timestamp"
            && self.all_candidates_measured
            && self.selected_cost_measured
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            MeasuredPlannerStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"source_measurements\":{},\"runtime_timestamp_entries\":{},\"execution_decisions\":{},\"candidate_count\":{},\"measured_candidate_costs\":{},\"estimated_candidate_costs\":{},\"selected_label\":\"{}\",\"selected_executor\":\"{}\",\"predicted_visible_ns\":{},\"actual_visible_ns\":{},\"decision_metric_source\":\"{}\",\"all_candidates_measured\":{},\"selected_cost_measured\":{},\"hot_path_allocations\":{}}}",
            status,
            self.source_measurements,
            self.runtime_timestamp_entries,
            self.execution_decisions,
            self.candidate_count,
            self.measured_candidate_costs,
            self.estimated_candidate_costs,
            json_escape(self.selected_label),
            self.selected_executor,
            self.predicted_visible_ns,
            self.actual_visible_ns,
            self.decision_metric_source,
            self.all_candidates_measured,
            self.selected_cost_measured,
            self.hot_path_allocations,
        )
    }
}

fn execution_owner_name(owner: ExecutionOwner) -> &'static str {
    match owner {
        ExecutionOwner::Cpu => "cpu",
        ExecutionOwner::Gpu(_) => "gpu",
        ExecutionOwner::Nic(_) => "nic",
        ExecutionOwner::SharedReadOnly => "shared_read_only",
        ExecutionOwner::PhaseTransition => "phase_transition",
        ExecutionOwner::None => "none",
    }
}

fn json_escape(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}
