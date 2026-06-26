use nerva_core::types::cost::source::CostSource;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::weights::execution::step::ResidentWeightExecutionStep;
use crate::weights::json::json_opt_string;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightExecutionPlan {
    pub steps: Vec<ResidentWeightExecutionStep>,
    pub total_weight_bytes: usize,
    pub total_predicted_visible_ns: u64,
    pub cpu_steps: u64,
    pub gpu_resident_steps: u64,
    pub gpu_staged_steps: u64,
    pub fallback_steps: u64,
    pub fallback_decisions: u64,
    pub block_version_dependencies: u64,
    pub first_tensor: Option<String>,
    pub last_tensor: Option<String>,
    pub ledger: TokenLedger,
}

impl ResidentWeightExecutionPlan {
    pub fn runtime_timestamp_decisions(&self) -> u64 {
        self.ledger
            .execution_decisions
            .iter()
            .filter(|decision| decision.metric_source == MetricSource::RuntimeTimestamp)
            .count() as u64
    }

    pub fn estimated_decisions(&self) -> u64 {
        self.ledger
            .execution_decisions
            .iter()
            .filter(|decision| decision.metric_source == MetricSource::EstimatedModel)
            .count() as u64
    }

    pub fn measured_candidate_costs(&self) -> u64 {
        self.candidate_costs_with_source(CostSource::Measured)
    }

    pub fn estimated_candidate_costs(&self) -> u64 {
        self.candidate_costs_with_source(CostSource::Estimated)
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"steps\":{},\"total_weight_bytes\":{},\"total_predicted_visible_ns\":{},\"cpu_steps\":{},\"gpu_resident_steps\":{},\"gpu_staged_steps\":{},\"fallback_steps\":{},\"fallback_decisions\":{},\"block_version_dependencies\":{},\"first_tensor\":{},\"last_tensor\":{},\"execution_decisions\":{},\"runtime_timestamp_decisions\":{},\"estimated_decisions\":{},\"measured_candidate_costs\":{},\"estimated_candidate_costs\":{},\"hot_path_allocations\":{}}}",
            self.steps.len(),
            self.total_weight_bytes,
            self.total_predicted_visible_ns,
            self.cpu_steps,
            self.gpu_resident_steps,
            self.gpu_staged_steps,
            self.fallback_steps,
            self.fallback_decisions,
            self.block_version_dependencies,
            json_opt_string(self.first_tensor.as_deref()),
            json_opt_string(self.last_tensor.as_deref()),
            self.ledger.execution_decisions.len(),
            self.runtime_timestamp_decisions(),
            self.estimated_decisions(),
            self.measured_candidate_costs(),
            self.estimated_candidate_costs(),
            self.ledger.hot_path_allocations,
        )
    }

    fn candidate_costs_with_source(&self, source: CostSource) -> u64 {
        self.ledger
            .execution_decisions
            .iter()
            .flat_map(|decision| decision.candidate_costs.iter())
            .filter(|cost| cost.source == source)
            .count() as u64
    }
}
