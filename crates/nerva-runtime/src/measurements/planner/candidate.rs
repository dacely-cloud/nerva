use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::decision::CandidateCost;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct MeasuredPlannerCandidate {
    pub label: &'static str,
    pub executor: ExecutionOwner,
    pub visible_ns: u64,
}

impl MeasuredPlannerCandidate {
    pub(crate) const fn as_cost(self) -> CandidateCost {
        CandidateCost::measured(self.label, self.visible_ns)
    }
}
