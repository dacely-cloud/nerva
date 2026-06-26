use crate::types::decision::{BlockVersionDependency, ExecutionDecision, ResidencyDecision};
use crate::types::event::{DeviceTimelineSpan, LedgerEvent};
use crate::types::fallback::FallbackDecision;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TokenLedger {
    pub token_index: u64,
    pub events: Vec<LedgerEvent>,
    pub device_timeline: Vec<DeviceTimelineSpan>,
    pub fallback_decisions: Vec<FallbackDecision>,
    pub block_version_dependencies: Vec<BlockVersionDependency>,
    pub residency_decisions: Vec<ResidencyDecision>,
    pub execution_decisions: Vec<ExecutionDecision>,
    pub hot_path_allocations: u64,
}

impl TokenLedger {
    pub fn new(token_index: u64) -> Self {
        Self {
            token_index,
            events: Vec::new(),
            device_timeline: Vec::new(),
            fallback_decisions: Vec::new(),
            block_version_dependencies: Vec::new(),
            residency_decisions: Vec::new(),
            execution_decisions: Vec::new(),
            hot_path_allocations: 0,
        }
    }
}
