use nerva_core::types::cost::source::CostSource;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;

use crate::types::metric::MetricSource;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockVersionDependency {
    pub block_id: ResidentBlockId,
    pub required_version: u64,
    pub observed_version: u64,
    pub label: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateCost {
    pub label: &'static str,
    pub visible_ns: Option<u64>,
    pub source: CostSource,
}

impl CandidateCost {
    pub const fn estimated(label: &'static str, visible_ns: u64) -> Self {
        Self {
            label,
            visible_ns: Some(visible_ns),
            source: CostSource::Estimated,
        }
    }

    pub const fn measured(label: &'static str, visible_ns: u64) -> Self {
        Self {
            label,
            visible_ns: Some(visible_ns),
            source: CostSource::Measured,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidencyDecision {
    pub block_id: ResidentBlockId,
    pub old_tier: MemoryTier,
    pub new_tier: MemoryTier,
    pub executor_selected: ExecutionOwner,
    pub candidate_costs: Vec<CandidateCost>,
    pub reason: &'static str,
    pub predicted_overlap_ns: u64,
    pub actual_visible_ns: Option<u64>,
    pub metric_source: MetricSource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionDecision {
    pub operation: &'static str,
    pub executor_selected: ExecutionOwner,
    pub candidate_costs: Vec<CandidateCost>,
    pub reason: &'static str,
    pub predicted_visible_ns: u64,
    pub actual_visible_ns: Option<u64>,
    pub metric_source: MetricSource,
}
