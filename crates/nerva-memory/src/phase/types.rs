use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::ownership::owner::ExecutionOwner;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PhaseHandoffRequest {
    pub block_id: ResidentBlockId,
    pub from: ExecutionOwner,
    pub to: ExecutionOwner,
    pub required_version: u64,
    pub reason: &'static str,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PhaseHandoffEntry {
    pub block_id: ResidentBlockId,
    pub from: ExecutionOwner,
    pub to: ExecutionOwner,
    pub bytes: usize,
    pub version_before: u64,
    pub predicted_visible_ns: u64,
    pub reason: &'static str,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PhaseHandoffRejectionKind {
    MissingBlock,
    BlockNotReady,
    StaleVersion,
    OwnerMismatch,
    IllegalTransition,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PhaseHandoffRejection {
    pub block_id: ResidentBlockId,
    pub requested_from: ExecutionOwner,
    pub requested_to: ExecutionOwner,
    pub kind: PhaseHandoffRejectionKind,
    pub observed_owner: ExecutionOwner,
    pub observed_version: u64,
    pub reason: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseHandoffPlan {
    pub entries: Vec<PhaseHandoffEntry>,
    pub rejections: Vec<PhaseHandoffRejection>,
}

impl PhaseHandoffPlan {
    pub fn rejected_count(&self, kind: PhaseHandoffRejectionKind) -> u64 {
        self.rejections
            .iter()
            .filter(|rejection| rejection.kind == kind)
            .count() as u64
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PhaseHandoffApplySummary {
    pub applied_handoffs: u64,
    pub version_publications: u64,
    pub final_max_version: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PhaseHandoffPlanner;
