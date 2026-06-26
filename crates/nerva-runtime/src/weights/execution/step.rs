use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::ownership::owner::ExecutionOwner;

use crate::weights::execution::strategy::ResidentWeightExecutionStrategy;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightExecutionStep {
    pub step_index: u64,
    pub block_id: ResidentBlockId,
    pub name: String,
    pub strategy: ResidentWeightExecutionStrategy,
    pub executor: ExecutionOwner,
    pub bytes: usize,
    pub block_version: u64,
    pub predicted_visible_ns: u64,
    pub kernel_name: &'static str,
    pub fallback: bool,
}
