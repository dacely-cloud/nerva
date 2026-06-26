use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::weights::json::json_opt_string;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightPrefetchTask {
    pub task_index: u64,
    pub block_id: ResidentBlockId,
    pub name: String,
    pub source_shard: String,
    pub file_offset_begin: usize,
    pub file_offset_end: usize,
    pub bytes: usize,
    pub target_tier: MemoryTier,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightPrefetchPlan {
    pub tasks: Vec<ResidentWeightPrefetchTask>,
    pub total_bytes: usize,
    pub shard_count: usize,
    pub max_task_bytes: usize,
    pub prefetch_events: u64,
    pub copy_events: u64,
    pub first_source_shard: Option<String>,
    pub last_source_shard: Option<String>,
    pub ledger: TokenLedger,
}

impl ResidentWeightPrefetchPlan {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"tasks\":{},\"total_bytes\":{},\"shard_count\":{},\"max_task_bytes\":{},\"prefetch_events\":{},\"copy_events\":{},\"first_source_shard\":{},\"last_source_shard\":{},\"hot_path_allocations\":{}}}",
            self.tasks.len(),
            self.total_bytes,
            self.shard_count,
            self.max_task_bytes,
            self.prefetch_events,
            self.copy_events,
            json_opt_string(self.first_source_shard.as_deref()),
            json_opt_string(self.last_source_shard.as_deref()),
            self.ledger.hot_path_allocations,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightPrefetchExecutionSummary {
    pub tasks: usize,
    pub completed_blocks: usize,
    pub total_bytes: usize,
    pub prefetch_events: u64,
    pub copy_events: u64,
    pub ready_blocks: usize,
    pub hot_path_allocations: u64,
    pub ledger: TokenLedger,
}

impl ResidentWeightPrefetchExecutionSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"tasks\":{},\"completed_blocks\":{},\"total_bytes\":{},\"prefetch_events\":{},\"copy_events\":{},\"ready_blocks\":{},\"hot_path_allocations\":{}}}",
            self.tasks,
            self.completed_blocks,
            self.total_bytes,
            self.prefetch_events,
            self.copy_events,
            self.ready_blocks,
            self.hot_path_allocations,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightPrefetchIoSummary {
    pub tasks: usize,
    pub completed_blocks: usize,
    pub total_bytes: usize,
    pub shard_count: usize,
    pub disk_read_events: u64,
    pub copy_events: u64,
    pub ready_blocks: usize,
    pub data_hash: u64,
    pub hot_path_allocations: u64,
    pub ledger: TokenLedger,
}

impl ResidentWeightPrefetchIoSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"tasks\":{},\"completed_blocks\":{},\"total_bytes\":{},\"shard_count\":{},\"disk_read_events\":{},\"copy_events\":{},\"ready_blocks\":{},\"data_hash\":{},\"hot_path_allocations\":{}}}",
            self.tasks,
            self.completed_blocks,
            self.total_bytes,
            self.shard_count,
            self.disk_read_events,
            self.copy_events,
            self.ready_blocks,
            self.data_hash,
            self.hot_path_allocations,
        )
    }
}
