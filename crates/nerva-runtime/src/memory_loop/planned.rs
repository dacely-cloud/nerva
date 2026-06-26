use nerva_ledger::types::token::ledger::TokenLedger;

use crate::memory_loop::summary::MemoryLoopSummary;
use crate::memory_loop::types::MemoryLoopTaskSpec;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryLoopTask {
    pub task_index: u64,
    pub spec: MemoryLoopTaskSpec,
    pub actual_visible_ns: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryLoopPlan {
    pub queue_capacity: usize,
    pub max_inflight: usize,
    pub tasks: Vec<MemoryLoopTask>,
    pub ledger: TokenLedger,
    pub summary: MemoryLoopSummary,
}
