use nerva_core::types::error::Result;

use crate::engine::runtime::Runtime;
use crate::memory_loop::probe;
use crate::memory_loop::summary::MemoryLoopSummary;

impl Runtime {
    pub fn run_memory_loop_probe(&self) -> Result<MemoryLoopSummary> {
        let _ = self.config;
        probe::run_memory_loop_probe()
    }
}
