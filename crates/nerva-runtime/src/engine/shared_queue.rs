use nerva_core::types::error::Result;
use nerva_memory::queue::probe::run::run_shared_work_queue_probe;
use nerva_memory::queue::summary::SharedQueueProbeSummary;

use crate::engine::runtime::Runtime;

impl Runtime {
    pub fn run_shared_work_queue_probe(&self) -> Result<SharedQueueProbeSummary> {
        let _ = self.config;
        run_shared_work_queue_probe()
    }
}
