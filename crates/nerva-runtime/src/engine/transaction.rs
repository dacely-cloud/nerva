use nerva_core::types::error::Result;

use crate::engine::runtime::Runtime;
use crate::execution::probe;
use crate::execution::summary::ExecutionTransactionSummary;

impl Runtime {
    pub fn run_execution_transaction_probe(&self) -> Result<ExecutionTransactionSummary> {
        probe::run_execution_transaction_probe(self.config.device)
    }
}
