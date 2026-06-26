use nerva_core::types::error::Result;

use crate::correctness::probe::run_correctness_validation_probe;
use crate::correctness::summary::CorrectnessValidationSummary;
use crate::engine::runtime::Runtime;

impl Runtime {
    pub fn run_correctness_validation_probe(&self) -> Result<CorrectnessValidationSummary> {
        let _ = self.config;
        run_correctness_validation_probe()
    }
}
