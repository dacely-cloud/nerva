use nerva_core::types::error::Result;

use crate::engine::runtime::Runtime;
use crate::production::probe::run_production_invariant_probe;
use crate::production::summary::ProductionInvariantSummary;

impl Runtime {
    pub fn run_production_invariant_probe(&self) -> Result<ProductionInvariantSummary> {
        let _ = self.config;
        run_production_invariant_probe()
    }
}
