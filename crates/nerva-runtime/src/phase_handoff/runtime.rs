use nerva_core::types::error::Result;
use nerva_memory::phase::probe::run_phase_handoff_probe;
use nerva_memory::phase::summary::PhaseHandoffProbeSummary;

use crate::engine::runtime::Runtime;

impl Runtime {
    pub fn run_phase_handoff_probe(&self) -> Result<PhaseHandoffProbeSummary> {
        let _ = self.config;
        run_phase_handoff_probe()
    }
}
