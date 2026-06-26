use nerva_core::types::error::Result;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::runtime::Runtime;
use crate::measurements::planner::decision::record_measured_planner_decision;
use crate::measurements::planner::sources::PlannerMeasurementSources;
use crate::measurements::planner::summary::MeasuredPlannerSummary;
use crate::measurements::probe::run_measurement_table_probe;

impl Runtime {
    pub fn run_measured_planner_probe(&self) -> Result<MeasuredPlannerSummary> {
        let _ = self.config();
        run_measured_planner_probe()
    }
}

pub fn run_measured_planner_probe() -> Result<MeasuredPlannerSummary> {
    let measurements = run_measurement_table_probe()?;
    let sources = PlannerMeasurementSources::from_entries(&measurements.entries)?;
    let candidates = sources.candidates();
    let mut ledger = TokenLedger::new(0);
    let decision = record_measured_planner_decision(&mut ledger, &candidates)?;
    ledger.require_zero_hot_path_allocations()?;
    Ok(MeasuredPlannerSummary::from_ledger(
        &sources, decision, &ledger,
    ))
}
