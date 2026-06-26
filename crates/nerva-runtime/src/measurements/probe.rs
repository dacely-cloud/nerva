use nerva_core::types::error::Result;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::runtime::Runtime;
use crate::measurements::copy::measure_cpu_copy;
use crate::measurements::cpu::measure_cpu_dot;
use crate::measurements::merge::measure_merge;
use crate::measurements::queue::measure_queue_round_trip;
use crate::measurements::summary::MeasurementTableSummary;
use crate::measurements::sync_loop::measure_sync_loop;
use crate::measurements::table::MeasurementTable;

impl Runtime {
    pub fn run_measurement_table_probe(&self) -> Result<MeasurementTableSummary> {
        let _ = self.config();
        run_measurement_table_probe()
    }
}

pub fn run_measurement_table_probe() -> Result<MeasurementTableSummary> {
    let ledger = TokenLedger::new(0);
    let table = MeasurementTable::new(vec![
        measure_cpu_copy(),
        measure_cpu_dot(),
        measure_merge(),
        measure_queue_round_trip(),
        measure_sync_loop(),
    ]);
    ledger.require_zero_hot_path_allocations()?;
    Ok(MeasurementTableSummary::from_table(
        table,
        ledger.hot_path_allocations,
    ))
}
