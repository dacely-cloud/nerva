mod allocation;
mod blocks;
pub(crate) mod fixture;
mod spec;

use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;

use crate::execution::plan::planner::plan_execution_transaction;
use crate::execution::probe::fixture::reference_transaction_fixture;
use crate::execution::summary::ExecutionTransactionSummary;

pub fn run_execution_transaction_probe(
    device: DeviceOrdinal,
) -> Result<ExecutionTransactionSummary> {
    let (registry, spec, _) = reference_transaction_fixture(device)?;
    let plan = plan_execution_transaction(spec, &registry)?;
    Ok(plan.summary)
}
