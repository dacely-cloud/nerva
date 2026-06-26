use crate::capabilities::snapshot::CapabilitySnapshot;
use crate::transport::matrix::entry;
use crate::transport::matrix::summary;
use crate::transport::matrix::types::TransportCapabilityMatrixSummary;
use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_ledger::types::token::ledger::TokenLedger;

pub fn run_transport_capability_matrix_probe(
    device: DeviceOrdinal,
    capabilities: &CapabilitySnapshot,
) -> Result<TransportCapabilityMatrixSummary> {
    let (sizes, entries) = entry::build_entries(device, capabilities)?;
    let ledger = TokenLedger::new(0);
    ledger.require_zero_hot_path_allocations()?;
    Ok(summary::transport_capability_matrix_summary(
        sizes,
        entries,
        ledger.hot_path_allocations,
    ))
}
