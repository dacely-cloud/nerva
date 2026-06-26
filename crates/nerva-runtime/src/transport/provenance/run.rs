use crate::capabilities::snapshot::CapabilitySnapshot;
use crate::engine::runtime::Runtime;
use crate::transport::kernel_udp::matrix::run::run_kernel_udp_baseline_matrix_probe;
use crate::transport::matrix::run::run_transport_capability_matrix_probe;
use crate::transport::provenance::entry::build_transport_metric_provenance_entries;
use crate::transport::provenance::ledger::record_transport_provenance_events;
use crate::transport::provenance::summary::TransportMetricProvenanceSummary;
use nerva_core::types::error::Result;
use nerva_core::types::id::DeviceOrdinal;
use nerva_ledger::types::token::ledger::TokenLedger;

impl Runtime {
    pub fn run_transport_metric_provenance_probe(
        &self,
    ) -> Result<TransportMetricProvenanceSummary> {
        let capabilities = self.discover_capabilities();
        run_transport_metric_provenance_probe(self.config.device, &capabilities)
    }
}

pub fn run_transport_metric_provenance_probe(
    device: DeviceOrdinal,
    capabilities: &CapabilitySnapshot,
) -> Result<TransportMetricProvenanceSummary> {
    let measured = run_kernel_udp_baseline_matrix_probe()?;
    let estimated = run_transport_capability_matrix_probe(device, capabilities)?;
    let entries = build_transport_metric_provenance_entries(&measured, &estimated)?;
    let mut ledger = TokenLedger::new(0);
    record_transport_provenance_events(&mut ledger, &entries);
    ledger.require_zero_hot_path_allocations()?;
    Ok(TransportMetricProvenanceSummary::from_parts(
        &measured, &estimated, entries, &ledger,
    ))
}
