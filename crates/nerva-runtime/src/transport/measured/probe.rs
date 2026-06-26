use nerva_core::types::error::Result;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::runtime::Runtime;
use crate::transport::kernel_udp::matrix::run::run_kernel_udp_baseline_matrix_probe;
use crate::transport::measured::decision::record_measured_transport_decision;
use crate::transport::measured::source::MeasuredTransportSource;
use crate::transport::measured::summary::MeasuredTransportSelectorSummary;

const REFERENCE_DECODE_ACTIVATION_BYTES: usize = 32 * 1024;

impl Runtime {
    pub fn run_measured_transport_selector_probe(
        &self,
    ) -> Result<MeasuredTransportSelectorSummary> {
        let _ = self.config();
        run_measured_transport_selector_probe()
    }
}

pub fn run_measured_transport_selector_probe() -> Result<MeasuredTransportSelectorSummary> {
    let matrix = run_kernel_udp_baseline_matrix_probe()?;
    let source = MeasuredTransportSource::from_kernel_udp_matrix(
        REFERENCE_DECODE_ACTIVATION_BYTES,
        &matrix,
    )?;
    let mut ledger = TokenLedger::new(0);
    let decision = record_measured_transport_decision(&mut ledger, &source.candidates)?;
    ledger.require_zero_hot_path_allocations()?;
    Ok(MeasuredTransportSelectorSummary::from_ledger(
        &source, decision, &ledger,
    ))
}
