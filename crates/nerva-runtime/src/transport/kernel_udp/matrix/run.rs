use nerva_core::types::error::Result;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::runtime::Runtime;
use crate::transport::kernel_udp::config::KernelUdpProbeConfig;
use crate::transport::kernel_udp::matrix::entry::KernelUdpBaselineMatrixEntry;
use crate::transport::kernel_udp::matrix::summary::KernelUdpBaselineMatrixSummary;
use crate::transport::kernel_udp::run::run_kernel_udp_baseline_probe;

const DEFAULT_MATRIX_PAYLOAD_BYTES: [usize; 3] = [32 * 1024, 256 * 1024, 1024 * 1024];
const DEFAULT_CHUNK_PAYLOAD_BYTES: usize = 4 * 1024;

impl Runtime {
    pub fn run_kernel_udp_baseline_matrix_probe(&self) -> Result<KernelUdpBaselineMatrixSummary> {
        let _ = self.config();
        run_kernel_udp_baseline_matrix_probe()
    }
}

pub fn run_kernel_udp_baseline_matrix_probe() -> Result<KernelUdpBaselineMatrixSummary> {
    let ledger = TokenLedger::new(0);
    let mut entries = Vec::with_capacity(DEFAULT_MATRIX_PAYLOAD_BYTES.len());
    for (index, payload_bytes) in DEFAULT_MATRIX_PAYLOAD_BYTES.iter().copied().enumerate() {
        let summary = run_kernel_udp_baseline_probe(matrix_config(index, payload_bytes))?;
        entries.push(KernelUdpBaselineMatrixEntry::from_summary(&summary));
    }
    ledger.require_zero_hot_path_allocations()?;
    Ok(KernelUdpBaselineMatrixSummary::from_entries(
        entries,
        ledger.hot_path_allocations,
    ))
}

fn matrix_config(index: usize, payload_bytes: usize) -> KernelUdpProbeConfig {
    let mut config = KernelUdpProbeConfig::reference_decode_activation();
    config.request_id = config.request_id.saturating_add(index as u64);
    config.sequence_id = config.sequence_id.saturating_add(index as u64);
    config.block_id = config.block_id.saturating_add(index as u64);
    config.payload_bytes = payload_bytes;
    config.chunk_payload_bytes = DEFAULT_CHUNK_PAYLOAD_BYTES;
    config
}
