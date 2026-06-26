use std::time::Instant;

use nerva_core::types::error::Result;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::hash::hash_f32s;
use crate::common::validate::require_len;
use crate::warm_compute::probe::footprint::WarmComputeFootprint;
use crate::warm_compute::probe::strategy_run::{
    run_cpu_dram, run_gpu_resident, run_gpu_staged, run_hybrid_split,
};
use crate::warm_compute::strategy::WarmComputeStrategy;
use crate::warm_compute::summary::WarmComputeCandidate;

pub(crate) fn run_warm_compute_candidate(
    strategy: WarmComputeStrategy,
    rows: usize,
    cols: usize,
    matrix: &[f32],
    input: &[f32],
    ledger: &mut TokenLedger,
) -> Result<WarmComputeCandidate> {
    require_len("warm compute matrix", matrix.len(), rows * cols)?;
    require_len("warm compute input", input.len(), cols)?;
    let mut output = vec![0.0; rows];
    let footprint = WarmComputeFootprint::new(matrix.len(), input.len(), output.len());

    let start = Instant::now();
    let visible_ns = match strategy {
        WarmComputeStrategy::CpuDram => {
            run_cpu_dram(rows, cols, matrix, input, &mut output, footprint, ledger)
        }
        WarmComputeStrategy::GpuResident => {
            run_gpu_resident(rows, matrix, input, &mut output, footprint, ledger)
        }
        WarmComputeStrategy::GpuStaged => {
            run_gpu_staged(rows, matrix, input, &mut output, footprint, ledger)
        }
        WarmComputeStrategy::HybridSplit => {
            run_hybrid_split(rows, cols, matrix, input, &mut output, ledger)?
        }
    };
    let measured_ns = elapsed_ns(start);

    Ok(WarmComputeCandidate {
        strategy,
        visible_ns,
        measured_ns,
        output_hash: hash_f32s(&output),
    })
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().max(1).min(u64::MAX as u128) as u64
}
