use nerva_core::types::error::Result;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::math::{mat_vec_row_major, mat_vec_row_range};
use crate::warm_compute::probe::footprint::WarmComputeFootprint;

pub(crate) fn run_cpu_dram(
    rows: usize,
    cols: usize,
    matrix: &[f32],
    input: &[f32],
    output: &mut [f32],
    footprint: WarmComputeFootprint,
    ledger: &mut TokenLedger,
) -> u64 {
    mat_vec_row_major(matrix, input, output);
    let compute_ns = (rows * cols) as u64;
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::CpuActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(MemoryTier::Dram),
        to_tier: Some(MemoryTier::Dram),
        bytes: footprint.matrix_bytes + footprint.input_bytes + footprint.output_bytes,
        latency_ns: compute_ns,
        label: "warm_matvec_cpu_dram",
    });
    compute_ns
}

pub(crate) fn run_gpu_resident(
    rows: usize,
    matrix: &[f32],
    input: &[f32],
    output: &mut [f32],
    footprint: WarmComputeFootprint,
    ledger: &mut TokenLedger,
) -> u64 {
    mat_vec_row_major(matrix, input, output);
    let compute_ns = rows as u64;
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::DeviceActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(MemoryTier::Vram),
        to_tier: Some(MemoryTier::Vram),
        bytes: footprint.matrix_bytes + footprint.input_bytes + footprint.output_bytes,
        latency_ns: compute_ns,
        label: "warm_matvec_gpu_resident",
    });
    compute_ns
}

pub(crate) fn run_gpu_staged(
    rows: usize,
    matrix: &[f32],
    input: &[f32],
    output: &mut [f32],
    footprint: WarmComputeFootprint,
    ledger: &mut TokenLedger,
) -> u64 {
    let copy_in_ns = (footprint.matrix_bytes + footprint.input_bytes) as u64;
    let compute_ns = rows as u64;
    let copy_out_ns = footprint.output_bytes as u64;
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Copy,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(MemoryTier::Dram),
        to_tier: Some(MemoryTier::Vram),
        bytes: footprint.matrix_bytes + footprint.input_bytes,
        latency_ns: copy_in_ns,
        label: "warm_matvec_stage_to_gpu",
    });
    mat_vec_row_major(matrix, input, output);
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::DeviceActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(MemoryTier::Vram),
        to_tier: Some(MemoryTier::Vram),
        bytes: footprint.matrix_bytes + footprint.input_bytes + footprint.output_bytes,
        latency_ns: compute_ns,
        label: "warm_matvec_gpu_staged_compute",
    });
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Copy,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(MemoryTier::Vram),
        to_tier: Some(MemoryTier::Dram),
        bytes: footprint.output_bytes,
        latency_ns: copy_out_ns,
        label: "warm_matvec_stage_from_gpu",
    });
    copy_in_ns + compute_ns + copy_out_ns
}

pub(crate) fn run_hybrid_split(
    rows: usize,
    cols: usize,
    matrix: &[f32],
    input: &[f32],
    output: &mut [f32],
    ledger: &mut TokenLedger,
) -> Result<u64> {
    let split = rows / 2;
    mat_vec_row_range(matrix, input, cols, 0, split, output)?;
    mat_vec_row_range(matrix, input, cols, split, rows, output)?;
    let cpu_ns = (split * cols) as u64;
    let gpu_ns = rows.saturating_sub(split) as u64;
    let merge_bytes = rows.saturating_sub(split) * core::mem::size_of::<f32>();
    let merge_ns = merge_bytes as u64;
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::CpuActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(MemoryTier::Dram),
        to_tier: Some(MemoryTier::Dram),
        bytes: split * cols * core::mem::size_of::<f32>(),
        latency_ns: cpu_ns,
        label: "warm_matvec_hybrid_cpu_rows",
    });
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::DeviceActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(MemoryTier::Vram),
        to_tier: Some(MemoryTier::Vram),
        bytes: rows.saturating_sub(split) * cols * core::mem::size_of::<f32>(),
        latency_ns: gpu_ns,
        label: "warm_matvec_hybrid_gpu_rows",
    });
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Copy,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(MemoryTier::Vram),
        to_tier: Some(MemoryTier::Dram),
        bytes: merge_bytes,
        latency_ns: merge_ns,
        label: "warm_matvec_hybrid_merge",
    });
    Ok(cpu_ns.max(gpu_ns) + merge_ns)
}
