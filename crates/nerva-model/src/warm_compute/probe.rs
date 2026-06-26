use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::MemoryTier;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::TokenLedger;

use crate::common::hash::hash_f32s;
use crate::common::math::{mat_vec_row_major, mat_vec_row_range};
use crate::common::validate::require_len;
use crate::warm_compute::strategy::WarmComputeStrategy;
use crate::warm_compute::summary::{
    WarmComputeCandidate, WarmComputeProbeStatus, WarmComputeProbeSummary,
};

pub fn warm_compute_probe() -> Result<WarmComputeProbeSummary> {
    const ROWS: usize = 4;
    const COLS: usize = 4;
    let matrix = [
        1.0, 0.0, 0.0, 1.0, 0.5, -1.0, 2.0, 0.0, -1.0, 0.0, 1.0, 0.5, 0.0, 2.0, 0.25, -0.5,
    ];
    let input = [1.0, -2.0, 0.5, 3.0];
    let mut ledger = TokenLedger::new(0);
    let mut candidates = Vec::new();

    for strategy in [
        WarmComputeStrategy::CpuDram,
        WarmComputeStrategy::GpuResident,
        WarmComputeStrategy::GpuStaged,
        WarmComputeStrategy::HybridSplit,
    ] {
        candidates.push(run_warm_compute_candidate(
            strategy,
            ROWS,
            COLS,
            &matrix,
            &input,
            &mut ledger,
        )?);
    }

    let output_hash = candidates
        .first()
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "warm compute probe produced no candidates".to_string(),
        })?
        .output_hash;
    let parity = candidates
        .iter()
        .all(|candidate| candidate.output_hash == output_hash);
    if !parity {
        return Err(NervaError::InvalidArgument {
            reason: "warm compute candidate parity failed".to_string(),
        });
    }

    let selected = candidates
        .iter()
        .min_by_key(|candidate| candidate.visible_ns)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "warm compute candidate selection failed".to_string(),
        })?;
    let selected_strategy = selected.strategy;
    let selected_visible_ns = selected.visible_ns;
    let cpu_visible = candidate_visible_ns(&candidates, WarmComputeStrategy::CpuDram)?;
    let staged_visible = candidate_visible_ns(&candidates, WarmComputeStrategy::GpuStaged)?;

    ledger.record_execution_decision(ExecutionDecision {
        operation: "dense_matvec",
        executor_selected: selected_strategy.executor(),
        candidate_costs: candidates
            .iter()
            .map(|candidate| {
                CandidateCost::estimated(candidate.strategy.label(), candidate.visible_ns)
            })
            .collect(),
        reason: "select exact candidate with lowest visible critical-path cost",
        predicted_visible_ns: selected_visible_ns,
        actual_visible_ns: Some(selected_visible_ns),
        metric_source: MetricSource::EstimatedModel,
    });
    ledger.require_zero_hot_path_allocations()?;

    let copy_bytes = ledger
        .events
        .iter()
        .filter(|event| event.kind == LedgerEventKind::Copy)
        .map(|event| event.bytes)
        .sum();

    Ok(WarmComputeProbeSummary {
        status: WarmComputeProbeStatus::Ok,
        rows: ROWS,
        cols: COLS,
        candidates,
        selected_strategy,
        parity,
        cpu_beats_staged: cpu_visible < staged_visible,
        execution_decisions: ledger.execution_decisions.len() as u64,
        cpu_events: ledger.event_count(LedgerEventKind::CpuActivity),
        device_events: ledger.event_count(LedgerEventKind::DeviceActivity),
        copy_events: ledger.event_count(LedgerEventKind::Copy),
        copy_bytes,
        total_latency_ns: ledger.total_latency_ns(),
        hot_path_allocations: ledger.hot_path_allocations,
        output_hash,
    })
}

fn run_warm_compute_candidate(
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
    let matrix_bytes = matrix.len() * core::mem::size_of::<f32>();
    let input_bytes = input.len() * core::mem::size_of::<f32>();
    let output_bytes = output.len() * core::mem::size_of::<f32>();

    let visible_ns = match strategy {
        WarmComputeStrategy::CpuDram => {
            mat_vec_row_major(matrix, input, &mut output);
            let compute_ns = (rows * cols) as u64;
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::CpuActivity,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: None,
                from_tier: Some(MemoryTier::Dram),
                to_tier: Some(MemoryTier::Dram),
                bytes: matrix_bytes + input_bytes + output_bytes,
                latency_ns: compute_ns,
                label: "warm_matvec_cpu_dram",
            });
            compute_ns
        }
        WarmComputeStrategy::GpuResident => {
            mat_vec_row_major(matrix, input, &mut output);
            let compute_ns = rows as u64;
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::DeviceActivity,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: None,
                from_tier: Some(MemoryTier::Vram),
                to_tier: Some(MemoryTier::Vram),
                bytes: matrix_bytes + input_bytes + output_bytes,
                latency_ns: compute_ns,
                label: "warm_matvec_gpu_resident",
            });
            compute_ns
        }
        WarmComputeStrategy::GpuStaged => {
            let copy_in_ns = (matrix_bytes + input_bytes) as u64;
            let compute_ns = rows as u64;
            let copy_out_ns = output_bytes as u64;
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::Copy,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: None,
                from_tier: Some(MemoryTier::Dram),
                to_tier: Some(MemoryTier::Vram),
                bytes: matrix_bytes + input_bytes,
                latency_ns: copy_in_ns,
                label: "warm_matvec_stage_to_gpu",
            });
            mat_vec_row_major(matrix, input, &mut output);
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::DeviceActivity,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: None,
                from_tier: Some(MemoryTier::Vram),
                to_tier: Some(MemoryTier::Vram),
                bytes: matrix_bytes + input_bytes + output_bytes,
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
                bytes: output_bytes,
                latency_ns: copy_out_ns,
                label: "warm_matvec_stage_from_gpu",
            });
            copy_in_ns + compute_ns + copy_out_ns
        }
        WarmComputeStrategy::HybridSplit => {
            let split = rows / 2;
            mat_vec_row_range(matrix, input, cols, 0, split, &mut output)?;
            mat_vec_row_range(matrix, input, cols, split, rows, &mut output)?;
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
            cpu_ns.max(gpu_ns) + merge_ns
        }
    };

    Ok(WarmComputeCandidate {
        strategy,
        visible_ns,
        output_hash: hash_f32s(&output),
    })
}

fn candidate_visible_ns(
    candidates: &[WarmComputeCandidate],
    strategy: WarmComputeStrategy,
) -> Result<u64> {
    candidates
        .iter()
        .find(|candidate| candidate.strategy == strategy)
        .map(|candidate| candidate.visible_ns)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("missing warm compute candidate {}", strategy.label()),
        })
}
