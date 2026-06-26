use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::compute_near_data::shard::ResidentMatvecShard;
use crate::measurements::entry::{MeasurementEntry, MeasurementKind};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct ComputeNearDataCosts {
    pub(crate) cpu_kernel_ns: u64,
    pub(crate) cpu_copy_ns: u64,
    pub(crate) merge_ns: u64,
}

impl ComputeNearDataCosts {
    pub(crate) fn from_measurements(entries: &[MeasurementEntry]) -> Result<Self> {
        Ok(Self {
            cpu_kernel_ns: per_iteration_ns(entries, MeasurementKind::CpuKernel)?,
            cpu_copy_ns: per_iteration_ns(entries, MeasurementKind::CpuCopy)?,
            merge_ns: per_iteration_ns(entries, MeasurementKind::Merge)?,
        })
    }
}

pub(crate) fn record_cpu_shard_decision(
    shard: &ResidentMatvecShard<'_>,
    costs: ComputeNearDataCosts,
    ledger: &mut TokenLedger,
) {
    let rows = shard_rows(shard);
    let cpu_visible_ns = costs.cpu_kernel_ns.saturating_mul(rows);
    let staged_visible_ns = costs.cpu_copy_ns.saturating_add(costs.merge_ns);
    ledger.record_execution_decision(ExecutionDecision {
        operation: "resident_split_matvec",
        executor_selected: ExecutionOwner::Cpu,
        candidate_costs: vec![
            CandidateCost::measured("compute-near-dram", cpu_visible_ns),
            CandidateCost::measured("stage-dram-to-gpu", staged_visible_ns),
        ],
        reason: "DRAM resident shard stays on CPU because moving it would expose copy latency",
        predicted_visible_ns: cpu_visible_ns,
        actual_visible_ns: Some(cpu_visible_ns),
        metric_source: MetricSource::RuntimeTimestamp,
    });
}

pub(crate) fn record_gpu_shard_decision(
    device: DeviceOrdinal,
    shard: &ResidentMatvecShard<'_>,
    costs: ComputeNearDataCosts,
    ledger: &mut TokenLedger,
) {
    let rows = shard_rows(shard);
    let resident_visible_ns = costs.cpu_kernel_ns.saturating_mul(rows);
    let copy_to_cpu_ns = costs.cpu_copy_ns.saturating_add(costs.merge_ns);
    ledger.record_execution_decision(ExecutionDecision {
        operation: "resident_split_matvec",
        executor_selected: ExecutionOwner::Gpu(device),
        candidate_costs: vec![
            CandidateCost::measured("gpu-resident", resident_visible_ns),
            CandidateCost::measured("copy-to-cpu", copy_to_cpu_ns),
        ],
        reason: "VRAM resident shard executes on GPU and merges only the small output rows",
        predicted_visible_ns: resident_visible_ns,
        actual_visible_ns: Some(resident_visible_ns),
        metric_source: MetricSource::RuntimeTimestamp,
    });
}

fn shard_rows(shard: &ResidentMatvecShard<'_>) -> u64 {
    (shard.row_end - shard.row_start).max(1) as u64
}

fn per_iteration_ns(entries: &[MeasurementEntry], kind: MeasurementKind) -> Result<u64> {
    entries
        .iter()
        .find(|entry| entry.kind == kind)
        .map(|entry| {
            entry
                .elapsed_ns
                .saturating_div(entry.iterations.max(1))
                .max(1)
        })
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("compute-near-data missing {} measurement", kind.as_str()),
        })
}
