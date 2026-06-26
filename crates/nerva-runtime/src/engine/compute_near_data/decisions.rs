use nerva_core::types::id::DeviceOrdinal;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::compute_near_data::shard::ResidentMatvecShard;

pub(crate) fn record_cpu_shard_decision(shard: &ResidentMatvecShard<'_>, ledger: &mut TokenLedger) {
    let bytes = shard.weights.len() * core::mem::size_of::<f32>();
    ledger.record_execution_decision(ExecutionDecision {
        operation: "resident_split_matvec",
        executor_selected: ExecutionOwner::Cpu,
        candidate_costs: vec![
            CandidateCost::estimated("compute-near-dram", bytes as u64),
            CandidateCost::estimated("stage-dram-to-gpu", bytes as u64 + 8),
            CandidateCost::estimated("gpu-resident", u64::MAX / 4),
        ],
        reason: "DRAM resident shard stays on CPU because moving it would expose copy latency",
        predicted_visible_ns: bytes as u64,
        actual_visible_ns: Some(bytes as u64),
        metric_source: MetricSource::EstimatedModel,
    });
}

pub(crate) fn record_gpu_shard_decision(
    device: DeviceOrdinal,
    shard: &ResidentMatvecShard<'_>,
    ledger: &mut TokenLedger,
) {
    let rows = shard.row_end - shard.row_start;
    ledger.record_execution_decision(ExecutionDecision {
        operation: "resident_split_matvec",
        executor_selected: ExecutionOwner::Gpu(device),
        candidate_costs: vec![
            CandidateCost::estimated("gpu-resident", rows as u64),
            CandidateCost::estimated("copy-to-cpu", shard.weights.len() as u64),
            CandidateCost::estimated("compute-near-dram", u64::MAX / 4),
        ],
        reason: "VRAM resident shard executes on GPU and merges only the small output rows",
        predicted_visible_ns: rows as u64,
        actual_visible_ns: Some(rows as u64),
        metric_source: MetricSource::EstimatedModel,
    });
}
