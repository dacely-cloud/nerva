use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

pub(crate) fn record_tiny_decode_event(hidden: usize, vocab_size: usize, ledger: &mut TokenLedger) {
    ledger.record_execution_decision(ExecutionDecision {
        operation: "tiny_greedy_decode",
        executor_selected: ExecutionOwner::Cpu,
        candidate_costs: vec![
            CandidateCost::estimated("cpu-resident-reference", 1),
            CandidateCost::estimated("gpu-staged-reference", 3),
        ],
        reason: "tiny reference model is already resident in DRAM",
        predicted_visible_ns: 1,
        actual_visible_ns: Some(1),
        metric_source: MetricSource::EstimatedModel,
    });
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::DeviceActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(MemoryTier::Dram),
        to_tier: Some(MemoryTier::Dram),
        bytes: (hidden + vocab_size) * core::mem::size_of::<f32>(),
        latency_ns: 1,
        label: "tiny_greedy_decode_reference",
    });
}
