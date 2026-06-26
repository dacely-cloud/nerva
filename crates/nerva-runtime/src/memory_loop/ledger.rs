use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ResidencyDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::TokenLedger;

use crate::memory_loop::types::{MemoryLoopTaskKind, MemoryLoopTaskSpec};

pub(crate) fn record_task_plan(
    ledger: &mut TokenLedger,
    spec: MemoryLoopTaskSpec,
    actual_visible_ns: u64,
) {
    ledger.record_residency_decision(ResidencyDecision {
        block_id: spec.block_id,
        old_tier: spec.from_tier,
        new_tier: spec.to_tier,
        executor_selected: executor_for_task(spec.kind),
        candidate_costs: vec![
            CandidateCost::estimated("planned", actual_visible_ns),
            CandidateCost::estimated("reactive-critical-path", spec.predicted_visible_ns * 2),
        ],
        reason: spec.kind.label(),
        predicted_overlap_ns: spec.overlap_window_ns,
        actual_visible_ns: None,
        metric_source: MetricSource::EstimatedModel,
    });
    ledger.record(LedgerEvent {
        kind: event_kind_for_task(spec.kind),
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(spec.block_id),
        from_tier: Some(spec.from_tier),
        to_tier: Some(spec.to_tier),
        bytes: spec.bytes,
        latency_ns: actual_visible_ns,
        label: spec.label,
    });
    if spec.from_tier != spec.to_tier {
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Copy,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: Some(spec.block_id),
            from_tier: Some(spec.from_tier),
            to_tier: Some(spec.to_tier),
            bytes: spec.bytes,
            latency_ns: actual_visible_ns,
            label: "memory_loop_explicit_copy",
        });
    }
    ledger.record_sync(
        SyncClass::PhaseHandoff,
        Some(spec.block_id),
        Some(spec.from_tier),
        Some(spec.to_tier),
        spec.bytes,
        1,
        MetricSource::EstimatedModel,
        "memory_loop_phase_handoff",
    );
}

fn executor_for_task(kind: MemoryLoopTaskKind) -> ExecutionOwner {
    match kind {
        MemoryLoopTaskKind::DiskRead
        | MemoryLoopTaskKind::Prefetch
        | MemoryLoopTaskKind::Stage
        | MemoryLoopTaskKind::Evict
        | MemoryLoopTaskKind::PrepareTransportBuffer => ExecutionOwner::Cpu,
    }
}

fn event_kind_for_task(kind: MemoryLoopTaskKind) -> LedgerEventKind {
    match kind {
        MemoryLoopTaskKind::DiskRead | MemoryLoopTaskKind::Prefetch | MemoryLoopTaskKind::Stage => {
            LedgerEventKind::Prefetch
        }
        MemoryLoopTaskKind::Evict => LedgerEventKind::Eviction,
        MemoryLoopTaskKind::PrepareTransportBuffer => LedgerEventKind::Transport,
    }
}
