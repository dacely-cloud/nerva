use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::memory_loop::planned::MemoryLoopTask;
use crate::memory_loop::summary::{MemoryLoopStatus, MemoryLoopSummary};
use crate::memory_loop::types::MemoryLoopTaskKind;

pub(crate) fn summarize_plan(
    queue_capacity: usize,
    max_inflight: usize,
    tasks: &[MemoryLoopTask],
    ledger: &TokenLedger,
    ready_blocks: u64,
) -> MemoryLoopSummary {
    let disk_read_tasks = count_kind(tasks, MemoryLoopTaskKind::DiskRead);
    let prefetch_tasks = count_kind(tasks, MemoryLoopTaskKind::Prefetch);
    let staging_tasks = count_kind(tasks, MemoryLoopTaskKind::Stage);
    let eviction_tasks = count_kind(tasks, MemoryLoopTaskKind::Evict);
    let transport_prepare_tasks = count_kind(tasks, MemoryLoopTaskKind::PrepareTransportBuffer);
    let total_predicted_visible_ns = tasks
        .iter()
        .map(|task| task.spec.predicted_visible_ns)
        .sum::<u64>();
    let actual_visible_ns = tasks.iter().map(|task| task.actual_visible_ns).sum::<u64>();
    MemoryLoopSummary {
        status: MemoryLoopStatus::Ok,
        tasks: tasks.len() as u64,
        queue_capacity,
        max_inflight,
        queue_overflows: u64::from(tasks.len() > queue_capacity),
        disk_read_tasks,
        prefetch_tasks,
        staging_tasks,
        eviction_tasks,
        transport_prepare_tasks,
        total_bytes: tasks.iter().map(|task| task.spec.bytes).sum(),
        total_predicted_visible_ns,
        overlapped_ns: total_predicted_visible_ns.saturating_sub(actual_visible_ns),
        actual_visible_ns,
        ready_blocks,
        prefetch_events: ledger.event_count(LedgerEventKind::Prefetch),
        eviction_events: ledger.event_count(LedgerEventKind::Eviction),
        copy_events: ledger.event_count(LedgerEventKind::Copy),
        transport_events: ledger.event_count(LedgerEventKind::Transport),
        phase_handoff_syncs: ledger.sync_count_for(SyncClass::PhaseHandoff),
        residency_decisions: ledger.residency_decisions.len() as u64,
        pageable_copies: 0,
        per_token_registrations: 0,
        page_faults: 0,
        hot_path_allocations: ledger.hot_path_allocations,
        error: None,
    }
}

fn count_kind(tasks: &[MemoryLoopTask], kind: MemoryLoopTaskKind) -> u64 {
    tasks.iter().filter(|task| task.spec.kind == kind).count() as u64
}
