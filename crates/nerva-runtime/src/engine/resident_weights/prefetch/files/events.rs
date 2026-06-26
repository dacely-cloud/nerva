use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::weights::prefetch::ResidentWeightPrefetchTask;

pub(super) fn record_file_read(ledger: &mut TokenLedger, task: &ResidentWeightPrefetchTask) {
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Prefetch,
        sync_class: None,
        metric_source: MetricSource::RuntimeTimestamp,
        block_id: Some(task.block_id),
        from_tier: Some(MemoryTier::Disk),
        to_tier: Some(MemoryTier::PinnedDram),
        bytes: task.bytes,
        latency_ns: 0,
        label: "weight_prefetch_file_read",
    });
}

pub(super) fn record_file_commit(ledger: &mut TokenLedger, task: &ResidentWeightPrefetchTask) {
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Copy,
        sync_class: None,
        metric_source: MetricSource::RuntimeTimestamp,
        block_id: Some(task.block_id),
        from_tier: Some(MemoryTier::PinnedDram),
        to_tier: Some(task.target_tier),
        bytes: task.bytes,
        latency_ns: 0,
        label: "weight_prefetch_file_commit",
    });
}
