use nerva_core::types::memory::tier::MemoryTier;

use crate::types::event::{LedgerEvent, LedgerEventKind};
use crate::types::metric::MetricSource;
use crate::types::sync::SyncClass;
use crate::types::token::ledger::TokenLedger;

#[test]
fn classified_sync_validation_rejects_missing_or_misplaced_classes() {
    let mut missing = TokenLedger::new(0);
    missing.record(LedgerEvent {
        kind: LedgerEventKind::Sync,
        sync_class: None,
        metric_source: MetricSource::RuntimeTimestamp,
        block_id: None,
        from_tier: None,
        to_tier: None,
        bytes: 0,
        latency_ns: 1,
        label: "unclassified_wait",
    });
    assert!(missing.require_classified_syncs().is_err());

    let mut misplaced = TokenLedger::new(1);
    misplaced.record(LedgerEvent {
        kind: LedgerEventKind::Copy,
        sync_class: Some(SyncClass::HardSync),
        metric_source: MetricSource::RuntimeTimestamp,
        block_id: None,
        from_tier: Some(MemoryTier::Dram),
        to_tier: Some(MemoryTier::Vram),
        bytes: 4,
        latency_ns: 1,
        label: "copy_with_sync_class",
    });
    assert!(misplaced.require_classified_syncs().is_err());
}
