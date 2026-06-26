use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::transport::registration::types::TransportRegistration;

pub(crate) fn record_registration_revocation(
    ledger: &mut TokenLedger,
    registration: TransportRegistration,
    label: &'static str,
) {
    ledger.record_sync(
        SyncClass::PhaseHandoff,
        Some(registration.key.block_id),
        Some(registration.tier),
        Some(registration.tier),
        registration.bytes,
        1,
        MetricSource::RuntimeTimestamp,
        label,
    );
}

pub(crate) fn record_lookup_hit(
    ledger: &mut TokenLedger,
    registration: TransportRegistration,
    label: &'static str,
) {
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Transport,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(registration.key.block_id),
        from_tier: Some(registration.tier),
        to_tier: Some(registration.tier),
        bytes: registration.bytes,
        latency_ns: 1,
        label,
    });
}

pub(crate) fn record_lookup_miss(
    ledger: &mut TokenLedger,
    block_id: ResidentBlockId,
    tier: MemoryTier,
    bytes: usize,
    label: &'static str,
) {
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Stall,
        sync_class: None,
        metric_source: MetricSource::RuntimeTimestamp,
        block_id: Some(block_id),
        from_tier: Some(tier),
        to_tier: Some(tier),
        bytes,
        latency_ns: 1,
        label,
    });
}

pub(crate) fn record_stale_mapping_rejection(
    ledger: &mut TokenLedger,
    block_id: ResidentBlockId,
    registered_tier: MemoryTier,
    current_tier: MemoryTier,
    bytes: usize,
    label: &'static str,
) {
    ledger.record_sync(
        SyncClass::PhaseHandoff,
        Some(block_id),
        Some(registered_tier),
        Some(current_tier),
        bytes,
        1,
        MetricSource::RuntimeTimestamp,
        label,
    );
}
