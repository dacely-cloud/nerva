use nerva_core::types::id::ResidentBlockId;
use nerva_core::types::memory::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::registry::table::BlockRegistry;

use crate::transport::registration::cache::TransportRegistrationCache;
use crate::transport::registration::probe::counters::RegistrationProbeCounters;
use crate::transport::registration::types::{
    TransportRegistrationBackend, TransportRegistrationLookup,
};

pub(crate) fn record_lookup(
    registry: &BlockRegistry,
    cache: &TransportRegistrationCache,
    ledger: &mut TokenLedger,
    counters: &mut RegistrationProbeCounters,
    block_id: ResidentBlockId,
    backend: TransportRegistrationBackend,
    required_version: u64,
    label: &'static str,
) {
    let block = registry.block(block_id).expect("probe block exists");
    match cache.lookup(block, block.authoritative_copy, backend, required_version) {
        TransportRegistrationLookup::Hit(registration) => {
            counters.cache_hits += 1;
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::Transport,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: Some(block.id),
                from_tier: Some(registration.tier),
                to_tier: Some(registration.tier),
                bytes: registration.bytes,
                latency_ns: 1,
                label,
            });
        }
        TransportRegistrationLookup::Miss => {
            counters.cache_misses += 1;
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::Stall,
                sync_class: None,
                metric_source: MetricSource::RuntimeTimestamp,
                block_id: Some(block.id),
                from_tier: Some(block.tier),
                to_tier: Some(block.tier),
                bytes: block.bytes,
                latency_ns: 1,
                label,
            });
        }
        TransportRegistrationLookup::StaleAddress(registration) => {
            counters.stale_address_rejections += 1;
            record_phase_handoff_rejection(
                ledger,
                block.id,
                registration.tier,
                block.tier,
                block.bytes,
                label,
            );
        }
        TransportRegistrationLookup::StaleVersion(registration) => {
            counters.stale_version_rejections += 1;
            record_phase_handoff_rejection(
                ledger,
                block.id,
                registration.tier,
                block.tier,
                block.bytes,
                label,
            );
        }
    }
}

fn record_phase_handoff_rejection(
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
