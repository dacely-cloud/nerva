use crate::capabilities::snapshot::{CapabilitySnapshot, CapabilityState};
use nerva_core::types::error::Result;
use nerva_core::types::id::allocation::AllocationId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::transport::registration::cache::TransportRegistrationCache;
use crate::transport::registration::probe::bootstrap::{
    bootstrap_registration_cache, record_registered_lookup_hits,
};
use crate::transport::registration::probe::counters::RegistrationProbeCounters;
use crate::transport::registration::probe::fixture::allocate_registration_probe_blocks;
use crate::transport::registration::probe::lookup::record_lookup;
use crate::transport::registration::summary::{
    TransportRegistrationStatus, TransportRegistrationSummary,
};
use crate::transport::registration::types::TransportRegistrationBackend;

pub fn run_transport_registration_probe(
    capabilities: &CapabilitySnapshot,
) -> Result<TransportRegistrationSummary> {
    let mut registry = BlockRegistry::new([
        (MemoryTier::PinnedDram, 8 * 1024 * 1024),
        (MemoryTier::Vram, 8 * 1024 * 1024),
        (MemoryTier::Dram, 8 * 1024 * 1024),
    ]);
    let blocks = allocate_registration_probe_blocks(&mut registry)?;
    let mut cache = TransportRegistrationCache::new(8)?;
    let mut ledger = TokenLedger::new(0);
    let mut counters = RegistrationProbeCounters::new();
    let direct_verified = capabilities.gpu_direct_rdma == CapabilityState::SupportedAndVerified;
    bootstrap_registration_cache(
        &registry,
        &mut cache,
        &mut counters,
        blocks,
        direct_verified,
    )?;
    record_registered_lookup_hits(
        &registry,
        &cache,
        &mut ledger,
        &mut counters,
        blocks,
        direct_verified,
    );

    record_lookup(
        &registry,
        &cache,
        &mut ledger,
        &mut counters,
        blocks.unregistered,
        TransportRegistrationBackend::RdmaPinnedHost,
        0,
        "registration_cache_unregistered_miss",
    );

    registry.move_block(
        blocks.pinned_recv,
        MemoryTier::PinnedDram,
        AllocationId(99),
        4096,
    )?;
    registry.mark_ready(blocks.pinned_recv)?;
    record_lookup(
        &registry,
        &cache,
        &mut ledger,
        &mut counters,
        blocks.pinned_recv,
        TransportRegistrationBackend::RdmaPinnedHost,
        0,
        "registration_cache_moved_block_rejected",
    );

    record_lookup(
        &registry,
        &cache,
        &mut ledger,
        &mut counters,
        blocks.pinned_send,
        TransportRegistrationBackend::RdmaPinnedHost,
        99,
        "registration_cache_stale_version_rejected",
    );

    counters.record_hot_path_registration_rejection();
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Stall,
        sync_class: None,
        metric_source: MetricSource::RuntimeTimestamp,
        block_id: Some(blocks.unregistered),
        from_tier: Some(MemoryTier::PinnedDram),
        to_tier: Some(MemoryTier::PinnedDram),
        bytes: 32 * 1024,
        latency_ns: 1,
        label: "hot_path_transport_registration_rejected",
    });

    ledger.require_zero_hot_path_allocations()?;
    ledger.require_classified_syncs()?;

    let lookup_count = counters.lookup_count();
    let registration_cache_hit_rate_per_mille = if lookup_count == 0 {
        0
    } else {
        counters.cache_hits.saturating_mul(1_000) / lookup_count
    };

    Ok(TransportRegistrationSummary {
        status: TransportRegistrationStatus::Ok,
        cache_capacity: cache.capacity() as u64,
        registered_entries: cache.len() as u64,
        bootstrap_registrations: counters.bootstrap_registrations,
        cache_hits: counters.cache_hits,
        cache_misses: counters.cache_misses,
        stale_address_rejections: counters.stale_address_rejections,
        stale_version_rejections: counters.stale_version_rejections,
        hot_path_registration_attempts: counters.hot_path_registration_attempts,
        hot_path_registration_rejections: counters.hot_path_registration_rejections,
        per_token_registrations: counters.per_token_registrations,
        pinned_host_registrations: counters.pinned_host_registrations,
        gpu_direct_registrations: counters.gpu_direct_registrations,
        gpu_direct_rdma_capability: capabilities.gpu_direct_rdma,
        gpu_direct_registration_skips: counters.gpu_direct_registration_skips,
        false_gpu_direct_registrations: counters.false_gpu_direct_registrations,
        transport_events: ledger.event_count(LedgerEventKind::Transport),
        sync_events: ledger.event_count(LedgerEventKind::Sync),
        phase_handoff_syncs: ledger.sync_count_for(SyncClass::PhaseHandoff),
        registration_cache_hit_rate_per_mille,
        hot_path_allocations: ledger.hot_path_allocations,
        error: None,
    })
}
