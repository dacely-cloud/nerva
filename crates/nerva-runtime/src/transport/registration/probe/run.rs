use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;
use nerva_core::types::id::AllocationId;
use nerva_core::types::memory::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::TokenLedger;
use nerva_memory::registry::BlockRegistry;

use crate::transport::registration::cache::TransportRegistrationCache;
use crate::transport::registration::probe::blocks::allocate_ready_transport_block;
use crate::transport::registration::probe::counters::RegistrationProbeCounters;
use crate::transport::registration::probe::lookup::record_lookup;
use crate::transport::registration::summary::{
    TransportRegistrationStatus, TransportRegistrationSummary,
};
use crate::transport::registration::types::TransportRegistrationBackend;

pub fn run_transport_registration_probe() -> Result<TransportRegistrationSummary> {
    let mut registry = BlockRegistry::new([
        (MemoryTier::PinnedDram, 8 * 1024 * 1024),
        (MemoryTier::Vram, 8 * 1024 * 1024),
        (MemoryTier::Dram, 8 * 1024 * 1024),
    ]);
    let pinned_send = allocate_ready_transport_block(
        &mut registry,
        MemoryTier::PinnedDram,
        DType::U8,
        64 * 1024,
        AllocationId(10),
        0,
    )?;
    let pinned_recv = allocate_ready_transport_block(
        &mut registry,
        MemoryTier::PinnedDram,
        DType::U8,
        64 * 1024,
        AllocationId(11),
        0,
    )?;
    let gpu_direct = allocate_ready_transport_block(
        &mut registry,
        MemoryTier::Vram,
        DType::U8,
        64 * 1024,
        AllocationId(12),
        0,
    )?;
    let unregistered = allocate_ready_transport_block(
        &mut registry,
        MemoryTier::PinnedDram,
        DType::U8,
        32 * 1024,
        AllocationId(13),
        0,
    )?;

    let mut cache = TransportRegistrationCache::new(8)?;
    let mut ledger = TokenLedger::new(0);
    let mut counters = RegistrationProbeCounters::new();

    for (id, backend) in [
        (pinned_send, TransportRegistrationBackend::RdmaPinnedHost),
        (pinned_recv, TransportRegistrationBackend::RdmaPinnedHost),
        (pinned_send, TransportRegistrationBackend::DpdkPinnedHost),
        (gpu_direct, TransportRegistrationBackend::RdmaGpuDirect),
    ] {
        let block = registry.block(id).expect("probe block exists");
        cache.register(block, block.authoritative_copy, backend)?;
        counters.record_bootstrap_registration(backend);
    }

    for (id, backend, label) in [
        (
            pinned_send,
            TransportRegistrationBackend::RdmaPinnedHost,
            "registration_cache_rdma_send_hit",
        ),
        (
            pinned_recv,
            TransportRegistrationBackend::RdmaPinnedHost,
            "registration_cache_rdma_recv_hit",
        ),
        (
            pinned_send,
            TransportRegistrationBackend::DpdkPinnedHost,
            "registration_cache_dpdk_send_hit",
        ),
        (
            gpu_direct,
            TransportRegistrationBackend::RdmaGpuDirect,
            "registration_cache_gpu_direct_hit",
        ),
    ] {
        record_lookup(
            &registry,
            &cache,
            &mut ledger,
            &mut counters,
            id,
            backend,
            0,
            label,
        );
    }

    record_lookup(
        &registry,
        &cache,
        &mut ledger,
        &mut counters,
        unregistered,
        TransportRegistrationBackend::RdmaPinnedHost,
        0,
        "registration_cache_unregistered_miss",
    );

    registry.move_block(pinned_recv, MemoryTier::PinnedDram, AllocationId(99), 4096)?;
    registry.mark_ready(pinned_recv)?;
    record_lookup(
        &registry,
        &cache,
        &mut ledger,
        &mut counters,
        pinned_recv,
        TransportRegistrationBackend::RdmaPinnedHost,
        0,
        "registration_cache_moved_block_rejected",
    );

    record_lookup(
        &registry,
        &cache,
        &mut ledger,
        &mut counters,
        pinned_send,
        TransportRegistrationBackend::RdmaPinnedHost,
        99,
        "registration_cache_stale_version_rejected",
    );

    counters.record_hot_path_registration_rejection();
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Stall,
        sync_class: None,
        metric_source: MetricSource::RuntimeTimestamp,
        block_id: Some(unregistered),
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
        transport_events: ledger.event_count(LedgerEventKind::Transport),
        sync_events: ledger.event_count(LedgerEventKind::Sync),
        phase_handoff_syncs: ledger.sync_count_for(SyncClass::PhaseHandoff),
        registration_cache_hit_rate_per_mille,
        hot_path_allocations: ledger.hot_path_allocations,
        error: None,
    })
}
