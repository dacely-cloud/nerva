use nerva_core::types::block::address::GlobalBlockAddress;
use nerva_core::types::block::taxonomy::BlockKind;
use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;
use nerva_core::types::id::{AllocationId, LayoutId, MemoryDomainId};
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::TokenLedger;
use nerva_memory::registry::{BlockAllocationRequest, BlockRegistry};

use crate::transport::registration::cache::TransportRegistrationCache;
use crate::transport::registration::summary::{
    TransportRegistrationStatus, TransportRegistrationSummary,
};
use crate::transport::registration::types::{
    TransportRegistrationBackend, TransportRegistrationLookup,
};

#[derive(Clone, Debug, Eq, PartialEq)]
struct RegistrationProbeCounters {
    bootstrap_registrations: u64,
    cache_hits: u64,
    cache_misses: u64,
    stale_address_rejections: u64,
    stale_version_rejections: u64,
    hot_path_registration_attempts: u64,
    hot_path_registration_rejections: u64,
    per_token_registrations: u64,
    pinned_host_registrations: u64,
    gpu_direct_registrations: u64,
}

impl RegistrationProbeCounters {
    const fn new() -> Self {
        Self {
            bootstrap_registrations: 0,
            cache_hits: 0,
            cache_misses: 0,
            stale_address_rejections: 0,
            stale_version_rejections: 0,
            hot_path_registration_attempts: 0,
            hot_path_registration_rejections: 0,
            per_token_registrations: 0,
            pinned_host_registrations: 0,
            gpu_direct_registrations: 0,
        }
    }
}

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
        counters.bootstrap_registrations += 1;
        match backend {
            TransportRegistrationBackend::RdmaPinnedHost
            | TransportRegistrationBackend::DpdkPinnedHost => {
                counters.pinned_host_registrations += 1
            }
            TransportRegistrationBackend::RdmaGpuDirect | TransportRegistrationBackend::DpdkGpu => {
                counters.gpu_direct_registrations += 1
            }
        }
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

    counters.hot_path_registration_attempts += 1;
    counters.hot_path_registration_rejections += 1;
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

    let lookup_count = counters.cache_hits
        + counters.cache_misses
        + counters.stale_address_rejections
        + counters.stale_version_rejections;
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

fn allocate_ready_transport_block(
    registry: &mut BlockRegistry,
    tier: MemoryTier,
    dtype: DType,
    bytes: usize,
    allocation: AllocationId,
    offset: u64,
) -> Result<nerva_core::types::id::ResidentBlockId> {
    let id = registry.allocate(
        BlockAllocationRequest::new(BlockKind::TransportBuffer, tier, bytes)
            .with_dtype(dtype)
            .with_layout(LayoutId(1)),
    )?;
    registry.bind_address(
        id,
        GlobalBlockAddress {
            domain: MemoryDomainId::for_tier(tier),
            allocation,
            offset,
        },
    )?;
    {
        let block = registry.block_mut(id).expect("allocated block exists");
        block.owner = ExecutionOwner::Nic(nerva_core::types::id::TransportDeviceId(0));
        block.version = 1;
    }
    registry.mark_ready(id)?;
    Ok(id)
}

fn record_lookup(
    registry: &BlockRegistry,
    cache: &TransportRegistrationCache,
    ledger: &mut TokenLedger,
    counters: &mut RegistrationProbeCounters,
    block_id: nerva_core::types::id::ResidentBlockId,
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
            ledger.record_sync(
                SyncClass::PhaseHandoff,
                Some(block.id),
                Some(registration.tier),
                Some(block.tier),
                block.bytes,
                1,
                MetricSource::RuntimeTimestamp,
                label,
            );
        }
        TransportRegistrationLookup::StaleVersion(registration) => {
            counters.stale_version_rejections += 1;
            ledger.record_sync(
                SyncClass::PhaseHandoff,
                Some(block.id),
                Some(registration.tier),
                Some(block.tier),
                block.bytes,
                1,
                MetricSource::RuntimeTimestamp,
                label,
            );
        }
    }
}
