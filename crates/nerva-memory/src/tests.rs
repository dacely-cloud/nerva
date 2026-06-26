use crate::arena::host::HostArena;
use crate::arena::kind::{AllocationPhase, ArenaKind};
use crate::arena::region::ArenaReservation;
use crate::arena::resident::resident_block_for_reservation;
use crate::arena::set::{StaticArenaBootstrapSpec, StaticArenaSet};
use crate::arena::static_arena::StaticArena;
use crate::kv::page::{KvPageSpec, KvPrefixKey};
use crate::kv::pool::KvPagePool;
use crate::kv::residency::{KvResidencyAction, KvResidencyPlanner, KvResidencyPolicy};
use crate::registry::{BlockAllocationRequest, BlockRegistry};
use nerva_core::types::block::{BlockKind, ResidencyState};
use nerva_core::types::error::NervaError;
use nerva_core::types::id::{AllocationId, MemoryDomainId, ResidentBlockId};
use nerva_core::types::memory::MemoryTier;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::TokenLedger;

#[test]
fn host_arena_respects_alignment() {
    let mut arena = HostArena::new(1024);
    let a = arena.reserve(3, 1).unwrap();
    let b = arena.reserve(8, 64).unwrap();
    assert_eq!(a.offset, 0);
    assert_eq!(b.offset % 64, 0);
    assert!(arena.used() >= b.offset + 8);
}

#[test]
fn registry_tracks_tier_capacity() {
    let mut registry = BlockRegistry::new([(MemoryTier::Dram, 128), (MemoryTier::Vram, 64)]);
    let first = registry
        .allocate(BlockAllocationRequest::new(
            BlockKind::Weight,
            MemoryTier::Dram,
            96,
        ))
        .unwrap();
    assert_eq!(first, ResidentBlockId(1));
    assert_eq!(registry.used_bytes(MemoryTier::Dram), 96);
    assert_eq!(registry.remaining_bytes(MemoryTier::Dram), Some(32));

    let err = registry
        .allocate(BlockAllocationRequest::new(
            BlockKind::Activation,
            MemoryTier::Dram,
            64,
        ))
        .unwrap_err();
    assert!(matches!(err, NervaError::AllocationFailed { .. }));
    assert_eq!(registry.used_bytes(MemoryTier::Dram), 96);
}

#[test]
fn registry_moves_blocks_between_tiers_with_accounting() {
    let mut registry = BlockRegistry::new([(MemoryTier::Dram, 128), (MemoryTier::Vram, 128)]);
    let id = registry
        .allocate(BlockAllocationRequest::new(
            BlockKind::KvPage,
            MemoryTier::Dram,
            64,
        ))
        .unwrap();
    registry.mark_ready(id).unwrap();
    registry
        .move_block(id, MemoryTier::Vram, AllocationId(99), 256)
        .unwrap();

    let block = registry.block(id).unwrap();
    assert_eq!(block.tier, MemoryTier::Vram);
    assert_eq!(block.state, ResidencyState::Prefetching);
    assert_eq!(block.address.domain, MemoryDomainId::GPU_VRAM);
    assert_eq!(block.address.allocation, AllocationId(99));
    assert_eq!(block.address.offset, 256);
    assert_eq!(registry.used_bytes(MemoryTier::Dram), 0);
    assert_eq!(registry.used_bytes(MemoryTier::Vram), 64);
}

#[test]
fn host_reservation_becomes_dram_block_address() {
    let reservation = ArenaReservation {
        offset: 32,
        bytes: 16,
        align: 8,
    };
    let block =
        resident_block_for_reservation(ResidentBlockId(77), BlockKind::Metadata, reservation);
    assert_eq!(block.tier, MemoryTier::Dram);
    assert_eq!(block.address.domain, MemoryDomainId::CPU_DRAM);
    assert_eq!(block.address.offset, 32);
}

#[test]
fn static_arena_reserves_stable_aligned_regions() {
    let mut arena = StaticArena::new(ArenaKind::Device, AllocationId(10), 1024);
    let first = arena
        .reserve("weights", 33, 1, AllocationPhase::Initialization)
        .unwrap();
    let second = arena
        .reserve("workspace", 64, 128, AllocationPhase::Initialization)
        .unwrap();

    assert_eq!(first.address.domain, MemoryDomainId::GPU_VRAM);
    assert_eq!(first.address.allocation, AllocationId(10));
    assert_eq!(first.offset, 0);
    assert_eq!(second.offset % 128, 0);
    assert_eq!(second.address.offset, second.offset as u64);
    assert!(arena.used() >= second.offset + second.bytes);
}

#[test]
fn static_arena_rejects_hot_path_reservation() {
    let mut arena = StaticArena::new(ArenaKind::PinnedHost, AllocationId(22), 1024);
    let err = arena
        .reserve("token-ring", 64, 64, AllocationPhase::HotPath)
        .unwrap_err();
    assert!(matches!(err, NervaError::AllocationFailed { .. }));
    assert_eq!(arena.used(), 0);
}

#[test]
fn static_arena_checkpoint_restore_rewinds_scratch() {
    let mut arena = StaticArena::new(ArenaKind::Host, AllocationId(33), 256);
    let _metadata = arena
        .reserve("metadata", 32, 8, AllocationPhase::Initialization)
        .unwrap();
    let checkpoint = arena.checkpoint();
    let _scratch = arena
        .reserve("scratch", 128, 16, AllocationPhase::Initialization)
        .unwrap();
    assert!(arena.used() > checkpoint.used);
    arena.restore(checkpoint).unwrap();
    assert_eq!(arena.used(), checkpoint.used);
}

#[test]
fn arena_set_reserves_blocks_and_binds_addresses() {
    let mut arenas = StaticArenaSet::new(512, 512, 512);
    let mut registry = BlockRegistry::new([
        (MemoryTier::Vram, 512),
        (MemoryTier::PinnedDram, 512),
        (MemoryTier::Dram, 512),
    ]);

    let id = arenas
        .reserve_resident_block(
            &mut registry,
            ArenaKind::Device,
            "kv-page",
            BlockAllocationRequest::new(BlockKind::KvPage, MemoryTier::Vram, 128),
            128,
            AllocationPhase::Initialization,
        )
        .unwrap();
    let block = registry.block(id).unwrap();
    assert_eq!(block.tier, MemoryTier::Vram);
    assert_eq!(block.address.domain, MemoryDomainId::GPU_VRAM);
    assert_eq!(block.address.allocation, AllocationId(1));
    assert_eq!(block.address.offset, 0);
    assert_eq!(arenas.device().used(), 128);
}

#[test]
fn static_arena_bootstrap_preallocates_cpu_pinned_and_gpu_regions() {
    let mut arenas = StaticArenaSet::new(1024, 1024, 2048);
    let mut registry = BlockRegistry::new([
        (MemoryTier::Vram, 1024),
        (MemoryTier::PinnedDram, 1024),
        (MemoryTier::Dram, 2048),
    ]);

    let bootstrap = arenas
        .preallocate_decode_bootstrap(
            &mut registry,
            StaticArenaBootstrapSpec {
                device_token_state_bytes: 256,
                pinned_observation_bytes: 128,
                host_metadata_bytes: 512,
                align: 64,
            },
        )
        .unwrap();

    assert_eq!(arenas.device().used(), 256);
    assert_eq!(arenas.pinned_host().used(), 128);
    assert_eq!(arenas.host().used(), 512);

    let device = registry.block(bootstrap.device_token_state).unwrap();
    let pinned = registry.block(bootstrap.pinned_observation).unwrap();
    let host = registry.block(bootstrap.host_metadata).unwrap();
    assert_eq!(device.tier, MemoryTier::Vram);
    assert_eq!(device.address.domain, MemoryDomainId::GPU_VRAM);
    assert_eq!(device.state, ResidencyState::Ready);
    assert_eq!(pinned.tier, MemoryTier::PinnedDram);
    assert_eq!(pinned.address.domain, MemoryDomainId::PINNED_DRAM);
    assert_eq!(pinned.state, ResidencyState::Ready);
    assert_eq!(host.tier, MemoryTier::Dram);
    assert_eq!(host.address.domain, MemoryDomainId::CPU_DRAM);
    assert_eq!(host.state, ResidencyState::Ready);
}

#[test]
fn hot_path_arena_attempts_are_rejected_and_ledgered() {
    let mut arenas = StaticArenaSet::new(256, 256, 256);
    let mut ledger = TokenLedger::new(10);

    for kind in [ArenaKind::Device, ArenaKind::PinnedHost, ArenaKind::Host] {
        let before = arenas.arena_mut(kind).used();
        let err = arenas
            .reject_hot_path_reservation_with_ledger(
                kind,
                "forbidden-hot-path-reservation",
                64,
                64,
                &mut ledger,
            )
            .unwrap_err();
        assert!(matches!(err, NervaError::AllocationFailed { .. }));
        assert_eq!(arenas.arena_mut(kind).used(), before);
    }

    assert_eq!(ledger.hot_path_allocations, 3);
    assert_eq!(ledger.events.len(), 3);
    assert_eq!(ledger.event_count(LedgerEventKind::Allocation), 3);
    assert!(ledger.require_zero_hot_path_allocations().is_err());
}

#[test]
fn arena_set_rewinds_if_registry_rejects_block() {
    let mut arenas = StaticArenaSet::new(512, 0, 0);
    let mut registry = BlockRegistry::new([(MemoryTier::Vram, 64)]);

    let err = arenas
        .reserve_resident_block(
            &mut registry,
            ArenaKind::Device,
            "too-large",
            BlockAllocationRequest::new(BlockKind::Activation, MemoryTier::Vram, 128),
            1,
            AllocationPhase::Initialization,
        )
        .unwrap_err();
    assert!(matches!(err, NervaError::AllocationFailed { .. }));
    assert_eq!(arenas.device().used(), 0);
    assert_eq!(registry.used_bytes(MemoryTier::Vram), 0);
}

#[test]
fn kv_page_pool_preallocates_resident_blocks() {
    let mut arenas = StaticArenaSet::new(1024, 0, 0);
    let mut registry = BlockRegistry::new([(MemoryTier::Vram, 1024)]);
    let pool = KvPagePool::preallocate(
        &mut arenas,
        &mut registry,
        4,
        KvPageSpec::new(2, 1, 16, 128, MemoryTier::Vram, ArenaKind::Device, 128),
    )
    .unwrap();

    assert_eq!(pool.len(), 4);
    assert_eq!(pool.num_free_pages(), 4);
    assert_eq!(registry.used_bytes(MemoryTier::Vram), 512);
    assert_eq!(arenas.device().used(), 512);
    assert!(
        pool.page(0)
            .and_then(|page| registry.block(page.block_id))
            .is_some_and(|block| block.state == ResidencyState::Ready)
    );
}

#[test]
fn kv_page_pool_allocates_and_releases_pages() {
    let mut arenas = StaticArenaSet::new(512, 0, 0);
    let mut registry = BlockRegistry::new([(MemoryTier::Vram, 512)]);
    let mut pool = KvPagePool::preallocate(
        &mut arenas,
        &mut registry,
        2,
        KvPageSpec::new(0, 0, 16, 128, MemoryTier::Vram, ArenaKind::Device, 64),
    )
    .unwrap();

    let handle = pool.allocate_page(32, 8, 7).unwrap();
    assert_eq!(pool.num_free_pages(), 1);
    let page = pool.page(handle.page_index).unwrap();
    assert_eq!(page.token_start, 32);
    assert_eq!(page.token_count, 8);
    assert_eq!(page.ref_count, 1);
    assert_eq!(page.last_use, 7);

    pool.release_page(handle.page_index, 8).unwrap();
    assert_eq!(pool.num_free_pages(), 2);
    assert_eq!(pool.page(handle.page_index).unwrap().ref_count, 0);
}

#[test]
fn kv_page_pool_caches_prefix_keys_and_retain_hits() {
    let mut arenas = StaticArenaSet::new(512, 0, 0);
    let mut registry = BlockRegistry::new([(MemoryTier::Vram, 512)]);
    let mut pool = KvPagePool::preallocate(
        &mut arenas,
        &mut registry,
        2,
        KvPageSpec::new(0, 3, 16, 128, MemoryTier::Vram, ArenaKind::Device, 64),
    )
    .unwrap();

    let key = KvPrefixKey {
        hash: [7; 32],
        group_id: 3,
    };
    let handle = pool.allocate_page(0, 16, 1).unwrap();
    pool.cache_page(handle.page_index, key, 16).unwrap();
    assert_eq!(pool.lookup_cached(key), Some(handle));

    pool.release_page(handle.page_index, 2).unwrap();
    assert_eq!(pool.num_free_pages(), 2);
    let retained = pool.retain_cached(key, 3).unwrap().unwrap();
    assert_eq!(retained, handle);
    assert_eq!(pool.num_free_pages(), 1);
    assert_eq!(pool.page(handle.page_index).unwrap().ref_count, 1);

    assert_eq!(
        pool.evict_cached_page(handle.page_index).unwrap(),
        Some(key)
    );
    assert_eq!(pool.lookup_cached(key), None);
}

#[test]
fn kv_residency_planner_prefetches_hot_and_evicts_cold_cached_pages() {
    let mut arenas = StaticArenaSet::new(0, 0, 1024);
    let mut registry = BlockRegistry::new([(MemoryTier::Dram, 1024), (MemoryTier::Vram, 1024)]);
    let mut pool = KvPagePool::preallocate(
        &mut arenas,
        &mut registry,
        3,
        KvPageSpec::new(0, 0, 16, 128, MemoryTier::Dram, ArenaKind::Host, 64),
    )
    .unwrap();

    let hot = pool.allocate_page(0, 16, 10).unwrap();
    pool.set_next_use(hot.page_index, Some(12)).unwrap();

    let cold_key = KvPrefixKey {
        hash: [4; 32],
        group_id: 0,
    };
    let cold = pool.allocate_page(16, 16, 1).unwrap();
    pool.cache_page(cold.page_index, cold_key, 16).unwrap();
    pool.release_page(cold.page_index, 1).unwrap();

    let plan =
        KvResidencyPlanner::plan(&pool, &registry, 20, KvResidencyPolicy::new(1, 4, 8)).unwrap();

    assert_eq!(plan.entries.len(), 2);
    assert!(plan.entries.iter().any(|entry| {
        entry.page_index == hot.page_index
            && entry.action == KvResidencyAction::PrefetchToHot
            && entry.old_tier == MemoryTier::Dram
            && entry.new_tier == MemoryTier::Vram
    }));
    assert!(plan.entries.iter().any(|entry| {
        entry.page_index == cold.page_index
            && entry.action == KvResidencyAction::EvictCold
            && entry.new_tier == MemoryTier::Dram
    }));
}

#[test]
fn kv_residency_plan_applies_tier_changes_to_registry() {
    let mut arenas = StaticArenaSet::new(512, 0, 0);
    let mut registry = BlockRegistry::new([(MemoryTier::Vram, 512), (MemoryTier::Dram, 512)]);
    let mut pool = KvPagePool::preallocate(
        &mut arenas,
        &mut registry,
        1,
        KvPageSpec::new(0, 0, 16, 128, MemoryTier::Vram, ArenaKind::Device, 64),
    )
    .unwrap();
    let key = KvPrefixKey {
        hash: [9; 32],
        group_id: 0,
    };
    let page = pool.allocate_page(0, 16, 1).unwrap();
    pool.cache_page(page.page_index, key, 16).unwrap();
    pool.release_page(page.page_index, 2).unwrap();

    let plan =
        KvResidencyPlanner::plan(&pool, &registry, 10, KvResidencyPolicy::new(0, 0, 100)).unwrap();
    assert_eq!(plan.entries[0].action, KvResidencyAction::DemoteToWarm);
    plan.apply(&mut registry).unwrap();

    let block = registry.block(page.block_id).unwrap();
    assert_eq!(block.tier, MemoryTier::Dram);
    assert_eq!(block.state, ResidencyState::Draining);
}

#[test]
fn kv_residency_plan_records_ledger_decisions() {
    let mut arenas = StaticArenaSet::new(0, 0, 512);
    let mut registry = BlockRegistry::new([(MemoryTier::Dram, 512), (MemoryTier::Vram, 512)]);
    let mut pool = KvPagePool::preallocate(
        &mut arenas,
        &mut registry,
        1,
        KvPageSpec::new(0, 0, 16, 128, MemoryTier::Dram, ArenaKind::Host, 64),
    )
    .unwrap();
    let page = pool.allocate_page(0, 16, 5).unwrap();
    pool.set_next_use(page.page_index, Some(6)).unwrap();

    let plan =
        KvResidencyPlanner::plan(&pool, &registry, 5, KvResidencyPolicy::new(1, 2, 100)).unwrap();
    let mut ledger = TokenLedger::new(5);
    plan.record_to_ledger(&mut ledger);

    assert_eq!(ledger.residency_decisions.len(), 1);
    let decision = &ledger.residency_decisions[0];
    assert_eq!(decision.block_id, page.block_id);
    assert_eq!(decision.old_tier, MemoryTier::Dram);
    assert_eq!(decision.new_tier, MemoryTier::Vram);
    assert_eq!(
        decision.reason,
        "KV page next use is within prefetch window"
    );
    assert_eq!(ledger.event_count(LedgerEventKind::Prefetch), 1);
    assert_eq!(ledger.event_count(LedgerEventKind::Copy), 1);
    assert_eq!(ledger.event_count(LedgerEventKind::Stall), 1);
    assert_eq!(ledger.latency_ns_for(LedgerEventKind::Stall), 228);
}
