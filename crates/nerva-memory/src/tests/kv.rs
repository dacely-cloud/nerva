use crate::arena::kind::ArenaKind;
use crate::arena::set::StaticArenaSet;
use crate::kv::page::{KvPageSpec, KvPrefixKey};
use crate::kv::pool::table::KvPagePool;
use crate::kv::residency::types::{KvResidencyAction, KvResidencyPlanner, KvResidencyPolicy};
use crate::registry::table::BlockRegistry;
use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::memory::MemoryTier;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::ledger::TokenLedger;

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
