use nerva_core::types::error::Result;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::transport::registration::cache::TransportRegistrationCache;
use crate::transport::registration::lifetime::bootstrap::{
    allocate_lifecycle_blocks, register_lifecycle_blocks,
};
use crate::transport::registration::lifetime::counters::LifecycleCounters;
use crate::transport::registration::lifetime::phases::{
    cleanup_after_stale_rejection, record_initial_lookup_hit, reject_moved_recv_stale_mapping,
    require_miss_after_key_revoke, require_safe_move_miss, revoke_before_arena_destroy,
    revoke_before_safe_move, revoke_pinned_send_rdma_key,
};
use crate::transport::registration::lifetime::summary::TransportRegistrationLifecycleSummary;

pub fn run_transport_registration_lifecycle_probe() -> Result<TransportRegistrationLifecycleSummary>
{
    let mut registry = BlockRegistry::new([
        (MemoryTier::PinnedDram, 8 * 1024 * 1024),
        (MemoryTier::Vram, 8 * 1024 * 1024),
    ]);
    let blocks = allocate_lifecycle_blocks(&mut registry)?;
    let mut cache = TransportRegistrationCache::new(8)?;
    let mut ledger = TokenLedger::new(0);
    let bootstrap_registrations = register_lifecycle_blocks(&registry, &mut cache, blocks)?;
    let registered_before_revoke = cache.len() as u64;
    let mut counters = LifecycleCounters::default();

    counters.lookup_hits_before_revoke +=
        record_initial_lookup_hit(&registry, &cache, blocks, &mut ledger)?;
    counters.explicit_key_revocations +=
        revoke_pinned_send_rdma_key(&registry, &mut cache, blocks, &mut ledger)?;
    counters.post_revoke_misses +=
        require_miss_after_key_revoke(&registry, &cache, blocks, &mut ledger);
    counters.stale_mapping_reuse_rejections +=
        reject_moved_recv_stale_mapping(&mut registry, &cache, blocks, &mut ledger)?;
    counters.block_lifetime_revocations +=
        cleanup_after_stale_rejection(&mut cache, blocks, &mut ledger);
    counters.block_lifetime_revocations += revoke_before_safe_move(&mut cache, blocks, &mut ledger);
    counters.safe_move_post_revoke_misses +=
        require_safe_move_miss(&mut registry, &cache, blocks, &mut ledger)?;
    counters.destroy_revocations += revoke_before_arena_destroy(&mut cache, &mut ledger);

    ledger.require_zero_hot_path_allocations()?;
    ledger.require_classified_syncs()?;

    Ok(counters.summary(
        bootstrap_registrations,
        registered_before_revoke,
        cache.len() as u64,
        &ledger,
    ))
}
