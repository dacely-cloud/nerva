use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::{AllocationId, ResidentBlockId};
use nerva_core::types::memory::MemoryTier;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::registry::table::BlockRegistry;

use crate::transport::registration::cache::TransportRegistrationCache;
use crate::transport::registration::lifetime::ledger::{
    record_lookup_hit, record_lookup_miss, record_registration_revocation,
    record_stale_mapping_rejection,
};
use crate::transport::registration::lifetime::summary::{
    TransportRegistrationLifecycleStatus, TransportRegistrationLifecycleSummary,
};
use crate::transport::registration::probe::blocks::allocate_ready_transport_block;
use crate::transport::registration::types::{
    TransportRegistration, TransportRegistrationBackend, TransportRegistrationKey,
    TransportRegistrationLookup,
};

pub fn run_transport_registration_lifecycle_probe() -> Result<TransportRegistrationLifecycleSummary>
{
    let mut registry = BlockRegistry::new([
        (MemoryTier::PinnedDram, 8 * 1024 * 1024),
        (MemoryTier::Vram, 8 * 1024 * 1024),
    ]);
    let pinned_send = allocate_ready_transport_block(
        &mut registry,
        MemoryTier::PinnedDram,
        DType::U8,
        64 * 1024,
        AllocationId(20),
        0,
    )?;
    let pinned_recv = allocate_ready_transport_block(
        &mut registry,
        MemoryTier::PinnedDram,
        DType::U8,
        64 * 1024,
        AllocationId(21),
        0,
    )?;
    let gpu_direct = allocate_ready_transport_block(
        &mut registry,
        MemoryTier::Vram,
        DType::U8,
        64 * 1024,
        AllocationId(22),
        0,
    )?;

    let mut cache = TransportRegistrationCache::new(8)?;
    let mut ledger = TokenLedger::new(0);
    let mut bootstrap_registrations = 0u64;
    let mut explicit_key_revocations = 0u64;
    let mut block_lifetime_revocations = 0u64;
    let mut lookup_hits_before_revoke = 0u64;
    let mut post_revoke_misses = 0u64;
    let mut safe_move_post_revoke_misses = 0u64;
    let mut stale_mapping_reuse_rejections = 0u64;

    for (id, backend) in [
        (pinned_send, TransportRegistrationBackend::RdmaPinnedHost),
        (pinned_send, TransportRegistrationBackend::DpdkPinnedHost),
        (pinned_recv, TransportRegistrationBackend::RdmaPinnedHost),
        (gpu_direct, TransportRegistrationBackend::RdmaGpuDirect),
    ] {
        let block = registry.block(id).expect("probe block exists");
        cache.register(block, block.authoritative_copy, backend)?;
        bootstrap_registrations += 1;
    }

    let registered_before_revoke = cache.len() as u64;
    let pinned_send_block = registry.block(pinned_send).expect("probe block exists");
    let pinned_send_rdma_key = TransportRegistrationKey {
        block_id: pinned_send,
        replica: pinned_send_block.authoritative_copy,
        backend: TransportRegistrationBackend::RdmaPinnedHost,
    };

    match cache.lookup(
        pinned_send_block,
        pinned_send_block.authoritative_copy,
        TransportRegistrationBackend::RdmaPinnedHost,
        0,
    ) {
        TransportRegistrationLookup::Hit(registration) => {
            lookup_hits_before_revoke += 1;
            record_lookup_hit(
                &mut ledger,
                registration,
                "registration_lifecycle_initial_hit",
            );
        }
        _ => {
            return Err(NervaError::InvalidArgument {
                reason: "registration lifecycle probe expected initial lookup hit".to_string(),
            });
        }
    }

    let revoked =
        cache
            .revoke(pinned_send_rdma_key)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "registration lifecycle probe expected key revocation".to_string(),
            })?;
    explicit_key_revocations += 1;
    record_registration_revocation(
        &mut ledger,
        revoked,
        "registration_lifecycle_explicit_key_revoke",
    );
    if lookup_after_revoke_is_miss(
        &registry,
        &cache,
        pinned_send,
        TransportRegistrationBackend::RdmaPinnedHost,
        &mut ledger,
        "registration_lifecycle_key_revoke_miss",
    ) {
        post_revoke_misses += 1;
    }

    registry.move_block(pinned_recv, MemoryTier::PinnedDram, AllocationId(99), 4096)?;
    registry.mark_ready(pinned_recv)?;
    let pinned_recv_block = registry.block(pinned_recv).expect("probe block exists");
    if let TransportRegistrationLookup::StaleAddress(registration) = cache.lookup(
        pinned_recv_block,
        pinned_recv_block.authoritative_copy,
        TransportRegistrationBackend::RdmaPinnedHost,
        0,
    ) {
        stale_mapping_reuse_rejections += 1;
        record_stale_mapping_rejection(
            &mut ledger,
            pinned_recv,
            registration.tier,
            pinned_recv_block.tier,
            pinned_recv_block.bytes,
            "registration_lifecycle_stale_mapping_rejected",
        );
    } else {
        return Err(NervaError::InvalidArgument {
            reason: "registration lifecycle probe expected stale mapping rejection".to_string(),
        });
    }

    block_lifetime_revocations += record_revocations(
        &mut ledger,
        cache.revoke_block(pinned_recv),
        "registration_lifecycle_cleanup_after_stale_rejection",
    );
    block_lifetime_revocations += record_revocations(
        &mut ledger,
        cache.revoke_block(pinned_send),
        "registration_lifecycle_revoke_before_block_move",
    );
    registry.move_block(pinned_send, MemoryTier::PinnedDram, AllocationId(100), 8192)?;
    registry.mark_ready(pinned_send)?;
    if lookup_after_revoke_is_miss(
        &registry,
        &cache,
        pinned_send,
        TransportRegistrationBackend::DpdkPinnedHost,
        &mut ledger,
        "registration_lifecycle_safe_move_miss",
    ) {
        safe_move_post_revoke_misses += 1;
    }

    let destroy_revocations = record_revocations(
        &mut ledger,
        cache.revoke_all(),
        "registration_lifecycle_revoke_before_arena_destroy",
    );

    ledger.require_zero_hot_path_allocations()?;
    ledger.require_classified_syncs()?;

    let total_revocations =
        explicit_key_revocations + block_lifetime_revocations + destroy_revocations;
    Ok(TransportRegistrationLifecycleSummary {
        status: TransportRegistrationLifecycleStatus::Ok,
        bootstrap_registrations,
        registered_before_revoke,
        explicit_key_revocations,
        block_lifetime_revocations,
        destroy_revocations,
        total_revocations,
        final_registered_entries: cache.len() as u64,
        lookup_hits_before_revoke,
        post_revoke_misses,
        safe_move_post_revoke_misses,
        stale_mapping_reuse_rejections,
        revocation_syncs: total_revocations,
        phase_handoff_syncs: ledger.sync_count_for(SyncClass::PhaseHandoff),
        transport_events: ledger.event_count(LedgerEventKind::Transport),
        stale_handle_reuse_prevented: post_revoke_misses > 0
            && safe_move_post_revoke_misses > 0
            && stale_mapping_reuse_rejections > 0,
        hot_path_allocations: ledger.hot_path_allocations,
        error: None,
    })
}

fn record_revocations(
    ledger: &mut TokenLedger,
    registrations: Vec<TransportRegistration>,
    label: &'static str,
) -> u64 {
    let count = registrations.len() as u64;
    for registration in registrations {
        record_registration_revocation(ledger, registration, label);
    }
    count
}

fn lookup_after_revoke_is_miss(
    registry: &BlockRegistry,
    cache: &TransportRegistrationCache,
    block_id: ResidentBlockId,
    backend: TransportRegistrationBackend,
    ledger: &mut TokenLedger,
    label: &'static str,
) -> bool {
    let block = registry.block(block_id).expect("probe block exists");
    match cache.lookup(block, block.authoritative_copy, backend, 0) {
        TransportRegistrationLookup::Miss => {
            record_lookup_miss(ledger, block.id, block.tier, block.bytes, label);
            true
        }
        _ => false,
    }
}
