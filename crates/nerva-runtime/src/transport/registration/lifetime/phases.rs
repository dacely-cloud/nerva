use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::allocation::AllocationId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::transport::registration::cache::TransportRegistrationCache;
use crate::transport::registration::lifetime::bootstrap::LifecycleBlocks;
use crate::transport::registration::lifetime::ledger::{
    record_lookup_hit, record_registration_revocation,
};
use crate::transport::registration::lifetime::revoke::{
    lookup_after_revoke_is_miss, record_revocations, reject_stale_mapping,
};
use crate::transport::registration::types::{
    TransportRegistrationBackend, TransportRegistrationKey, TransportRegistrationLookup,
};

pub(super) fn record_initial_lookup_hit(
    registry: &BlockRegistry,
    cache: &TransportRegistrationCache,
    blocks: LifecycleBlocks,
    ledger: &mut TokenLedger,
) -> Result<u64> {
    let block = registry
        .block(blocks.pinned_send)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "registration lifecycle probe expected pinned send block".to_string(),
        })?;
    match cache.lookup(
        block,
        block.authoritative_copy,
        TransportRegistrationBackend::RdmaPinnedHost,
        0,
    ) {
        TransportRegistrationLookup::Hit(registration) => {
            record_lookup_hit(ledger, registration, "registration_lifecycle_initial_hit");
            Ok(1)
        }
        _ => Err(NervaError::InvalidArgument {
            reason: "registration lifecycle probe expected initial lookup hit".to_string(),
        }),
    }
}

pub(super) fn revoke_pinned_send_rdma_key(
    registry: &BlockRegistry,
    cache: &mut TransportRegistrationCache,
    blocks: LifecycleBlocks,
    ledger: &mut TokenLedger,
) -> Result<u64> {
    let block = registry
        .block(blocks.pinned_send)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "registration lifecycle probe expected pinned send block".to_string(),
        })?;
    let key = TransportRegistrationKey {
        block_id: blocks.pinned_send,
        replica: block.authoritative_copy,
        backend: TransportRegistrationBackend::RdmaPinnedHost,
    };
    let revoked = cache
        .revoke(key)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "registration lifecycle probe expected key revocation".to_string(),
        })?;
    record_registration_revocation(
        ledger,
        revoked,
        "registration_lifecycle_explicit_key_revoke",
    );
    Ok(1)
}

pub(super) fn require_miss_after_key_revoke(
    registry: &BlockRegistry,
    cache: &TransportRegistrationCache,
    blocks: LifecycleBlocks,
    ledger: &mut TokenLedger,
) -> u64 {
    lookup_after_revoke_is_miss(
        registry,
        cache,
        blocks.pinned_send,
        TransportRegistrationBackend::RdmaPinnedHost,
        ledger,
        "registration_lifecycle_key_revoke_miss",
    ) as u64
}

pub(super) fn reject_moved_recv_stale_mapping(
    registry: &mut BlockRegistry,
    cache: &TransportRegistrationCache,
    blocks: LifecycleBlocks,
    ledger: &mut TokenLedger,
) -> Result<u64> {
    registry.move_block(
        blocks.pinned_recv,
        MemoryTier::PinnedDram,
        AllocationId(99),
        4096,
    )?;
    registry.mark_ready(blocks.pinned_recv)?;
    let block = registry
        .block(blocks.pinned_recv)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "registration lifecycle probe expected moved recv block".to_string(),
        })?;
    match cache.lookup(
        block,
        block.authoritative_copy,
        TransportRegistrationBackend::RdmaPinnedHost,
        0,
    ) {
        TransportRegistrationLookup::StaleAddress(registration) => {
            reject_stale_mapping(
                ledger,
                blocks.pinned_recv,
                registration.tier,
                block.tier,
                block.bytes,
                "registration_lifecycle_stale_mapping_rejected",
            );
            Ok(1)
        }
        _ => Err(NervaError::InvalidArgument {
            reason: "registration lifecycle probe expected stale mapping rejection".to_string(),
        }),
    }
}

pub(super) fn cleanup_after_stale_rejection(
    cache: &mut TransportRegistrationCache,
    blocks: LifecycleBlocks,
    ledger: &mut TokenLedger,
) -> u64 {
    record_revocations(
        ledger,
        cache.revoke_block(blocks.pinned_recv),
        "registration_lifecycle_cleanup_after_stale_rejection",
    )
}

pub(super) fn revoke_before_safe_move(
    cache: &mut TransportRegistrationCache,
    blocks: LifecycleBlocks,
    ledger: &mut TokenLedger,
) -> u64 {
    record_revocations(
        ledger,
        cache.revoke_block(blocks.pinned_send),
        "registration_lifecycle_revoke_before_block_move",
    )
}

pub(super) fn require_safe_move_miss(
    registry: &mut BlockRegistry,
    cache: &TransportRegistrationCache,
    blocks: LifecycleBlocks,
    ledger: &mut TokenLedger,
) -> Result<u64> {
    registry.move_block(
        blocks.pinned_send,
        MemoryTier::PinnedDram,
        AllocationId(100),
        8192,
    )?;
    registry.mark_ready(blocks.pinned_send)?;
    Ok(lookup_after_revoke_is_miss(
        registry,
        cache,
        blocks.pinned_send,
        TransportRegistrationBackend::DpdkPinnedHost,
        ledger,
        "registration_lifecycle_safe_move_miss",
    ) as u64)
}

pub(super) fn revoke_before_arena_destroy(
    cache: &mut TransportRegistrationCache,
    ledger: &mut TokenLedger,
) -> u64 {
    record_revocations(
        ledger,
        cache.revoke_all(),
        "registration_lifecycle_revoke_before_arena_destroy",
    )
}
