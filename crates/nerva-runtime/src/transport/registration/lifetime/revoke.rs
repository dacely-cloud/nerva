use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::transport::registration::cache::TransportRegistrationCache;
use crate::transport::registration::lifetime::ledger::{
    record_lookup_miss, record_registration_revocation,
};
use crate::transport::registration::types::{
    TransportRegistration, TransportRegistrationBackend, TransportRegistrationLookup,
};

pub(super) fn record_revocations(
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

pub(super) fn lookup_after_revoke_is_miss(
    registry: &BlockRegistry,
    cache: &TransportRegistrationCache,
    block_id: ResidentBlockId,
    backend: TransportRegistrationBackend,
    ledger: &mut TokenLedger,
    label: &'static str,
) -> bool {
    let Some(block) = registry.block(block_id) else {
        return false;
    };
    match cache.lookup(block, block.authoritative_copy, backend, 0) {
        TransportRegistrationLookup::Miss => {
            record_lookup_miss(ledger, block.id, block.tier, block.bytes, label);
            true
        }
        _ => false,
    }
}

pub(super) fn reject_stale_mapping(
    ledger: &mut TokenLedger,
    block_id: ResidentBlockId,
    stale_tier: MemoryTier,
    current_tier: MemoryTier,
    bytes: usize,
    label: &'static str,
) {
    crate::transport::registration::lifetime::ledger::record_stale_mapping_rejection(
        ledger,
        block_id,
        stale_tier,
        current_tier,
        bytes,
        label,
    );
}
