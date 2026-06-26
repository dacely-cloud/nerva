use crate::registry::{BlockAllocationRequest, BlockRegistry};
use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::block::taxonomy::BlockKind;
use nerva_core::types::error::NervaError;
use nerva_core::types::id::{AllocationId, MemoryDomainId, ResidentBlockId};
use nerva_core::types::memory::MemoryTier;

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
