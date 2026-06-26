#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use std::collections::BTreeMap;

use nerva_core::{
    AllocationId, BlockKind, DType, GlobalBlockAddress, LayoutId, MemoryDomainId, MemoryTier,
    NervaError, ResidencyState, ResidentBlock, ResidentBlockId, ResidentBlockKind, Result,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArenaKind {
    Device,
    PinnedHost,
    Host,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArenaReservation {
    pub offset: usize,
    pub bytes: usize,
    pub align: usize,
}

#[derive(Clone, Debug)]
pub struct HostArena {
    bytes: Vec<u8>,
    used: usize,
}

impl HostArena {
    pub fn new(capacity: usize) -> Self {
        Self {
            bytes: vec![0; capacity],
            used: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.bytes.len()
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.used
    }

    pub fn reserve(&mut self, bytes: usize, align: usize) -> Result<ArenaReservation> {
        let align = align.max(1);
        let offset = self.used.next_multiple_of(align);
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes,
                reason: "arena offset overflow".to_string(),
            })?;
        if end > self.bytes.len() {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: "host arena exhausted".to_string(),
            });
        }
        self.used = end;
        Ok(ArenaReservation {
            offset,
            bytes,
            align,
        })
    }

    pub fn reset(&mut self) {
        self.used = 0;
    }
}

pub fn resident_block_for_reservation(
    id: ResidentBlockId,
    kind: ResidentBlockKind,
    reservation: ArenaReservation,
) -> ResidentBlock {
    ResidentBlock::new(id, kind, MemoryTier::Dram, reservation.bytes).with_address(
        GlobalBlockAddress {
            domain: MemoryDomainId::CPU_DRAM,
            allocation: AllocationId(id.0),
            offset: reservation.offset as u64,
        },
    )
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TierAccount {
    pub tier: MemoryTier,
    pub capacity_bytes: usize,
    pub used_bytes: usize,
}

impl TierAccount {
    pub const fn remaining_bytes(self) -> usize {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockAllocationRequest {
    pub kind: BlockKind,
    pub tier: MemoryTier,
    pub bytes: usize,
    pub dtype: DType,
    pub layout: LayoutId,
}

impl BlockAllocationRequest {
    pub const fn new(kind: BlockKind, tier: MemoryTier, bytes: usize) -> Self {
        Self {
            kind,
            tier,
            bytes,
            dtype: DType::U8,
            layout: LayoutId(0),
        }
    }

    pub const fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub const fn with_layout(mut self, layout: LayoutId) -> Self {
        self.layout = layout;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockRegistry {
    next_id: u64,
    accounts: BTreeMap<MemoryTier, TierAccount>,
    blocks: BTreeMap<ResidentBlockId, ResidentBlock>,
}

impl BlockRegistry {
    pub fn new(accounts: impl IntoIterator<Item = (MemoryTier, usize)>) -> Self {
        let mut registry = Self {
            next_id: 1,
            accounts: BTreeMap::new(),
            blocks: BTreeMap::new(),
        };
        for (tier, capacity_bytes) in accounts {
            registry.accounts.insert(
                tier,
                TierAccount {
                    tier,
                    capacity_bytes,
                    used_bytes: 0,
                },
            );
        }
        registry
    }

    pub fn account(&self, tier: MemoryTier) -> Option<TierAccount> {
        self.accounts.get(&tier).copied()
    }

    pub fn used_bytes(&self, tier: MemoryTier) -> usize {
        self.account(tier).map_or(0, |account| account.used_bytes)
    }

    pub fn remaining_bytes(&self, tier: MemoryTier) -> Option<usize> {
        self.account(tier).map(|account| account.remaining_bytes())
    }

    pub fn block(&self, id: ResidentBlockId) -> Option<&ResidentBlock> {
        self.blocks.get(&id)
    }

    pub fn block_mut(&mut self, id: ResidentBlockId) -> Option<&mut ResidentBlock> {
        self.blocks.get_mut(&id)
    }

    pub fn allocate(&mut self, request: BlockAllocationRequest) -> Result<ResidentBlockId> {
        self.reserve_tier(request.tier, request.bytes)?;
        let id = ResidentBlockId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: request.bytes,
                reason: "resident block id overflow".to_string(),
            })?;

        let block = ResidentBlock::new(id, request.kind, request.tier, request.bytes).with_shape(
            request.dtype,
            nerva_core::BlockShape::scalar(),
            request.layout,
        );
        self.blocks.insert(id, block);
        Ok(id)
    }

    pub fn mark_ready(&mut self, id: ResidentBlockId) -> Result<()> {
        let block = self.require_block_mut(id)?;
        block.mark_ready();
        Ok(())
    }

    pub fn transition(&mut self, id: ResidentBlockId, state: ResidencyState) -> Result<()> {
        let block = self.require_block_mut(id)?;
        block.state = state;
        Ok(())
    }

    pub fn move_block(
        &mut self,
        id: ResidentBlockId,
        to_tier: MemoryTier,
        allocation: AllocationId,
        offset: u64,
    ) -> Result<()> {
        let (from_tier, bytes) = {
            let block = self.require_block(id)?;
            (block.tier, block.bytes)
        };

        if from_tier == to_tier {
            let block = self.require_block_mut(id)?;
            block.address = GlobalBlockAddress {
                domain: MemoryDomainId::for_tier(to_tier),
                allocation,
                offset,
            };
            block.memory_domain = MemoryDomainId::for_tier(to_tier);
            return Ok(());
        }

        self.reserve_tier(to_tier, bytes)?;
        self.release_tier(from_tier, bytes);

        let block = self.require_block_mut(id)?;
        block.tier = to_tier;
        block.address = GlobalBlockAddress {
            domain: MemoryDomainId::for_tier(to_tier),
            allocation,
            offset,
        };
        block.memory_domain = MemoryDomainId::for_tier(to_tier);
        block.state = ResidencyState::Prefetching;
        Ok(())
    }

    fn require_block(&self, id: ResidentBlockId) -> Result<&ResidentBlock> {
        self.blocks
            .get(&id)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("unknown resident block id {}", id.0),
            })
    }

    fn require_block_mut(&mut self, id: ResidentBlockId) -> Result<&mut ResidentBlock> {
        self.blocks
            .get_mut(&id)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("unknown resident block id {}", id.0),
            })
    }

    fn reserve_tier(&mut self, tier: MemoryTier, bytes: usize) -> Result<()> {
        let account = self
            .accounts
            .get_mut(&tier)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("memory tier {tier:?} is not configured"),
            })?;
        let new_used =
            account
                .used_bytes
                .checked_add(bytes)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes,
                    reason: "tier accounting overflow".to_string(),
                })?;
        if new_used > account.capacity_bytes {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: format!("memory tier {tier:?} exhausted"),
            });
        }
        account.used_bytes = new_used;
        Ok(())
    }

    fn release_tier(&mut self, tier: MemoryTier, bytes: usize) {
        if let Some(account) = self.accounts.get_mut(&tier) {
            account.used_bytes = account.used_bytes.saturating_sub(bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
