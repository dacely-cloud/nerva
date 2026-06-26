use std::collections::BTreeMap;

use nerva_core::{
    AllocationId, BlockKind, DType, GlobalBlockAddress, LayoutId, MemoryDomainId, MemoryTier,
    NervaError, ResidencyState, ResidentBlock, ResidentBlockId, Result,
};

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

    pub fn bind_address(&mut self, id: ResidentBlockId, address: GlobalBlockAddress) -> Result<()> {
        let block = self.require_block_mut(id)?;
        let address_tier =
            memory_tier_for_domain(address.domain).ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("unknown memory domain {}", address.domain.0),
            })?;
        if block.tier != address_tier {
            return Err(NervaError::InvalidArgument {
                reason: "block tier and arena address domain do not match".to_string(),
            });
        }
        block.address = address;
        block.memory_domain = address.domain;
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

fn memory_tier_for_domain(domain: MemoryDomainId) -> Option<MemoryTier> {
    Some(match domain {
        MemoryDomainId::GPU_VRAM => MemoryTier::Vram,
        MemoryDomainId::PINNED_DRAM => MemoryTier::PinnedDram,
        MemoryDomainId::CPU_DRAM => MemoryTier::Dram,
        MemoryDomainId::SHARED_HBM_OR_LPDDR => MemoryTier::SharedHbmOrLpddr,
        MemoryDomainId::CXL => MemoryTier::Cxl,
        MemoryDomainId::DISK => MemoryTier::Disk,
        _ => return None,
    })
}
