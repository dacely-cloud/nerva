use nerva_core::types::block::address::GlobalBlockAddress;
use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::block::resident::ResidentBlock;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::allocation::AllocationId;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::memory::MemoryDomainId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::shape::BlockShape;

use crate::registry::domain::memory_tier_for_domain;
use crate::registry::request::BlockAllocationRequest;
use crate::registry::table::registry::BlockRegistry;

impl BlockRegistry {
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
            BlockShape::scalar(),
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

    pub(crate) fn require_block(&self, id: ResidentBlockId) -> Result<&ResidentBlock> {
        self.blocks
            .get(&id)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("unknown resident block id {}", id.0),
            })
    }

    pub(crate) fn require_block_mut(&mut self, id: ResidentBlockId) -> Result<&mut ResidentBlock> {
        self.blocks
            .get_mut(&id)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("unknown resident block id {}", id.0),
            })
    }
}
