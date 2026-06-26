use nerva_core::types::block::address::GlobalBlockAddress;
use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;
use nerva_core::types::id::allocation::AllocationId;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::layout::LayoutId;
use nerva_core::types::id::memory::MemoryDomainId;

use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_memory::registry::request::BlockAllocationRequest;
use nerva_memory::registry::table::registry::BlockRegistry;

pub(crate) fn allocate_ready_transport_block(
    registry: &mut BlockRegistry,
    tier: MemoryTier,
    dtype: DType,
    bytes: usize,
    allocation: AllocationId,
    offset: u64,
) -> Result<ResidentBlockId> {
    let id = registry.allocate(
        BlockAllocationRequest::new(BlockKind::TransportBuffer, tier, bytes)
            .with_dtype(dtype)
            .with_layout(LayoutId(1)),
    )?;
    registry.bind_address(
        id,
        GlobalBlockAddress {
            domain: MemoryDomainId::for_tier(tier),
            allocation,
            offset,
        },
    )?;
    {
        let block = registry.block_mut(id).expect("allocated block exists");
        block.owner = ExecutionOwner::Nic(nerva_core::types::id::transport::TransportDeviceId(0));
        block.version = 1;
    }
    registry.mark_ready(id)?;
    Ok(id)
}
