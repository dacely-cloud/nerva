use nerva_core::types::block::address::GlobalBlockAddress;
use nerva_core::types::block::taxonomy::BlockKind;
use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;
use nerva_core::types::id::{
    AllocationId, LayoutId, MemoryDomainId, ResidentBlockId, TransportDeviceId,
};
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_memory::registry::request::BlockAllocationRequest;
use nerva_memory::registry::table::BlockRegistry;

pub(super) fn allocate_ready_contract_buffer(
    registry: &mut BlockRegistry,
    allocation: AllocationId,
    offset: u64,
) -> Result<ResidentBlockId> {
    let id = registry.allocate(
        BlockAllocationRequest::new(
            BlockKind::TransportBuffer,
            MemoryTier::PinnedDram,
            64 * 1024,
        )
        .with_dtype(DType::U8)
        .with_layout(LayoutId(2)),
    )?;
    registry.bind_address(
        id,
        GlobalBlockAddress {
            domain: MemoryDomainId::PINNED_DRAM,
            allocation,
            offset,
        },
    )?;
    {
        let block = registry.block_mut(id).expect("allocated block exists");
        block.owner = ExecutionOwner::Nic(TransportDeviceId(0));
        block.version = 1;
    }
    registry.mark_ready(id)?;
    Ok(id)
}
