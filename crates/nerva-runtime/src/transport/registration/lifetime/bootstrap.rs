use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::allocation::AllocationId;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::transport::registration::cache::TransportRegistrationCache;
use crate::transport::registration::probe::blocks::allocate_ready_transport_block;
use crate::transport::registration::types::TransportRegistrationBackend;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) struct LifecycleBlocks {
    pub(super) pinned_send: ResidentBlockId,
    pub(super) pinned_recv: ResidentBlockId,
    pub(super) gpu_direct: ResidentBlockId,
}

pub(super) fn allocate_lifecycle_blocks(registry: &mut BlockRegistry) -> Result<LifecycleBlocks> {
    let pinned_send = allocate_ready_transport_block(
        registry,
        MemoryTier::PinnedDram,
        DType::U8,
        64 * 1024,
        AllocationId(20),
        0,
    )?;
    let pinned_recv = allocate_ready_transport_block(
        registry,
        MemoryTier::PinnedDram,
        DType::U8,
        64 * 1024,
        AllocationId(21),
        0,
    )?;
    let gpu_direct = allocate_ready_transport_block(
        registry,
        MemoryTier::Vram,
        DType::U8,
        64 * 1024,
        AllocationId(22),
        0,
    )?;

    Ok(LifecycleBlocks {
        pinned_send,
        pinned_recv,
        gpu_direct,
    })
}

pub(super) fn register_lifecycle_blocks(
    registry: &BlockRegistry,
    cache: &mut TransportRegistrationCache,
    blocks: LifecycleBlocks,
) -> Result<u64> {
    let registrations = [
        (
            blocks.pinned_send,
            TransportRegistrationBackend::RdmaPinnedHost,
        ),
        (
            blocks.pinned_send,
            TransportRegistrationBackend::DpdkPinnedHost,
        ),
        (
            blocks.pinned_recv,
            TransportRegistrationBackend::RdmaPinnedHost,
        ),
        (
            blocks.gpu_direct,
            TransportRegistrationBackend::RdmaGpuDirect,
        ),
    ];
    for (id, backend) in registrations {
        let block = registry
            .block(id)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "registration lifecycle bootstrap references missing block".to_string(),
            })?;
        cache.register(block, block.authoritative_copy, backend)?;
    }
    Ok(registrations.len() as u64)
}
