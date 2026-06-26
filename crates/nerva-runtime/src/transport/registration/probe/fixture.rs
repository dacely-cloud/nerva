use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;
use nerva_core::types::id::allocation::AllocationId;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::transport::registration::probe::blocks::allocate_ready_transport_block;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct RegistrationProbeBlocks {
    pub pinned_send: ResidentBlockId,
    pub pinned_recv: ResidentBlockId,
    pub gpu_direct: ResidentBlockId,
    pub unregistered: ResidentBlockId,
}

pub(crate) fn allocate_registration_probe_blocks(
    registry: &mut BlockRegistry,
) -> Result<RegistrationProbeBlocks> {
    Ok(RegistrationProbeBlocks {
        pinned_send: allocate_ready_transport_block(
            registry,
            MemoryTier::PinnedDram,
            DType::U8,
            64 * 1024,
            AllocationId(10),
            0,
        )?,
        pinned_recv: allocate_ready_transport_block(
            registry,
            MemoryTier::PinnedDram,
            DType::U8,
            64 * 1024,
            AllocationId(11),
            0,
        )?,
        gpu_direct: allocate_ready_transport_block(
            registry,
            MemoryTier::Vram,
            DType::U8,
            64 * 1024,
            AllocationId(12),
            0,
        )?,
        unregistered: allocate_ready_transport_block(
            registry,
            MemoryTier::PinnedDram,
            DType::U8,
            32 * 1024,
            AllocationId(13),
            0,
        )?,
    })
}
