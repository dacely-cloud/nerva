use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::error::Result;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::device::DeviceOrdinal;

use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::coherence::CoherencePolicy;
use nerva_core::types::ownership::mutation::MutationSemantics;
use nerva_core::types::ownership::owner::ExecutionOwner;

use crate::queue::types::SharedQueueDescriptor;
use crate::registry::request::BlockAllocationRequest;
use crate::registry::table::registry::BlockRegistry;

pub(crate) fn allocate_queue_block(registry: &mut BlockRegistry) -> Result<ResidentBlockId> {
    let block_id = registry.allocate(BlockAllocationRequest::new(
        BlockKind::Queue,
        MemoryTier::SharedHbmOrLpddr,
        4096,
    ))?;
    {
        let block = registry
            .block_mut(block_id)
            .expect("allocated block exists");
        block.coherence = CoherencePolicy::AtomicControlOnly;
        block.semantics = MutationSemantics::AtomicControl;
        block.owner = ExecutionOwner::PhaseTransition;
    }
    registry.mark_ready(block_id)?;
    Ok(block_id)
}

pub(crate) fn allocate_tensor_block(
    registry: &mut BlockRegistry,
    bytes: usize,
) -> Result<ResidentBlockId> {
    let block_id = registry.allocate(BlockAllocationRequest::new(
        BlockKind::Activation,
        MemoryTier::Vram,
        bytes,
    ))?;
    {
        let block = registry
            .block_mut(block_id)
            .expect("allocated block exists");
        block.owner = ExecutionOwner::Gpu(DeviceOrdinal(0));
        block.version = 1;
    }
    registry.mark_ready(block_id)?;
    Ok(block_id)
}

pub(crate) fn descriptor(
    descriptor_id: u64,
    block_id: ResidentBlockId,
    block_version: u64,
    referenced_bytes: usize,
    payload_bytes_in_queue: usize,
) -> SharedQueueDescriptor {
    SharedQueueDescriptor {
        descriptor_id,
        block_id,
        block_version,
        referenced_bytes,
        metadata_bytes: core::mem::size_of::<SharedQueueDescriptor>(),
        payload_bytes_in_queue,
        label: "shared_queue_block_handle",
    }
}

pub(crate) fn count_queue_blocks(registry: &BlockRegistry, block_ids: [ResidentBlockId; 2]) -> u64 {
    block_ids
        .iter()
        .filter(|block_id| {
            registry
                .block(**block_id)
                .is_some_and(|block| block.kind == BlockKind::Queue)
        })
        .count() as u64
}

pub(crate) fn count_atomic_control_blocks(
    registry: &BlockRegistry,
    block_ids: [ResidentBlockId; 2],
) -> u64 {
    block_ids
        .iter()
        .filter(|block_id| {
            registry.block(**block_id).is_some_and(|block| {
                block.coherence == CoherencePolicy::AtomicControlOnly
                    && block.semantics == MutationSemantics::AtomicControl
            })
        })
        .count() as u64
}
