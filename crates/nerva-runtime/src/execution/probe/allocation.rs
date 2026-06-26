use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::layout::LayoutId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_memory::registry::request::BlockAllocationRequest;
use nerva_memory::registry::table::registry::BlockRegistry;

pub(crate) fn allocate_ready_block(
    registry: &mut BlockRegistry,
    kind: BlockKind,
    tier: MemoryTier,
    dtype: DType,
    bytes: usize,
    owner: ExecutionOwner,
) -> Result<ResidentBlockId> {
    let id = registry.allocate(
        BlockAllocationRequest::new(kind, tier, bytes)
            .with_dtype(dtype)
            .with_layout(LayoutId(1)),
    )?;
    let block = registry
        .block_mut(id)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("allocated block {} disappeared", id.0),
        })?;
    block.publish(owner);
    Ok(id)
}
