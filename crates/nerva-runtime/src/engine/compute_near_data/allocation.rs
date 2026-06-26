use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::layout::LayoutId;

use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_core::types::shape::BlockShape;
use nerva_memory::registry::request::BlockAllocationRequest;
use nerva_memory::registry::table::registry::BlockRegistry;

pub(crate) fn allocate_weight_shard(
    registry: &mut BlockRegistry,
    tier: MemoryTier,
    bytes: usize,
    rows: usize,
    cols: usize,
) -> Result<ResidentBlockId> {
    let block_id = registry.allocate(
        BlockAllocationRequest::new(BlockKind::Weight, tier, bytes)
            .with_dtype(DType::F32)
            .with_layout(LayoutId(7)),
    )?;
    let block = registry
        .block_mut(block_id)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("allocated weight shard {} is missing", block_id.0),
        })?;
    block.shape = BlockShape::from_dims([rows as u64, cols as u64])?;
    block.owner = ExecutionOwner::SharedReadOnly;
    registry.mark_ready(block_id)?;
    Ok(block_id)
}
