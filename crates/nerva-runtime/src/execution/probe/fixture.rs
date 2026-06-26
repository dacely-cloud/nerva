use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::layout::LayoutId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_memory::registry::request::BlockAllocationRequest;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::execution::probe::allocation::allocate_ready_block;
use crate::execution::probe::blocks::ReferenceTransactionBlocks;
use crate::execution::probe::spec::reference_transaction_spec;
use crate::execution::types::ExecutionTransactionSpec;

pub(crate) fn reference_transaction_fixture(
    device: DeviceOrdinal,
) -> Result<(
    BlockRegistry,
    ExecutionTransactionSpec,
    ReferenceTransactionBlocks,
)> {
    let gpu = ExecutionOwner::Gpu(device);
    let mut registry = BlockRegistry::new([
        (MemoryTier::Vram, 16 * 1024 * 1024),
        (MemoryTier::PinnedDram, 1024 * 1024),
        (MemoryTier::Dram, 16 * 1024 * 1024),
    ]);

    let device_token = allocate_ready_block(
        &mut registry,
        BlockKind::TokenState,
        MemoryTier::Vram,
        DType::U32,
        4096,
        gpu,
    )?;
    let weight_tile = allocate_ready_block(
        &mut registry,
        BlockKind::Weight,
        MemoryTier::Vram,
        DType::F16,
        128 * 1024,
        ExecutionOwner::SharedReadOnly,
    )?;
    let kv_page = allocate_ready_block(
        &mut registry,
        BlockKind::KvPage,
        MemoryTier::Vram,
        DType::F16,
        64 * 1024,
        gpu,
    )?;
    let logits = allocate_ready_block(
        &mut registry,
        BlockKind::Logits,
        MemoryTier::Vram,
        DType::F32,
        32 * 1024,
        ExecutionOwner::Cpu,
    )?;
    let host_token = registry.allocate(
        BlockAllocationRequest::new(BlockKind::TokenState, MemoryTier::PinnedDram, 4096)
            .with_dtype(DType::U32)
            .with_layout(LayoutId(1)),
    )?;
    registry.mark_ready(host_token)?;

    let blocks = ReferenceTransactionBlocks {
        device_token,
        weight_tile,
        kv_page,
        logits,
        host_token,
    };
    let spec = reference_transaction_spec(device, blocks);
    Ok((registry, spec, blocks))
}
