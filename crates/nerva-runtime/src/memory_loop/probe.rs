use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::layout::LayoutId;

use nerva_core::types::memory::tier::MemoryTier;
use nerva_memory::registry::request::BlockAllocationRequest;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::memory_loop::plan::plan_memory_loop;
use crate::memory_loop::run::execute_memory_loop_plan;
use crate::memory_loop::summary::MemoryLoopSummary;
use crate::memory_loop::types::{MemoryLoopConfig, MemoryLoopTaskKind, MemoryLoopTaskSpec};

pub fn run_memory_loop_probe() -> Result<MemoryLoopSummary> {
    let (mut registry, config) = reference_memory_loop_fixture()?;
    let plan = plan_memory_loop(config, &registry)?;
    execute_memory_loop_plan(&mut registry, &plan)
}

pub(crate) fn reference_memory_loop_fixture() -> Result<(BlockRegistry, MemoryLoopConfig)> {
    let mut registry = BlockRegistry::new([
        (MemoryTier::Disk, 2 * 1024 * 1024),
        (MemoryTier::Dram, 2 * 1024 * 1024),
        (MemoryTier::PinnedDram, 1024 * 1024),
        (MemoryTier::Vram, 2 * 1024 * 1024),
    ]);
    let cold_weight = allocate_block(
        &mut registry,
        BlockKind::Weight,
        MemoryTier::Disk,
        DType::F16,
        256 * 1024,
    )?;
    let warm_weight = allocate_block(
        &mut registry,
        BlockKind::Weight,
        MemoryTier::Dram,
        DType::F16,
        128 * 1024,
    )?;
    let hot_kv = allocate_block(
        &mut registry,
        BlockKind::KvPage,
        MemoryTier::Vram,
        DType::F16,
        64 * 1024,
    )?;
    let transport = allocate_block(
        &mut registry,
        BlockKind::TransportBuffer,
        MemoryTier::PinnedDram,
        DType::U8,
        32 * 1024,
    )?;

    let config = MemoryLoopConfig::new(8, 2)
        .with_task(
            MemoryLoopTaskSpec::new(
                cold_weight,
                MemoryLoopTaskKind::DiskRead,
                MemoryTier::Disk,
                MemoryTier::Dram,
                256 * 1024,
                2_000,
                "memory_loop_disk_to_dram",
            )
            .with_overlap(1_500),
        )
        .with_task(
            MemoryLoopTaskSpec::new(
                warm_weight,
                MemoryLoopTaskKind::Prefetch,
                MemoryTier::Dram,
                MemoryTier::PinnedDram,
                128 * 1024,
                1_000,
                "memory_loop_dram_to_pinned",
            )
            .with_overlap(600),
        )
        .with_task(
            MemoryLoopTaskSpec::new(
                warm_weight,
                MemoryLoopTaskKind::Stage,
                MemoryTier::PinnedDram,
                MemoryTier::Vram,
                128 * 1024,
                900,
                "memory_loop_pinned_to_vram",
            )
            .with_overlap(400),
        )
        .with_task(MemoryLoopTaskSpec::new(
            hot_kv,
            MemoryLoopTaskKind::Evict,
            MemoryTier::Vram,
            MemoryTier::Dram,
            64 * 1024,
            700,
            "memory_loop_vram_to_dram_eviction",
        ))
        .with_task(MemoryLoopTaskSpec::new(
            transport,
            MemoryLoopTaskKind::PrepareTransportBuffer,
            MemoryTier::PinnedDram,
            MemoryTier::PinnedDram,
            32 * 1024,
            200,
            "memory_loop_transport_ring_prepare",
        ));
    Ok((registry, config))
}

fn allocate_block(
    registry: &mut BlockRegistry,
    kind: BlockKind,
    tier: MemoryTier,
    dtype: DType,
    bytes: usize,
) -> Result<ResidentBlockId> {
    let id = registry.allocate(
        BlockAllocationRequest::new(kind, tier, bytes)
            .with_dtype(dtype)
            .with_layout(LayoutId(2)),
    )?;
    registry
        .mark_ready(id)
        .map_err(|err| NervaError::InvalidArgument {
            reason: format!("failed to ready memory-loop block {}: {err:?}", id.0),
        })?;
    Ok(id)
}
