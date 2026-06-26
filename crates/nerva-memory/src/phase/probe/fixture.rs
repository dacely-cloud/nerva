use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::Result;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::mutation::MutationSemantics;
use nerva_core::types::ownership::owner::ExecutionOwner;

use crate::registry::request::BlockAllocationRequest;
use crate::registry::table::registry::BlockRegistry;

pub(super) struct PhaseProbeFixture {
    pub registry: BlockRegistry,
    pub cpu_activation: ResidentBlockId,
    pub gpu_logits: ResidentBlockId,
    pub gpu_transport: ResidentBlockId,
    pub wrong_owner: ResidentBlockId,
    pub stale: ResidentBlockId,
    pub unready: ResidentBlockId,
    pub shared_read_only: ResidentBlockId,
}

impl PhaseProbeFixture {
    pub fn allocate() -> Result<Self> {
        let mut registry = BlockRegistry::new([
            (MemoryTier::Dram, 8 * 1024 * 1024),
            (MemoryTier::Vram, 8 * 1024 * 1024),
            (MemoryTier::PinnedDram, 8 * 1024 * 1024),
        ]);
        let cpu_activation = allocate_phase_block(
            &mut registry,
            BlockKind::Activation,
            MemoryTier::Dram,
            4096,
            ExecutionOwner::Cpu,
            1,
            true,
        )?;
        let gpu_logits = allocate_phase_block(
            &mut registry,
            BlockKind::Logits,
            MemoryTier::Vram,
            4096,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            2,
            true,
        )?;
        let gpu_transport = allocate_phase_block(
            &mut registry,
            BlockKind::TransportBuffer,
            MemoryTier::Vram,
            4096,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            3,
            true,
        )?;
        let wrong_owner = allocate_phase_block(
            &mut registry,
            BlockKind::Activation,
            MemoryTier::Dram,
            4096,
            ExecutionOwner::Cpu,
            1,
            true,
        )?;
        let stale = allocate_phase_block(
            &mut registry,
            BlockKind::Activation,
            MemoryTier::Dram,
            4096,
            ExecutionOwner::Cpu,
            1,
            true,
        )?;
        let unready = allocate_phase_block(
            &mut registry,
            BlockKind::Activation,
            MemoryTier::Dram,
            4096,
            ExecutionOwner::Cpu,
            1,
            false,
        )?;
        let shared_read_only = allocate_phase_block(
            &mut registry,
            BlockKind::Weight,
            MemoryTier::Dram,
            4096,
            ExecutionOwner::SharedReadOnly,
            1,
            true,
        )?;
        registry
            .block_mut(shared_read_only)
            .expect("probe block exists")
            .semantics = MutationSemantics::Immutable;

        Ok(Self {
            registry,
            cpu_activation,
            gpu_logits,
            gpu_transport,
            wrong_owner,
            stale,
            unready,
            shared_read_only,
        })
    }
}

fn allocate_phase_block(
    registry: &mut BlockRegistry,
    kind: BlockKind,
    tier: MemoryTier,
    bytes: usize,
    owner: ExecutionOwner,
    version: u64,
    ready: bool,
) -> Result<ResidentBlockId> {
    let id = registry.allocate(BlockAllocationRequest::new(kind, tier, bytes))?;
    {
        let block = registry.block_mut(id).expect("allocated block exists");
        block.owner = owner;
        block.version = version;
        if !ready {
            block.state = ResidencyState::Prefetching;
        }
    }
    if ready {
        registry.mark_ready(id)?;
    }
    Ok(id)
}
