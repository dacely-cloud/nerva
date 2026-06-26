use nerva_core::types::block::flags::BlockFlags;
use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::Result;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::transport::TransportDeviceId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;

use crate::registry::request::BlockAllocationRequest;
use crate::registry::table::registry::BlockRegistry;

pub(super) struct SecurityProbeFixture {
    pub registry: BlockRegistry,
    pub token_state: ResidentBlockId,
    pub activation: ResidentBlockId,
    pub transport: ResidentBlockId,
    pub non_sensitive: ResidentBlockId,
    pub unready_sensitive: ResidentBlockId,
}

impl SecurityProbeFixture {
    pub fn allocate() -> Result<Self> {
        let mut registry = BlockRegistry::new([
            (MemoryTier::Dram, 8 * 1024 * 1024),
            (MemoryTier::Vram, 8 * 1024 * 1024),
            (MemoryTier::PinnedDram, 8 * 1024 * 1024),
        ]);
        let token_state = allocate_probe_block(
            &mut registry,
            BlockKind::TokenState,
            MemoryTier::PinnedDram,
            4096,
            ExecutionOwner::Cpu,
            7,
            true,
            true,
        )?;
        let activation = allocate_probe_block(
            &mut registry,
            BlockKind::Activation,
            MemoryTier::Vram,
            8192,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            11,
            true,
            true,
        )?;
        let transport = allocate_probe_block(
            &mut registry,
            BlockKind::TransportBuffer,
            MemoryTier::PinnedDram,
            4096,
            ExecutionOwner::Nic(TransportDeviceId(0)),
            5,
            true,
            true,
        )?;
        let non_sensitive = allocate_probe_block(
            &mut registry,
            BlockKind::Metadata,
            MemoryTier::Dram,
            1024,
            ExecutionOwner::Cpu,
            1,
            true,
            false,
        )?;
        let unready_sensitive = allocate_probe_block(
            &mut registry,
            BlockKind::Workspace,
            MemoryTier::Dram,
            1024,
            ExecutionOwner::Cpu,
            1,
            false,
            true,
        )?;

        Ok(Self {
            registry,
            token_state,
            activation,
            transport,
            non_sensitive,
            unready_sensitive,
        })
    }

    pub const fn sensitive_blocks(&self) -> [ResidentBlockId; 3] {
        [self.token_state, self.activation, self.transport]
    }
}

#[allow(clippy::too_many_arguments)]
fn allocate_probe_block(
    registry: &mut BlockRegistry,
    kind: BlockKind,
    tier: MemoryTier,
    bytes: usize,
    owner: ExecutionOwner,
    version: u64,
    ready: bool,
    sensitive: bool,
) -> Result<ResidentBlockId> {
    let id = registry.allocate(BlockAllocationRequest::new(kind, tier, bytes))?;
    {
        let block = registry.block_mut(id).expect("allocated block exists");
        block.owner = owner;
        block.version = version;
        if sensitive {
            block.flags = BlockFlags::from_bits(block.flags.bits() | BlockFlags::SENSITIVE);
        }
        if !ready {
            block.state = ResidencyState::Prefetching;
        }
    }
    if ready {
        registry.mark_ready(id)?;
    }
    Ok(id)
}
