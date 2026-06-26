use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::error::Result;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;

use crate::arena::kind::{AllocationPhase, ArenaKind};
use crate::arena::set::static_set::StaticArenaSet;
use crate::registry::request::BlockAllocationRequest;
use crate::registry::table::registry::BlockRegistry;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StaticArenaBootstrapSpec {
    pub device_token_state_bytes: usize,
    pub pinned_observation_bytes: usize,
    pub host_metadata_bytes: usize,
    pub align: usize,
}

impl Default for StaticArenaBootstrapSpec {
    fn default() -> Self {
        Self {
            device_token_state_bytes: 256,
            pinned_observation_bytes: 256,
            host_metadata_bytes: 512,
            align: 64,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StaticArenaBootstrap {
    pub device_token_state: ResidentBlockId,
    pub pinned_observation: ResidentBlockId,
    pub host_metadata: ResidentBlockId,
}

impl StaticArenaSet {
    pub fn preallocate_decode_bootstrap(
        &mut self,
        registry: &mut BlockRegistry,
        spec: StaticArenaBootstrapSpec,
    ) -> Result<StaticArenaBootstrap> {
        let align = spec.align.max(1);
        let device_token_state = self.reserve_resident_block(
            registry,
            ArenaKind::Device,
            "device-token-ring",
            BlockAllocationRequest::new(
                BlockKind::TokenState,
                MemoryTier::Vram,
                spec.device_token_state_bytes,
            ),
            align,
            AllocationPhase::Initialization,
        )?;
        registry.mark_ready(device_token_state)?;

        let pinned_observation = self.reserve_resident_block(
            registry,
            ArenaKind::PinnedHost,
            "host-token-observation",
            BlockAllocationRequest::new(
                BlockKind::TokenState,
                MemoryTier::PinnedDram,
                spec.pinned_observation_bytes,
            ),
            align,
            AllocationPhase::Initialization,
        )?;
        registry.mark_ready(pinned_observation)?;

        let host_metadata = self.reserve_resident_block(
            registry,
            ArenaKind::Host,
            "runtime-metadata",
            BlockAllocationRequest::new(
                BlockKind::Metadata,
                MemoryTier::Dram,
                spec.host_metadata_bytes,
            ),
            align,
            AllocationPhase::Initialization,
        )?;
        registry.mark_ready(host_metadata)?;

        Ok(StaticArenaBootstrap {
            device_token_state,
            pinned_observation,
            host_metadata,
        })
    }
}
