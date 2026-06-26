use nerva_core::types::block::taxonomy::BlockKind;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::{AllocationId, ResidentBlockId};
use nerva_core::types::memory::MemoryTier;
use nerva_ledger::types::token::TokenLedger;

use crate::arena::kind::{AllocationPhase, ArenaKind};
use crate::arena::region::ArenaRegion;
use crate::arena::static_arena::StaticArena;
use crate::registry::{BlockAllocationRequest, BlockRegistry};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticArenaSet {
    device: StaticArena,
    pinned_host: StaticArena,
    host: StaticArena,
}

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
    pub const fn new(device_bytes: usize, pinned_host_bytes: usize, host_bytes: usize) -> Self {
        Self {
            device: StaticArena::new(ArenaKind::Device, AllocationId(1), device_bytes),
            pinned_host: StaticArena::new(
                ArenaKind::PinnedHost,
                AllocationId(2),
                pinned_host_bytes,
            ),
            host: StaticArena::new(ArenaKind::Host, AllocationId(3), host_bytes),
        }
    }

    pub const fn device(&self) -> &StaticArena {
        &self.device
    }

    pub const fn pinned_host(&self) -> &StaticArena {
        &self.pinned_host
    }

    pub const fn host(&self) -> &StaticArena {
        &self.host
    }

    pub fn arena_mut(&mut self, kind: ArenaKind) -> &mut StaticArena {
        match kind {
            ArenaKind::Device => &mut self.device,
            ArenaKind::PinnedHost => &mut self.pinned_host,
            ArenaKind::Host => &mut self.host,
        }
    }

    pub fn reserve(
        &mut self,
        kind: ArenaKind,
        name: &'static str,
        bytes: usize,
        align: usize,
        phase: AllocationPhase,
    ) -> Result<ArenaRegion> {
        self.arena_mut(kind).reserve(name, bytes, align, phase)
    }

    pub fn reject_hot_path_reservation_with_ledger(
        &mut self,
        kind: ArenaKind,
        name: &'static str,
        bytes: usize,
        align: usize,
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let before = self.arena_mut(kind).used();
        match self.reserve(kind, name, bytes, align, AllocationPhase::HotPath) {
            Ok(_) => Err(NervaError::InvalidArgument {
                reason: "static arena accepted forbidden hot-path reservation".to_string(),
            }),
            Err(err) => {
                debug_assert_eq!(self.arena_mut(kind).used(), before);
                ledger.record_hot_path_allocation_attempt(name, bytes, kind.tier());
                Err(err)
            }
        }
    }

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

    pub fn reserve_resident_block(
        &mut self,
        registry: &mut BlockRegistry,
        kind: ArenaKind,
        name: &'static str,
        request: BlockAllocationRequest,
        align: usize,
        phase: AllocationPhase,
    ) -> Result<ResidentBlockId> {
        if request.tier != kind.tier() {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "arena kind {:?} cannot reserve block requested for {:?}",
                    kind, request.tier
                ),
            });
        }

        let checkpoint = self.arena_mut(kind).checkpoint();
        let region = self.reserve(kind, name, request.bytes, align, phase)?;
        match registry.allocate(request) {
            Ok(id) => {
                registry.bind_address(id, region.address)?;
                Ok(id)
            }
            Err(err) => {
                let _ = self.arena_mut(kind).restore(checkpoint);
                Err(err)
            }
        }
    }
}
