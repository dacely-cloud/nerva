use nerva_core::{
    AllocationId, BlockKind, GlobalBlockAddress, MemoryDomainId, MemoryTier, NervaError,
    ResidentBlock, ResidentBlockId, ResidentBlockKind, Result,
};
use nerva_ledger::TokenLedger;

use crate::registry::{BlockAllocationRequest, BlockRegistry};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArenaKind {
    Device,
    PinnedHost,
    Host,
}

impl ArenaKind {
    pub const fn tier(self) -> MemoryTier {
        match self {
            Self::Device => MemoryTier::Vram,
            Self::PinnedHost => MemoryTier::PinnedDram,
            Self::Host => MemoryTier::Dram,
        }
    }

    pub const fn domain(self) -> MemoryDomainId {
        MemoryDomainId::for_tier(self.tier())
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AllocationPhase {
    Initialization,
    HotPath,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArenaReservation {
    pub offset: usize,
    pub bytes: usize,
    pub align: usize,
}

#[derive(Clone, Debug)]
pub struct HostArena {
    bytes: Vec<u8>,
    used: usize,
}

impl HostArena {
    pub fn new(capacity: usize) -> Self {
        Self {
            bytes: vec![0; capacity],
            used: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.bytes.len()
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.used
    }

    pub fn reserve(&mut self, bytes: usize, align: usize) -> Result<ArenaReservation> {
        let align = align.max(1);
        let offset = self.used.next_multiple_of(align);
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes,
                reason: "arena offset overflow".to_string(),
            })?;
        if end > self.bytes.len() {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: "host arena exhausted".to_string(),
            });
        }
        self.used = end;
        Ok(ArenaReservation {
            offset,
            bytes,
            align,
        })
    }

    pub fn reset(&mut self) {
        self.used = 0;
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArenaCheckpoint {
    pub(crate) used: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArenaRegion {
    pub name: &'static str,
    pub kind: ArenaKind,
    pub tier: MemoryTier,
    pub address: GlobalBlockAddress,
    pub offset: usize,
    pub bytes: usize,
    pub align: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticArena {
    kind: ArenaKind,
    allocation: AllocationId,
    capacity_bytes: usize,
    used_bytes: usize,
}

impl StaticArena {
    pub const fn new(kind: ArenaKind, allocation: AllocationId, capacity_bytes: usize) -> Self {
        Self {
            kind,
            allocation,
            capacity_bytes,
            used_bytes: 0,
        }
    }

    pub const fn kind(&self) -> ArenaKind {
        self.kind
    }

    pub const fn tier(&self) -> MemoryTier {
        self.kind.tier()
    }

    pub const fn domain(&self) -> MemoryDomainId {
        self.kind.domain()
    }

    pub const fn allocation(&self) -> AllocationId {
        self.allocation
    }

    pub const fn capacity(&self) -> usize {
        self.capacity_bytes
    }

    pub const fn used(&self) -> usize {
        self.used_bytes
    }

    pub const fn remaining(&self) -> usize {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }

    pub const fn checkpoint(&self) -> ArenaCheckpoint {
        ArenaCheckpoint {
            used: self.used_bytes,
        }
    }

    pub fn restore(&mut self, checkpoint: ArenaCheckpoint) -> Result<()> {
        if checkpoint.used > self.used_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "arena checkpoint is ahead of current usage".to_string(),
            });
        }
        self.used_bytes = checkpoint.used;
        Ok(())
    }

    pub fn reserve(
        &mut self,
        name: &'static str,
        bytes: usize,
        align: usize,
        phase: AllocationPhase,
    ) -> Result<ArenaRegion> {
        if phase == AllocationPhase::HotPath {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: "static arena allocation attempted during hot path".to_string(),
            });
        }
        let align = align.max(1);
        let offset = self.used_bytes.next_multiple_of(align);
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes,
                reason: "static arena offset overflow".to_string(),
            })?;
        if end > self.capacity_bytes {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: format!("static {:?} arena exhausted", self.kind),
            });
        }
        self.used_bytes = end;
        Ok(ArenaRegion {
            name,
            kind: self.kind,
            tier: self.tier(),
            address: GlobalBlockAddress {
                domain: self.domain(),
                allocation: self.allocation,
                offset: offset as u64,
            },
            offset,
            bytes,
            align,
        })
    }
}

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

pub fn resident_block_for_reservation(
    id: ResidentBlockId,
    kind: ResidentBlockKind,
    reservation: ArenaReservation,
) -> ResidentBlock {
    ResidentBlock::new(id, kind, MemoryTier::Dram, reservation.bytes).with_address(
        GlobalBlockAddress {
            domain: MemoryDomainId::CPU_DRAM,
            allocation: AllocationId(id.0),
            offset: reservation.offset as u64,
        },
    )
}
