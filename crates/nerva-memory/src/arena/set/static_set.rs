use nerva_core::types::error::Result;
use nerva_core::types::id::allocation::AllocationId;

use crate::arena::kind::{AllocationPhase, ArenaKind};
use crate::arena::region::ArenaRegion;
use crate::arena::static_arena::StaticArena;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticArenaSet {
    device: StaticArena,
    pinned_host: StaticArena,
    host: StaticArena,
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
}
