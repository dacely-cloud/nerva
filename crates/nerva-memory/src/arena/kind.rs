use nerva_core::types::id::MemoryDomainId;
use nerva_core::types::memory::MemoryTier;

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
