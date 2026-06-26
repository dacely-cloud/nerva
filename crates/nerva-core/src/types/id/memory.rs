use crate::types::memory::tier::MemoryTier;

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct MemoryDomainId(pub u32);

impl MemoryDomainId {
    pub const CPU_DRAM: Self = Self(1);
    pub const GPU_VRAM: Self = Self(2);
    pub const PINNED_DRAM: Self = Self(3);
    pub const SHARED_HBM_OR_LPDDR: Self = Self(4);
    pub const CXL: Self = Self(5);
    pub const DISK: Self = Self(6);

    pub const fn for_tier(tier: MemoryTier) -> Self {
        match tier {
            MemoryTier::Vram => Self::GPU_VRAM,
            MemoryTier::SharedHbmOrLpddr => Self::SHARED_HBM_OR_LPDDR,
            MemoryTier::PinnedDram => Self::PINNED_DRAM,
            MemoryTier::Dram => Self::CPU_DRAM,
            MemoryTier::Cxl => Self::CXL,
            MemoryTier::Disk => Self::DISK,
        }
    }
}
