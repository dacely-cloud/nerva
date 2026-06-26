use nerva_core::types::id::MemoryDomainId;
use nerva_core::types::memory::MemoryTier;

pub(crate) fn memory_tier_for_domain(domain: MemoryDomainId) -> Option<MemoryTier> {
    Some(match domain {
        MemoryDomainId::GPU_VRAM => MemoryTier::Vram,
        MemoryDomainId::PINNED_DRAM => MemoryTier::PinnedDram,
        MemoryDomainId::CPU_DRAM => MemoryTier::Dram,
        MemoryDomainId::SHARED_HBM_OR_LPDDR => MemoryTier::SharedHbmOrLpddr,
        MemoryDomainId::CXL => MemoryTier::Cxl,
        MemoryDomainId::DISK => MemoryTier::Disk,
        _ => return None,
    })
}
