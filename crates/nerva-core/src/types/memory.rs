#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MemoryFabricKind {
    DiscreteExplicit,
    UnifiedVirtualManaged,
    CoherentSharedPhysical,
    CxlCoherentFabric,
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MemoryTier {
    Vram,
    SharedHbmOrLpddr,
    PinnedDram,
    Dram,
    Cxl,
    Disk,
}
