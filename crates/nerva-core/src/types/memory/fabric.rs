#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MemoryFabricKind {
    DiscreteExplicit,
    UnifiedVirtualManaged,
    CoherentSharedPhysical,
    CxlCoherentFabric,
}
