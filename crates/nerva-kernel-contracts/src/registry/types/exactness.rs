#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelExactness {
    BitExact,
    ReferenceEquivalentWithinDeclaredFpTolerance,
    DistributionPreserving,
    Approximate,
}
