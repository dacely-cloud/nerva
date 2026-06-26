#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ExactnessClass {
    BitExact,
    ReferenceEquivalentWithinDeclaredFpTolerance,
    DistributionPreserving,
    Approximate,
}

impl ExactnessClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BitExact => "bit-exact",
            Self::ReferenceEquivalentWithinDeclaredFpTolerance => "fp-tolerance",
            Self::DistributionPreserving => "distribution-preserving",
            Self::Approximate => "approximate",
        }
    }

    pub const fn accepted_for_core_runtime(self) -> bool {
        !matches!(self, Self::Approximate)
    }
}
