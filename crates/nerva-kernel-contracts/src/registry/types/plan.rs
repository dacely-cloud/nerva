use crate::registry::types::fallback::KernelFallback;
use crate::registry::types::implementation::KernelImplementation;
use crate::registry::types::query::KernelQuery;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelPlan {
    Direct {
        implementation: KernelImplementation,
    },
    Fallback {
        requested: KernelQuery,
        fallback: KernelImplementation,
        policy: KernelFallback,
    },
}

impl KernelPlan {
    pub const fn is_fallback(self) -> bool {
        matches!(self, Self::Fallback { .. })
    }

    pub const fn implementation(self) -> KernelImplementation {
        match self {
            Self::Direct { implementation } => implementation,
            Self::Fallback { fallback, .. } => fallback,
        }
    }
}
