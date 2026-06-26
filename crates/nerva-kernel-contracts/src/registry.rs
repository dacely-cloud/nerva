pub mod bootstrap;
pub mod probe;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelBackend {
    CpuReference,
    Cuda,
    Hip,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelOperation {
    DenseMatVec,
    BlockwiseAttention,
    KvAppend,
    GreedySample,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelExactness {
    BitExact,
    ReferenceEquivalentWithinDeclaredFpTolerance,
    DistributionPreserving,
    Approximate,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelFallbackClass {
    ExactNamed,
    ApproximateNamed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArchitectureRange {
    pub min_compute_capability: u32,
    pub max_compute_capability: u32,
}

impl ArchitectureRange {
    pub const fn new(min_compute_capability: u32, max_compute_capability: u32) -> Self {
        Self {
            min_compute_capability,
            max_compute_capability,
        }
    }

    pub const fn contains(self, compute_capability: u32) -> bool {
        compute_capability >= self.min_compute_capability
            && compute_capability <= self.max_compute_capability
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KernelImplementation {
    pub name: &'static str,
    pub operation: KernelOperation,
    pub backend: KernelBackend,
    pub architecture: Option<ArchitectureRange>,
    pub dtypes: &'static [DType],
    pub graph_safe: bool,
    pub deterministic: bool,
    pub exactness: KernelExactness,
}

impl KernelImplementation {
    pub fn matches(self, query: KernelQuery) -> bool {
        if self.operation != query.operation || self.backend != query.backend {
            return false;
        }
        if !self.dtypes.contains(&query.dtype) {
            return false;
        }
        match (self.architecture, query.compute_capability) {
            (Some(range), Some(compute_capability)) => range.contains(compute_capability),
            (Some(_), None) => false,
            (None, _) => true,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KernelFallback {
    pub operation: KernelOperation,
    pub requested_backend: KernelBackend,
    pub requested_dtype: DType,
    pub fallback_backend: KernelBackend,
    pub fallback_dtype: DType,
    pub name: &'static str,
    pub class: KernelFallbackClass,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KernelQuery {
    pub operation: KernelOperation,
    pub backend: KernelBackend,
    pub dtype: DType,
    pub compute_capability: Option<u32>,
}

impl KernelQuery {
    pub const fn new(
        operation: KernelOperation,
        backend: KernelBackend,
        dtype: DType,
        compute_capability: Option<u32>,
    ) -> Self {
        Self {
            operation,
            backend,
            dtype,
            compute_capability,
        }
    }
}

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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KernelContractRegistry {
    implementations: Vec<KernelImplementation>,
    fallbacks: Vec<KernelFallback>,
}

impl KernelContractRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_implementation(mut self, implementation: KernelImplementation) -> Self {
        self.implementations.push(implementation);
        self
    }

    pub fn with_fallback(mut self, fallback: KernelFallback) -> Self {
        self.fallbacks.push(fallback);
        self
    }

    pub fn implementations(&self) -> &[KernelImplementation] {
        &self.implementations
    }

    pub fn fallbacks(&self) -> &[KernelFallback] {
        &self.fallbacks
    }

    pub fn resolve(&self, query: KernelQuery) -> Result<KernelPlan> {
        if let Some(implementation) = self
            .implementations
            .iter()
            .copied()
            .find(|implementation| implementation.matches(query))
        {
            return Ok(KernelPlan::Direct { implementation });
        }

        for policy in &self.fallbacks {
            if policy.operation != query.operation
                || policy.requested_backend != query.backend
                || policy.requested_dtype != query.dtype
            {
                continue;
            }
            if policy.class != KernelFallbackClass::ExactNamed {
                return Err(NervaError::InvalidArgument {
                    reason: format!("kernel fallback {} is not exact", policy.name),
                });
            }
            let fallback_query = KernelQuery::new(
                query.operation,
                policy.fallback_backend,
                policy.fallback_dtype,
                None,
            );
            if let Some(fallback) = self
                .implementations
                .iter()
                .copied()
                .find(|implementation| implementation.matches(fallback_query))
            {
                return Ok(KernelPlan::Fallback {
                    requested: query,
                    fallback,
                    policy: *policy,
                });
            }
            return Err(NervaError::InvalidArgument {
                reason: format!("declared fallback {} has no matching contract", policy.name),
            });
        }

        Err(NervaError::InvalidArgument {
            reason: format!("no kernel contract for {:?}", query),
        })
    }
}
