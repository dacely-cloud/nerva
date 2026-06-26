use nerva_core::types::{DType, NervaError, Result};

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

static DTYPE_F32: &[DType] = &[DType::F32];
static DTYPE_FP16_BF16: &[DType] = &[DType::F16, DType::BF16];
static DTYPE_U32: &[DType] = &[DType::U32];

pub fn bootstrap_registry() -> KernelContractRegistry {
    KernelContractRegistry::new()
        .with_implementation(KernelImplementation {
            name: "cpu_reference_dense_matvec_f32",
            operation: KernelOperation::DenseMatVec,
            backend: KernelBackend::CpuReference,
            architecture: None,
            dtypes: DTYPE_F32,
            graph_safe: false,
            deterministic: true,
            exactness: KernelExactness::BitExact,
        })
        .with_implementation(KernelImplementation {
            name: "cpu_reference_blockwise_attention_f32",
            operation: KernelOperation::BlockwiseAttention,
            backend: KernelBackend::CpuReference,
            architecture: None,
            dtypes: DTYPE_F32,
            graph_safe: false,
            deterministic: true,
            exactness: KernelExactness::ReferenceEquivalentWithinDeclaredFpTolerance,
        })
        .with_implementation(KernelImplementation {
            name: "cuda_decode_dense_matvec_fp16_bf16",
            operation: KernelOperation::DenseMatVec,
            backend: KernelBackend::Cuda,
            architecture: Some(ArchitectureRange::new(75, 121)),
            dtypes: DTYPE_FP16_BF16,
            graph_safe: true,
            deterministic: true,
            exactness: KernelExactness::ReferenceEquivalentWithinDeclaredFpTolerance,
        })
        .with_implementation(KernelImplementation {
            name: "cuda_greedy_sample_u32",
            operation: KernelOperation::GreedySample,
            backend: KernelBackend::Cuda,
            architecture: Some(ArchitectureRange::new(75, 121)),
            dtypes: DTYPE_U32,
            graph_safe: true,
            deterministic: true,
            exactness: KernelExactness::BitExact,
        })
        .with_fallback(KernelFallback {
            operation: KernelOperation::DenseMatVec,
            requested_backend: KernelBackend::Cuda,
            requested_dtype: DType::F32,
            fallback_backend: KernelBackend::CpuReference,
            fallback_dtype: DType::F32,
            name: "cuda_f32_dense_matvec_to_cpu_reference",
            class: KernelFallbackClass::ExactNamed,
        })
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelRegistryProbeStatus {
    Ok,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KernelRegistryProbeSummary {
    pub status: KernelRegistryProbeStatus,
    pub implementations: usize,
    pub fallbacks: usize,
    pub direct_plans: u64,
    pub fallback_plans: u64,
    pub rejected_plans: u64,
    pub graph_safe_direct: u64,
    pub exact_fallbacks: u64,
}

impl KernelRegistryProbeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            KernelRegistryProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"implementations\":{},\"fallbacks\":{},\"direct_plans\":{},\"fallback_plans\":{},\"rejected_plans\":{},\"graph_safe_direct\":{},\"exact_fallbacks\":{}}}",
            status,
            self.implementations,
            self.fallbacks,
            self.direct_plans,
            self.fallback_plans,
            self.rejected_plans,
            self.graph_safe_direct,
            self.exact_fallbacks,
        )
    }
}

pub fn kernel_registry_probe() -> Result<KernelRegistryProbeSummary> {
    let registry = bootstrap_registry();
    let direct = registry.resolve(KernelQuery::new(
        KernelOperation::DenseMatVec,
        KernelBackend::Cuda,
        DType::F16,
        Some(89),
    ))?;
    let fallback = registry.resolve(KernelQuery::new(
        KernelOperation::DenseMatVec,
        KernelBackend::Cuda,
        DType::F32,
        Some(89),
    ))?;
    let rejected = registry
        .resolve(KernelQuery::new(
            KernelOperation::DenseMatVec,
            KernelBackend::Hip,
            DType::BF16,
            Some(1100),
        ))
        .is_err();

    Ok(KernelRegistryProbeSummary {
        status: KernelRegistryProbeStatus::Ok,
        implementations: registry.implementations().len(),
        fallbacks: registry.fallbacks().len(),
        direct_plans: u64::from(!direct.is_fallback()),
        fallback_plans: u64::from(fallback.is_fallback()),
        rejected_plans: u64::from(rejected),
        graph_safe_direct: u64::from(direct.implementation().graph_safe),
        exact_fallbacks: u64::from(matches!(
            fallback,
            KernelPlan::Fallback {
                policy: KernelFallback {
                    class: KernelFallbackClass::ExactNamed,
                    ..
                },
                ..
            }
        )),
    })
}
