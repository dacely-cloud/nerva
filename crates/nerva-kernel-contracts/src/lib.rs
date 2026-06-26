#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use nerva_core::{BlockKind, DType, MemoryTier, NervaError, Result};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelContractKind {
    DecodeGraph,
    DenseMatvec,
    BlockwiseAttention,
    Sampler,
    ResidencyTransfer,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelBufferRole {
    Input,
    Output,
    InOut,
    Scratch,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LaunchBounds {
    pub max_grid_blocks: u32,
    pub max_threads_per_block: u32,
}

impl LaunchBounds {
    pub fn new(max_grid_blocks: u32, max_threads_per_block: u32) -> Result<Self> {
        if max_grid_blocks == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "kernel launch must allow at least one grid block".to_string(),
            });
        }
        if max_threads_per_block == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "kernel launch must allow at least one thread per block".to_string(),
            });
        }
        Ok(Self {
            max_grid_blocks,
            max_threads_per_block,
        })
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KernelBufferContract {
    pub name: &'static str,
    pub role: KernelBufferRole,
    pub block_kind: BlockKind,
    pub dtype: DType,
    pub expected_tier: MemoryTier,
    pub min_bytes: usize,
}

impl KernelBufferContract {
    pub fn new(
        name: &'static str,
        role: KernelBufferRole,
        block_kind: BlockKind,
        dtype: DType,
        expected_tier: MemoryTier,
        min_bytes: usize,
    ) -> Result<Self> {
        if name.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "kernel buffer contract name must be non-empty".to_string(),
            });
        }
        if min_bytes == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "kernel buffer contract must require non-zero bytes".to_string(),
            });
        }
        Ok(Self {
            name,
            role,
            block_kind,
            dtype,
            expected_tier,
            min_bytes,
        })
    }

    pub const fn requires_device_residency(self) -> bool {
        matches!(
            self.expected_tier,
            MemoryTier::Vram | MemoryTier::SharedHbmOrLpddr
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelContract {
    pub name: &'static str,
    pub kind: KernelContractKind,
    pub launch_bounds: LaunchBounds,
    pub buffers: Vec<KernelBufferContract>,
    pub hot_path_allocation_allowed: bool,
}

impl KernelContract {
    pub fn new(
        name: &'static str,
        kind: KernelContractKind,
        launch_bounds: LaunchBounds,
        buffers: Vec<KernelBufferContract>,
    ) -> Result<Self> {
        if name.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "kernel contract name must be non-empty".to_string(),
            });
        }
        if buffers.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "kernel contract must describe at least one buffer".to_string(),
            });
        }
        Ok(Self {
            name,
            kind,
            launch_bounds,
            buffers,
            hot_path_allocation_allowed: false,
        })
    }

    pub fn with_hot_path_allocation_allowed(mut self, allowed: bool) -> Self {
        self.hot_path_allocation_allowed = allowed;
        self
    }

    pub fn require_decode_ready(&self) -> Result<()> {
        if self.hot_path_allocation_allowed {
            return Err(NervaError::InvalidArgument {
                reason: format!("kernel contract {} permits hot-path allocation", self.name),
            });
        }
        if !self
            .buffers
            .iter()
            .any(|buffer| buffer.requires_device_residency())
        {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "kernel contract {} has no device-resident buffers",
                    self.name
                ),
            });
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelContractProbeStatus {
    Ok,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelContractProbeSummary {
    pub status: KernelContractProbeStatus,
    pub contract_count: usize,
    pub buffer_count: usize,
    pub device_resident_buffers: usize,
    pub hot_path_allocation_allowed: bool,
    pub max_grid_blocks: u32,
    pub max_threads_per_block: u32,
}

impl KernelContractProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            KernelContractProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"contract_count\":{},\"buffer_count\":{},\"device_resident_buffers\":{},\"hot_path_allocation_allowed\":{},\"max_grid_blocks\":{},\"max_threads_per_block\":{}}}",
            status,
            self.contract_count,
            self.buffer_count,
            self.device_resident_buffers,
            self.hot_path_allocation_allowed,
            self.max_grid_blocks,
            self.max_threads_per_block,
        )
    }
}

pub fn kernel_contract_probe() -> Result<KernelContractProbeSummary> {
    let bounds = LaunchBounds::new(64, 256)?;
    let token_ring = KernelBufferContract::new(
        "device_token_ring",
        KernelBufferRole::InOut,
        BlockKind::TokenState,
        DType::U32,
        MemoryTier::Vram,
        4096,
    )?;
    let logits = KernelBufferContract::new(
        "device_logits",
        KernelBufferRole::Output,
        BlockKind::Logits,
        DType::F32,
        MemoryTier::Vram,
        4096,
    )?;
    let contract = KernelContract::new(
        "synthetic_decode",
        KernelContractKind::DecodeGraph,
        bounds,
        vec![token_ring, logits],
    )?;
    contract.require_decode_ready()?;

    Ok(KernelContractProbeSummary {
        status: KernelContractProbeStatus::Ok,
        contract_count: 1,
        buffer_count: contract.buffers.len(),
        device_resident_buffers: contract
            .buffers
            .iter()
            .filter(|buffer| buffer.requires_device_residency())
            .count(),
        hot_path_allocation_allowed: contract.hot_path_allocation_allowed,
        max_grid_blocks: contract.launch_bounds.max_grid_blocks,
        max_threads_per_block: contract.launch_bounds.max_threads_per_block,
    })
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_accepts_device_resident_decode_buffer() {
        let bounds = LaunchBounds::new(64, 256).unwrap();
        let token_ring = KernelBufferContract::new(
            "device_token_ring",
            KernelBufferRole::InOut,
            BlockKind::TokenState,
            DType::U32,
            MemoryTier::Vram,
            4096,
        )
        .unwrap();
        let contract = KernelContract::new(
            "synthetic_decode",
            KernelContractKind::DecodeGraph,
            bounds,
            vec![token_ring],
        )
        .unwrap();

        assert!(contract.require_decode_ready().is_ok());
        assert_eq!(contract.buffers[0].name, "device_token_ring");
    }

    #[test]
    fn contract_rejects_hot_path_allocation() {
        let bounds = LaunchBounds::new(1, 32).unwrap();
        let scratch = KernelBufferContract::new(
            "scratch",
            KernelBufferRole::Scratch,
            BlockKind::Workspace,
            DType::U8,
            MemoryTier::Vram,
            1024,
        )
        .unwrap();
        let contract = KernelContract::new(
            "decode_with_alloc",
            KernelContractKind::DecodeGraph,
            bounds,
            vec![scratch],
        )
        .unwrap()
        .with_hot_path_allocation_allowed(true);

        assert!(contract.require_decode_ready().is_err());
    }

    #[test]
    fn contract_rejects_host_only_decode_buffers() {
        let bounds = LaunchBounds::new(1, 32).unwrap();
        let host_buffer = KernelBufferContract::new(
            "host_observation",
            KernelBufferRole::Output,
            BlockKind::TokenState,
            DType::U32,
            MemoryTier::Dram,
            4,
        )
        .unwrap();
        let contract = KernelContract::new(
            "host_only_decode",
            KernelContractKind::DecodeGraph,
            bounds,
            vec![host_buffer],
        )
        .unwrap();

        assert!(contract.require_decode_ready().is_err());
    }

    #[test]
    fn launch_bounds_reject_zero_dimensions() {
        assert!(LaunchBounds::new(0, 32).is_err());
        assert!(LaunchBounds::new(1, 0).is_err());
    }

    #[test]
    fn kernel_contract_probe_reports_decode_contract() {
        let summary = kernel_contract_probe().unwrap();

        assert_eq!(summary.status, KernelContractProbeStatus::Ok);
        assert_eq!(summary.contract_count, 1);
        assert_eq!(summary.buffer_count, 2);
        assert_eq!(summary.device_resident_buffers, 2);
        assert!(!summary.hot_path_allocation_allowed);
        assert!(summary.to_json().contains("\"status\":\"ok\""));
    }

    #[test]
    fn registry_resolves_direct_cuda_contract_by_dtype_and_architecture() {
        let registry = bootstrap_registry();
        let plan = registry
            .resolve(KernelQuery::new(
                KernelOperation::DenseMatVec,
                KernelBackend::Cuda,
                DType::BF16,
                Some(89),
            ))
            .unwrap();

        let KernelPlan::Direct { implementation } = plan else {
            panic!("expected direct kernel implementation");
        };
        assert_eq!(implementation.name, "cuda_decode_dense_matvec_fp16_bf16");
        assert!(implementation.graph_safe);
        assert_eq!(
            implementation.exactness,
            KernelExactness::ReferenceEquivalentWithinDeclaredFpTolerance
        );
    }

    #[test]
    fn registry_selects_only_named_exact_fallbacks() {
        let registry = bootstrap_registry();
        let plan = registry
            .resolve(KernelQuery::new(
                KernelOperation::DenseMatVec,
                KernelBackend::Cuda,
                DType::F32,
                Some(89),
            ))
            .unwrap();

        let KernelPlan::Fallback {
            requested,
            fallback,
            policy,
        } = plan
        else {
            panic!("expected explicit fallback");
        };
        assert_eq!(requested.backend, KernelBackend::Cuda);
        assert_eq!(fallback.backend, KernelBackend::CpuReference);
        assert_eq!(fallback.name, "cpu_reference_dense_matvec_f32");
        assert_eq!(policy.class, KernelFallbackClass::ExactNamed);
    }

    #[test]
    fn registry_rejects_missing_contracts_without_silent_fallback() {
        let registry = bootstrap_registry();
        assert!(
            registry
                .resolve(KernelQuery::new(
                    KernelOperation::DenseMatVec,
                    KernelBackend::Hip,
                    DType::BF16,
                    Some(1100),
                ))
                .is_err()
        );
        assert!(
            registry
                .resolve(KernelQuery::new(
                    KernelOperation::GreedySample,
                    KernelBackend::Cuda,
                    DType::U32,
                    Some(70),
                ))
                .is_err()
        );
    }

    #[test]
    fn registry_rejects_declared_approximate_fallback_for_exact_runtime() {
        let registry = KernelContractRegistry::new()
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
            .with_fallback(KernelFallback {
                operation: KernelOperation::DenseMatVec,
                requested_backend: KernelBackend::Cuda,
                requested_dtype: DType::F32,
                fallback_backend: KernelBackend::CpuReference,
                fallback_dtype: DType::F32,
                name: "approximate_test_fallback",
                class: KernelFallbackClass::ApproximateNamed,
            });

        assert!(
            registry
                .resolve(KernelQuery::new(
                    KernelOperation::DenseMatVec,
                    KernelBackend::Cuda,
                    DType::F32,
                    Some(89),
                ))
                .is_err()
        );
    }

    #[test]
    fn registry_probe_reports_direct_fallback_and_rejection_counts() {
        let summary = kernel_registry_probe().unwrap();
        assert_eq!(summary.status, KernelRegistryProbeStatus::Ok);
        assert_eq!(summary.implementations, 4);
        assert_eq!(summary.fallbacks, 1);
        assert_eq!(summary.direct_plans, 1);
        assert_eq!(summary.fallback_plans, 1);
        assert_eq!(summary.rejected_plans, 1);
        assert_eq!(summary.graph_safe_direct, 1);
        assert_eq!(summary.exact_fallbacks, 1);
        assert!(summary.to_json().contains("\"rejected_plans\":1"));
    }
}
