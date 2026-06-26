use nerva_core::types::{BlockKind, DType, MemoryTier};

use crate::contract::{
    KernelBufferContract, KernelBufferRole, KernelContract, KernelContractKind,
    KernelContractProbeStatus, LaunchBounds, kernel_contract_probe,
};
use crate::registry::{
    KernelBackend, KernelContractRegistry, KernelExactness, KernelFallback, KernelFallbackClass,
    KernelImplementation, KernelOperation, KernelPlan, KernelQuery, KernelRegistryProbeStatus,
    bootstrap_registry, kernel_registry_probe,
};

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
            dtypes: &[DType::F32],
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
