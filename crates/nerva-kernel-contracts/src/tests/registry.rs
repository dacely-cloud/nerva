use nerva_core::types::dtype::DType;

use crate::registry::bootstrap::bootstrap_registry;
use crate::registry::probe::{kernel_registry_probe, KernelRegistryProbeStatus};
use crate::registry::types::backend::KernelBackend;
use crate::registry::types::exactness::KernelExactness;
use crate::registry::types::fallback::{KernelFallback, KernelFallbackClass};
use crate::registry::types::implementation::KernelImplementation;
use crate::registry::types::operation::KernelOperation;
use crate::registry::types::plan::KernelPlan;
use crate::registry::types::query::KernelQuery;
use crate::registry::types::registry::KernelContractRegistry;

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
fn registry_resolves_direct_cuda_fp8_e4m3_dense_matvec_contract() {
    let registry = bootstrap_registry();
    let plan = registry
        .resolve(KernelQuery::new(
            KernelOperation::DenseMatVec,
            KernelBackend::Cuda,
            DType::F8E4M3,
            Some(120),
        ))
        .unwrap();

    let KernelPlan::Direct { implementation } = plan else {
        panic!("expected direct kernel implementation");
    };
    assert_eq!(implementation.name, "cuda_decode_dense_matvec_fp8_e4m3");
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
    assert!(registry
        .resolve(KernelQuery::new(
            KernelOperation::DenseMatVec,
            KernelBackend::Hip,
            DType::BF16,
            Some(1100),
        ))
        .is_err());
    assert!(registry
        .resolve(KernelQuery::new(
            KernelOperation::GreedySample,
            KernelBackend::Cuda,
            DType::U32,
            Some(70),
        ))
        .is_err());
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

    assert!(registry
        .resolve(KernelQuery::new(
            KernelOperation::DenseMatVec,
            KernelBackend::Cuda,
            DType::F32,
            Some(89),
        ))
        .is_err());
}

#[test]
fn registry_probe_reports_direct_fallback_and_rejection_counts() {
    let summary = kernel_registry_probe().unwrap();
    assert_eq!(summary.status, KernelRegistryProbeStatus::Ok);
    assert_eq!(summary.implementations, 5);
    assert_eq!(summary.fallbacks, 1);
    assert_eq!(summary.direct_plans, 1);
    assert_eq!(summary.fallback_plans, 1);
    assert_eq!(summary.rejected_plans, 1);
    assert_eq!(summary.graph_safe_direct, 1);
    assert_eq!(summary.exact_fallbacks, 1);
    assert!(summary.to_json().contains("\"rejected_plans\":1"));
}
