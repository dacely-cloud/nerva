use nerva_core::types::dtype::DType;

use crate::registry::types::{
    ArchitectureRange, KernelBackend, KernelContractRegistry, KernelExactness, KernelFallback,
    KernelFallbackClass, KernelImplementation, KernelOperation,
};

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
