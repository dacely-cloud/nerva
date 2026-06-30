use nerva_core::types::dtype::DType;

use crate::registry::types::architecture::ArchitectureRange;
use crate::registry::types::backend::KernelBackend;
use crate::registry::types::exactness::KernelExactness;
use crate::registry::types::fallback::{KernelFallback, KernelFallbackClass};
use crate::registry::types::implementation::KernelImplementation;
use crate::registry::types::operation::KernelOperation;
use crate::registry::types::registry::KernelContractRegistry;

static DTYPE_F32: &[DType] = &[DType::F32];
static DTYPE_FP16_BF16: &[DType] = &[DType::F16, DType::BF16];
static DTYPE_F8_E4M3: &[DType] = &[DType::F8E4M3];
static DTYPE_MXFP4_E2M1: &[DType] = &[DType::F4E2M1];
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
            name: "cuda_decode_dense_matvec_fp8_e4m3",
            operation: KernelOperation::DenseMatVec,
            backend: KernelBackend::Cuda,
            architecture: Some(ArchitectureRange::new(75, 121)),
            dtypes: DTYPE_F8_E4M3,
            graph_safe: true,
            deterministic: true,
            exactness: KernelExactness::ReferenceEquivalentWithinDeclaredFpTolerance,
        })
        .with_implementation(KernelImplementation {
            name: "cuda_deepseek_fp8_e4m3_e8m0_block_dequant",
            operation: KernelOperation::BlockDequant,
            backend: KernelBackend::Cuda,
            architecture: Some(ArchitectureRange::new(75, 121)),
            dtypes: DTYPE_F8_E4M3,
            graph_safe: true,
            deterministic: true,
            exactness: KernelExactness::ReferenceEquivalentWithinDeclaredFpTolerance,
        })
        .with_implementation(KernelImplementation {
            name: "cuda_deepseek_mxfp4_e2m1_e8m0_block_dequant",
            operation: KernelOperation::BlockDequant,
            backend: KernelBackend::Cuda,
            architecture: Some(ArchitectureRange::new(75, 121)),
            dtypes: DTYPE_MXFP4_E2M1,
            graph_safe: true,
            deterministic: true,
            exactness: KernelExactness::ReferenceEquivalentWithinDeclaredFpTolerance,
        })
        .with_implementation(KernelImplementation {
            name: "cuda_deepseek_megamoe_fp8_fp4_experts",
            operation: KernelOperation::SparseMoeExpert,
            backend: KernelBackend::Cuda,
            architecture: Some(ArchitectureRange::new(75, 121)),
            dtypes: DTYPE_MXFP4_E2M1,
            graph_safe: true,
            deterministic: true,
            exactness: KernelExactness::ReferenceEquivalentWithinDeclaredFpTolerance,
        })
        .with_implementation(KernelImplementation {
            name: "cuda_deepseek_fp8_ds_mla_kv_append",
            operation: KernelOperation::KvAppend,
            backend: KernelBackend::Cuda,
            architecture: Some(ArchitectureRange::new(75, 121)),
            dtypes: DTYPE_F8_E4M3,
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
