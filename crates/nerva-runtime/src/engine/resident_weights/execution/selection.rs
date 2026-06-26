use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_kernel_contracts::registry::types::backend::KernelBackend;
use nerva_kernel_contracts::registry::types::implementation::KernelImplementation;
use nerva_kernel_contracts::registry::types::operation::KernelOperation;
use nerva_kernel_contracts::registry::types::plan::KernelPlan;
use nerva_kernel_contracts::registry::types::query::KernelQuery;
use nerva_kernel_contracts::registry::types::registry::KernelContractRegistry;
use nerva_ledger::types::decision::CandidateCost;

use crate::engine::resident_weights::helpers::{
    estimate_cpu_dram_weight_ns, estimate_cpu_fallback_weight_ns, estimate_gpu_resident_weight_ns,
    estimate_gpu_staged_weight_ns,
};
use crate::weights::block::ResidentWeightBlockRef;
use crate::weights::execution::ResidentWeightExecutionStrategy;

pub(super) struct ResidentWeightStepSelection {
    pub(super) strategy: ResidentWeightExecutionStrategy,
    pub(super) executor: ExecutionOwner,
    pub(super) predicted_visible_ns: u64,
    pub(super) kernel_name: &'static str,
    pub(super) fallback: bool,
    pub(super) reason: &'static str,
}

pub(super) fn select_resident_weight_strategy(
    registry: &KernelContractRegistry,
    entry: &ResidentWeightBlockRef,
    device: DeviceOrdinal,
    compute_capability: Option<u32>,
) -> Result<ResidentWeightStepSelection> {
    let cuda_plan = registry.resolve(KernelQuery::new(
        KernelOperation::DenseMatVec,
        KernelBackend::Cuda,
        entry.dtype,
        compute_capability,
    ))?;
    let cpu_direct = resolve_cpu_dense_matvec(registry, entry);

    let selection = match cuda_plan {
        KernelPlan::Direct { implementation } => {
            if entry.tier == MemoryTier::Vram {
                ResidentWeightStepSelection {
                    strategy: ResidentWeightExecutionStrategy::GpuResident,
                    executor: ExecutionOwner::Gpu(device),
                    predicted_visible_ns: estimate_gpu_resident_weight_ns(entry.bytes),
                    kernel_name: implementation.name,
                    fallback: false,
                    reason: "weight is already resident in VRAM",
                }
            } else if let Some(cpu_implementation) = cpu_direct {
                let cpu_ns = estimate_cpu_dram_weight_ns(entry.bytes);
                let staged_ns = estimate_gpu_staged_weight_ns(entry.bytes);
                if cpu_ns <= staged_ns {
                    ResidentWeightStepSelection {
                        strategy: ResidentWeightExecutionStrategy::CpuDram,
                        executor: ExecutionOwner::Cpu,
                        predicted_visible_ns: cpu_ns,
                        kernel_name: cpu_implementation.name,
                        fallback: false,
                        reason: "CPU compute wins for DRAM-resident weight",
                    }
                } else {
                    ResidentWeightStepSelection {
                        strategy: ResidentWeightExecutionStrategy::GpuStaged,
                        executor: ExecutionOwner::Gpu(device),
                        predicted_visible_ns: staged_ns,
                        kernel_name: implementation.name,
                        fallback: false,
                        reason: "GPU staged compute wins despite transfer",
                    }
                }
            } else {
                ResidentWeightStepSelection {
                    strategy: ResidentWeightExecutionStrategy::GpuStaged,
                    executor: ExecutionOwner::Gpu(device),
                    predicted_visible_ns: estimate_gpu_staged_weight_ns(entry.bytes),
                    kernel_name: implementation.name,
                    fallback: false,
                    reason: "no exact CPU contract; use declared GPU staged kernel",
                }
            }
        }
        KernelPlan::Fallback {
            fallback: implementation,
            ..
        } => ResidentWeightStepSelection {
            strategy: ResidentWeightExecutionStrategy::CpuExactFallback,
            executor: ExecutionOwner::Cpu,
            predicted_visible_ns: estimate_cpu_fallback_weight_ns(entry.bytes, entry.tier),
            kernel_name: implementation.name,
            fallback: true,
            reason: "CUDA request selected exact named CPU fallback",
        },
    };
    Ok(selection)
}

pub(super) fn resident_weight_candidate_costs(bytes: usize) -> Vec<CandidateCost> {
    vec![
        CandidateCost::estimated("cpu-dram", estimate_cpu_dram_weight_ns(bytes)),
        CandidateCost::estimated("gpu-resident", estimate_gpu_resident_weight_ns(bytes)),
        CandidateCost::estimated("gpu-staged", estimate_gpu_staged_weight_ns(bytes)),
    ]
}

fn resolve_cpu_dense_matvec(
    registry: &KernelContractRegistry,
    entry: &ResidentWeightBlockRef,
) -> Option<KernelImplementation> {
    registry
        .resolve(KernelQuery::new(
            KernelOperation::DenseMatVec,
            KernelBackend::CpuReference,
            entry.dtype,
            None,
        ))
        .ok()
        .and_then(|plan| match plan {
            KernelPlan::Direct { implementation } => Some(implementation),
            KernelPlan::Fallback { .. } => None,
        })
}
