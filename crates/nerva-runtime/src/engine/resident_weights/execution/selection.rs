use nerva_core::types::error::{NervaError, Result};
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
use nerva_ledger::types::metric::MetricSource;

use crate::engine::resident_weights::helpers::{
    estimate_cpu_fallback_weight_ns, estimate_gpu_resident_weight_ns, estimate_gpu_staged_weight_ns,
};
use crate::measurements::entry::{MeasurementEntry, MeasurementKind};
use crate::weights::block::ResidentWeightBlockRef;
use crate::weights::execution::strategy::ResidentWeightExecutionStrategy;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) struct ResidentWeightCostModel {
    cpu_kernel_per_byte_ns: u64,
    copy_per_byte_ns: u64,
}

impl ResidentWeightCostModel {
    pub(super) fn from_measurements(entries: &[MeasurementEntry]) -> Result<Self> {
        Ok(Self {
            cpu_kernel_per_byte_ns: per_byte_ns(entries, MeasurementKind::CpuKernel)?,
            copy_per_byte_ns: per_byte_ns(entries, MeasurementKind::CpuCopy)?,
        })
    }

    fn cpu_dram_ns(self, bytes: usize) -> u64 {
        self.cpu_kernel_per_byte_ns.saturating_mul(bytes as u64)
    }

    fn copy_ns(self, bytes: usize) -> u64 {
        self.copy_per_byte_ns.saturating_mul(bytes as u64)
    }

    fn gpu_resident_ns(self, bytes: usize) -> u64 {
        estimate_gpu_resident_weight_ns(bytes)
    }

    fn gpu_staged_ns(self, bytes: usize) -> u64 {
        self.copy_ns(bytes)
            .saturating_add(self.gpu_resident_ns(bytes))
    }
}

pub(super) struct ResidentWeightStepSelection {
    pub(super) strategy: ResidentWeightExecutionStrategy,
    pub(super) executor: ExecutionOwner,
    pub(super) predicted_visible_ns: u64,
    pub(super) metric_source: MetricSource,
    pub(super) kernel_name: &'static str,
    pub(super) fallback: bool,
    pub(super) reason: &'static str,
}

pub(super) fn select_resident_weight_strategy(
    registry: &KernelContractRegistry,
    entry: &ResidentWeightBlockRef,
    device: DeviceOrdinal,
    compute_capability: Option<u32>,
    costs: ResidentWeightCostModel,
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
                    predicted_visible_ns: costs.gpu_resident_ns(entry.bytes),
                    metric_source: MetricSource::EstimatedModel,
                    kernel_name: implementation.name,
                    fallback: false,
                    reason: "weight is already resident in VRAM",
                }
            } else if let Some(cpu_implementation) = cpu_direct {
                let cpu_ns = costs.cpu_dram_ns(entry.bytes);
                let staged_ns = costs.gpu_staged_ns(entry.bytes);
                if cpu_ns <= staged_ns {
                    ResidentWeightStepSelection {
                        strategy: ResidentWeightExecutionStrategy::CpuDram,
                        executor: ExecutionOwner::Cpu,
                        predicted_visible_ns: cpu_ns,
                        metric_source: MetricSource::RuntimeTimestamp,
                        kernel_name: cpu_implementation.name,
                        fallback: false,
                        reason: "CPU compute wins for DRAM-resident weight",
                    }
                } else {
                    ResidentWeightStepSelection {
                        strategy: ResidentWeightExecutionStrategy::GpuStaged,
                        executor: ExecutionOwner::Gpu(device),
                        predicted_visible_ns: staged_ns,
                        metric_source: MetricSource::EstimatedModel,
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
                    metric_source: MetricSource::EstimatedModel,
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
            metric_source: MetricSource::EstimatedModel,
            kernel_name: implementation.name,
            fallback: true,
            reason: "CUDA request selected exact named CPU fallback",
        },
    };
    Ok(selection)
}

pub(super) fn resident_weight_candidate_costs(
    bytes: usize,
    costs: ResidentWeightCostModel,
) -> Vec<CandidateCost> {
    vec![
        CandidateCost::measured("cpu-dram-runtime-table", costs.cpu_dram_ns(bytes)),
        CandidateCost::estimated("gpu-resident-model", costs.gpu_resident_ns(bytes)),
        CandidateCost::measured("gpu-staged-transfer-runtime-table", costs.copy_ns(bytes)),
        CandidateCost::estimated("gpu-staged-total-model", costs.gpu_staged_ns(bytes)),
    ]
}

fn per_byte_ns(entries: &[MeasurementEntry], kind: MeasurementKind) -> Result<u64> {
    entries
        .iter()
        .find(|entry| entry.kind == kind)
        .map(|entry| {
            entry
                .elapsed_ns
                .saturating_div((entry.bytes as u64).saturating_mul(entry.iterations).max(1))
                .max(1)
        })
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!(
                "resident weight planning missing {} measurement",
                kind.as_str()
            ),
        })
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
