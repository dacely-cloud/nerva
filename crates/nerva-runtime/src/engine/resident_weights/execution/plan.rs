use nerva_core::types::block::{BlockKind, ResidencyState};
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_kernel_contracts::registry::bootstrap::bootstrap_registry;
use nerva_kernel_contracts::registry::{KernelBackend, KernelOperation, KernelPlan, KernelQuery};
use nerva_ledger::types::decision::{BlockVersionDependency, CandidateCost, ExecutionDecision};
use nerva_ledger::types::fallback::{FallbackClass, FallbackDecision};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::TokenLedger;

use crate::engine::resident_weights::helpers::{
    estimate_cpu_dram_weight_ns, estimate_cpu_fallback_weight_ns, estimate_gpu_resident_weight_ns,
    estimate_gpu_staged_weight_ns,
};
use crate::engine::runtime::Runtime;
use crate::weights::block::ResidentWeightTable;
use crate::weights::execution::{
    ResidentWeightExecutionPlan, ResidentWeightExecutionStep, ResidentWeightExecutionStrategy,
};

impl Runtime {
    pub fn plan_resident_weight_execution(
        &self,
        table: &ResidentWeightTable,
        max_steps: usize,
        compute_capability: Option<u32>,
    ) -> Result<ResidentWeightExecutionPlan> {
        if max_steps == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "resident weight execution max_steps must be non-zero".to_string(),
            });
        }

        let registry = bootstrap_registry();
        let mut ledger = TokenLedger::new(0);
        let mut steps = Vec::new();
        let mut total_weight_bytes = 0usize;
        let mut total_predicted_visible_ns = 0u64;
        let mut cpu_steps = 0u64;
        let mut gpu_resident_steps = 0u64;
        let mut gpu_staged_steps = 0u64;
        let mut fallback_steps = 0u64;

        for (index, entry) in table.entries.iter().take(max_steps).enumerate() {
            let block = table.registry.block(entry.block_id).ok_or_else(|| {
                NervaError::InvalidArgument {
                    reason: format!("resident weight {} references unknown block", entry.name),
                }
            })?;
            if block.kind != BlockKind::Weight
                || block.tier != entry.tier
                || block.dtype != entry.dtype
            {
                return Err(NervaError::InvalidArgument {
                    reason: format!("resident weight {} block metadata drifted", entry.name),
                });
            }
            if block.state != ResidencyState::Ready {
                return Err(NervaError::InvalidArgument {
                    reason: format!("resident weight {} is not Ready", entry.name),
                });
            }
            ledger.record_block_version_dependency(BlockVersionDependency {
                block_id: entry.block_id,
                required_version: block.version,
                observed_version: block.version,
                label: "resident_weight_execution_plan",
            });

            let cuda_plan = registry.resolve(KernelQuery::new(
                KernelOperation::DenseMatVec,
                KernelBackend::Cuda,
                entry.dtype,
                compute_capability,
            ))?;
            let cpu_direct = registry
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
                });

            let (strategy, executor, predicted_visible_ns, kernel_name, fallback, reason) =
                match cuda_plan {
                    KernelPlan::Direct { implementation } => {
                        if entry.tier == MemoryTier::Vram {
                            (
                                ResidentWeightExecutionStrategy::GpuResident,
                                ExecutionOwner::Gpu(self.config.device),
                                estimate_gpu_resident_weight_ns(entry.bytes),
                                implementation.name,
                                false,
                                "weight is already resident in VRAM",
                            )
                        } else if let Some(cpu_implementation) = cpu_direct {
                            let cpu_ns = estimate_cpu_dram_weight_ns(entry.bytes);
                            let staged_ns = estimate_gpu_staged_weight_ns(entry.bytes);
                            if cpu_ns <= staged_ns {
                                (
                                    ResidentWeightExecutionStrategy::CpuDram,
                                    ExecutionOwner::Cpu,
                                    cpu_ns,
                                    cpu_implementation.name,
                                    false,
                                    "CPU compute wins for DRAM-resident weight",
                                )
                            } else {
                                (
                                    ResidentWeightExecutionStrategy::GpuStaged,
                                    ExecutionOwner::Gpu(self.config.device),
                                    staged_ns,
                                    implementation.name,
                                    false,
                                    "GPU staged compute wins despite transfer",
                                )
                            }
                        } else {
                            (
                                ResidentWeightExecutionStrategy::GpuStaged,
                                ExecutionOwner::Gpu(self.config.device),
                                estimate_gpu_staged_weight_ns(entry.bytes),
                                implementation.name,
                                false,
                                "no exact CPU contract; use declared GPU staged kernel",
                            )
                        }
                    }
                    KernelPlan::Fallback {
                        fallback: implementation,
                        ..
                    } => (
                        ResidentWeightExecutionStrategy::CpuExactFallback,
                        ExecutionOwner::Cpu,
                        estimate_cpu_fallback_weight_ns(entry.bytes, entry.tier),
                        implementation.name,
                        true,
                        "CUDA request selected exact named CPU fallback",
                    ),
                };

            total_weight_bytes = total_weight_bytes.checked_add(entry.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: entry.bytes,
                    reason: "resident weight execution byte count overflow".to_string(),
                }
            })?;
            total_predicted_visible_ns = total_predicted_visible_ns
                .checked_add(predicted_visible_ns)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes: predicted_visible_ns as usize,
                    reason: "resident weight execution visible cost overflow".to_string(),
                })?;
            match strategy {
                ResidentWeightExecutionStrategy::CpuDram
                | ResidentWeightExecutionStrategy::CpuExactFallback => cpu_steps += 1,
                ResidentWeightExecutionStrategy::GpuResident => gpu_resident_steps += 1,
                ResidentWeightExecutionStrategy::GpuStaged => gpu_staged_steps += 1,
            }
            fallback_steps += u64::from(fallback);

            ledger.record_execution_decision(ExecutionDecision {
                operation: "resident_weight_dense_matvec",
                executor_selected: executor,
                candidate_costs: vec![
                    CandidateCost::estimated("cpu-dram", estimate_cpu_dram_weight_ns(entry.bytes)),
                    CandidateCost::estimated(
                        "gpu-resident",
                        estimate_gpu_resident_weight_ns(entry.bytes),
                    ),
                    CandidateCost::estimated(
                        "gpu-staged",
                        estimate_gpu_staged_weight_ns(entry.bytes),
                    ),
                ],
                reason,
                predicted_visible_ns,
                actual_visible_ns: Some(0),
                metric_source: MetricSource::EstimatedModel,
            });
            if fallback {
                ledger.record_fallback_decision(FallbackDecision {
                    label: "resident_weight_exact_cpu_fallback",
                    class: FallbackClass::ExactNamed,
                    requested: "cuda_dense_matvec",
                    selected: kernel_name,
                    reason,
                    visible_ns: Some(predicted_visible_ns),
                    metric_source: MetricSource::EstimatedModel,
                });
            }
            steps.push(ResidentWeightExecutionStep {
                step_index: index as u64,
                block_id: entry.block_id,
                name: entry.name.clone(),
                strategy,
                executor,
                bytes: entry.bytes,
                block_version: block.version,
                predicted_visible_ns,
                kernel_name,
                fallback,
            });
        }

        if steps.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "resident weight execution has no steps".to_string(),
            });
        }
        ledger.require_zero_hot_path_allocations()?;
        ledger.require_satisfied_block_versions()?;

        let first_tensor = steps.first().map(|step| step.name.clone());
        let last_tensor = steps.last().map(|step| step.name.clone());
        Ok(ResidentWeightExecutionPlan {
            steps,
            total_weight_bytes,
            total_predicted_visible_ns,
            cpu_steps,
            gpu_resident_steps,
            gpu_staged_steps,
            fallback_steps,
            fallback_decisions: ledger.fallback_count(),
            block_version_dependencies: ledger.block_version_dependencies.len() as u64,
            first_tensor,
            last_tensor,
            ledger,
        })
    }
}
