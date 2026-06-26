use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::decision::BlockVersionDependency;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::resident_weights::execution::events::{
    record_cpu_dram_step, record_cpu_exact_fallback_step, record_gpu_resident_step,
    record_gpu_staged_step,
};
use crate::engine::runtime::Runtime;
use crate::weights::block::ResidentWeightTable;
use crate::weights::execution::plan::ResidentWeightExecutionPlan;
use crate::weights::execution::run_summary::ResidentWeightExecutionRunSummary;
use crate::weights::execution::strategy::ResidentWeightExecutionStrategy;

impl Runtime {
    pub fn execute_resident_weight_execution_plan(
        &self,
        table: &ResidentWeightTable,
        plan: &ResidentWeightExecutionPlan,
    ) -> Result<ResidentWeightExecutionRunSummary> {
        if plan.steps.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "resident weight execution run has no steps".to_string(),
            });
        }

        let mut ledger = TokenLedger::new(0);
        let mut total_weight_bytes = 0usize;
        let mut gpu_resident_steps = 0u64;
        let mut gpu_staged_steps = 0u64;
        let mut fallback_steps = 0u64;

        for step in &plan.steps {
            let block =
                table
                    .registry
                    .block(step.block_id)
                    .ok_or_else(|| NervaError::InvalidArgument {
                        reason: format!(
                            "execution step references unknown block {}",
                            step.block_id.0
                        ),
                    })?;
            if block.kind != BlockKind::Weight || block.bytes != step.bytes {
                return Err(NervaError::InvalidArgument {
                    reason: format!("execution step {} block metadata drifted", step.step_index),
                });
            }
            if block.state != ResidencyState::Ready {
                return Err(NervaError::InvalidArgument {
                    reason: format!("execution step {} block is not Ready", step.step_index),
                });
            }
            ledger.record_block_version_dependency(BlockVersionDependency {
                block_id: step.block_id,
                required_version: step.block_version,
                observed_version: block.version,
                label: "resident_weight_execution_run",
            });
            ledger.require_satisfied_block_versions()?;
            let entry = table
                .entries
                .iter()
                .find(|entry| entry.block_id == step.block_id)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: format!("execution step {} has no table entry", step.step_index),
                })?;
            if entry.name != step.name || entry.tier != block.tier {
                return Err(NervaError::InvalidArgument {
                    reason: format!("execution step {} table entry drifted", step.step_index),
                });
            }

            total_weight_bytes = total_weight_bytes.checked_add(step.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: step.bytes,
                    reason: "resident weight execution run byte count overflow".to_string(),
                }
            })?;

            match step.strategy {
                ResidentWeightExecutionStrategy::CpuDram => {
                    if block.tier != MemoryTier::Dram {
                        return Err(NervaError::InvalidArgument {
                            reason: format!(
                                "CPU DRAM step {} is not DRAM-resident",
                                step.step_index
                            ),
                        });
                    }
                    record_cpu_dram_step(&mut ledger, step);
                }
                ResidentWeightExecutionStrategy::GpuResident => {
                    if block.tier != MemoryTier::Vram {
                        return Err(NervaError::InvalidArgument {
                            reason: format!(
                                "GPU resident step {} is not VRAM-resident",
                                step.step_index
                            ),
                        });
                    }
                    gpu_resident_steps += 1;
                    record_gpu_resident_step(&mut ledger, step);
                }
                ResidentWeightExecutionStrategy::GpuStaged => {
                    if block.tier == MemoryTier::Vram {
                        return Err(NervaError::InvalidArgument {
                            reason: format!(
                                "GPU staged step {} is already VRAM-resident",
                                step.step_index
                            ),
                        });
                    }
                    gpu_staged_steps += 1;
                    record_gpu_staged_step(&mut ledger, step, block.tier);
                }
                ResidentWeightExecutionStrategy::CpuExactFallback => {
                    fallback_steps += 1;
                    record_cpu_exact_fallback_step(&mut ledger, step, block.tier);
                }
            }
        }

        if total_weight_bytes != plan.total_weight_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "resident weight execution run bytes do not match plan".to_string(),
            });
        }
        ledger.require_zero_hot_path_allocations()?;

        Ok(ResidentWeightExecutionRunSummary {
            steps: plan.steps.len(),
            total_weight_bytes,
            total_latency_ns: ledger.total_latency_ns(),
            cpu_events: ledger.event_count(LedgerEventKind::CpuActivity),
            device_events: ledger.event_count(LedgerEventKind::DeviceActivity),
            copy_events: ledger.event_count(LedgerEventKind::Copy),
            gpu_resident_steps,
            gpu_staged_steps,
            fallback_steps,
            fallback_decisions: ledger.fallback_count(),
            block_version_dependencies: ledger.block_version_dependencies.len() as u64,
            hot_path_allocations: ledger.hot_path_allocations,
            ledger,
        })
    }
}
