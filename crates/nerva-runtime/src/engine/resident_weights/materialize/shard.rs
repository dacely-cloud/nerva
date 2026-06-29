use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::layout::LayoutId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ResidencyDecision};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::registry::request::BlockAllocationRequest;
use nerva_model::weights::safetensors::shard::SafetensorsShardPlan;

use crate::engine::resident_weights::helpers::weight_role_layout_id;
use crate::engine::runtime::Runtime;
use crate::residency::budget::ResidencyBudget;
use crate::weights::block::{ResidentWeightBlockRef, ResidentWeightTable};

impl Runtime {
    pub fn materialize_safetensors_shard_plan(
        &self,
        plan: &SafetensorsShardPlan,
    ) -> Result<ResidentWeightTable> {
        self.materialize_safetensors_shard_plan_with_budget(
            plan,
            ResidencyBudget::new(0, 0, plan.total_weight_bytes),
        )
    }

    pub fn materialize_safetensors_shard_plan_with_budget(
        &self,
        plan: &SafetensorsShardPlan,
        budget: ResidencyBudget,
    ) -> Result<ResidentWeightTable> {
        let _ = self.config;
        let mut registry = self.block_registry(budget);
        let mut ledger = TokenLedger::new(0);
        let mut entries = Vec::with_capacity(plan.entries.len());
        let mut materialized_bytes = 0usize;

        for entry in &plan.entries {
            let block_id = registry.allocate(
                BlockAllocationRequest::new(BlockKind::Weight, entry.tier, entry.bytes)
                    .with_dtype(entry.dtype)
                    .with_layout(LayoutId(weight_role_layout_id(entry.role))),
            )?;
            registry.mark_ready(block_id)?;
            materialized_bytes = materialized_bytes.checked_add(entry.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: entry.bytes,
                    reason: "resident shard-plan weight byte count overflow".to_string(),
                }
            })?;
            ledger.record_residency_decision(ResidencyDecision {
                block_id,
                old_tier: MemoryTier::Disk,
                new_tier: entry.tier,
                executor_selected: ExecutionOwner::Cpu,
                candidate_costs: vec![
                    CandidateCost::estimated("safetensors-shard-read", entry.bytes as u64),
                    CandidateCost::estimated("file-offset-begin", entry.file_offset_begin as u64),
                ],
                reason: "initialize exact sharded safetensors weight block as resident immutable backing",
                predicted_overlap_ns: 0,
                actual_visible_ns: Some(0),
                metric_source: MetricSource::EstimatedModel,
            });
            entries.push(ResidentWeightBlockRef {
                name: entry.tensor_name.clone(),
                block_id,
                role: entry.role,
                bytes: entry.bytes,
                dtype: entry.dtype,
                tier: entry.tier,
                source_shard: Some(entry.shard_file.clone()),
                file_offset_begin: Some(entry.file_offset_begin),
                file_offset_end: Some(entry.file_offset_end),
            });
        }

        if materialized_bytes != plan.total_weight_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "resident shard-plan weight byte count does not match plan".to_string(),
            });
        }
        ledger.require_zero_hot_path_allocations()?;

        Ok(ResidentWeightTable {
            registry,
            entries,
            total_weight_bytes: materialized_bytes,
            manifest_hash: plan.manifest_hash,
            ledger,
        })
    }
}
