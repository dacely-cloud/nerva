use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::phase::types::{PhaseHandoffApplySummary, PhaseHandoffPlan};
use crate::registry::table::BlockRegistry;

impl PhaseHandoffPlan {
    pub fn apply(
        &self,
        registry: &mut BlockRegistry,
        ledger: &mut TokenLedger,
    ) -> Result<PhaseHandoffApplySummary> {
        let mut applied_handoffs = 0u64;
        let mut version_publications = 0u64;
        let mut final_max_version = 0u64;

        for entry in &self.entries {
            let block =
                registry
                    .block_mut(entry.block_id)
                    .ok_or_else(|| NervaError::InvalidArgument {
                        reason: format!(
                            "phase handoff references unknown block {}",
                            entry.block_id.0
                        ),
                    })?;
            if block.state != ResidencyState::Ready
                || block.owner != entry.from
                || block.version != entry.version_before
            {
                return Err(NervaError::ResidencyViolation {
                    block_id: entry.block_id,
                    reason: "phase handoff plan is stale".to_string(),
                });
            }

            block.owner = ExecutionOwner::PhaseTransition;
            ledger.record_sync(
                SyncClass::PhaseHandoff,
                Some(entry.block_id),
                Some(block.tier),
                Some(block.tier),
                entry.bytes,
                entry.predicted_visible_ns,
                MetricSource::EstimatedModel,
                entry.reason,
            );
            let new_version = block.publish(entry.to);
            final_max_version = final_max_version.max(new_version);
            if new_version > entry.version_before {
                version_publications = version_publications.saturating_add(1);
            }
            applied_handoffs = applied_handoffs.saturating_add(1);
        }

        ledger.require_classified_syncs()?;
        ledger.require_zero_hot_path_allocations()?;
        Ok(PhaseHandoffApplySummary {
            applied_handoffs,
            version_publications,
            final_max_version,
        })
    }
}
