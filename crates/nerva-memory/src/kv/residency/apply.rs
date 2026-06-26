use nerva_core::types::block::ResidencyState;
use nerva_core::types::error::Result;
use nerva_core::types::id::AllocationId;

use crate::kv::residency::types::{KvResidencyAction, KvResidencyPlan};
use crate::registry::BlockRegistry;

impl KvResidencyPlan {
    pub fn apply(&self, registry: &mut BlockRegistry) -> Result<()> {
        for entry in &self.entries {
            if entry.changes_tier() {
                registry.move_block(
                    entry.block_id,
                    entry.new_tier,
                    AllocationId(entry.block_id.0),
                    0,
                )?;
            }
            match entry.action {
                KvResidencyAction::KeepHot
                | KvResidencyAction::PrefetchToHot
                | KvResidencyAction::KeepWarm => registry.mark_ready(entry.block_id)?,
                KvResidencyAction::DemoteToWarm => {
                    registry.transition(entry.block_id, ResidencyState::Draining)?
                }
                KvResidencyAction::EvictCold => {
                    registry.transition(entry.block_id, ResidencyState::Evicting)?
                }
            }
        }
        Ok(())
    }
}
