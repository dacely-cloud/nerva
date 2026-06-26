use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::AllocationId;
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ResidencyDecision};
use nerva_ledger::types::metric::MetricSource;

use crate::engine::runtime::Runtime;
use crate::weights::block::ResidentWeightTable;
use crate::weights::hotset::ResidentWeightHotsetSummary;

impl Runtime {
    pub fn promote_resident_weight_hotset(
        &self,
        table: &mut ResidentWeightTable,
        max_promote_bytes: usize,
    ) -> Result<ResidentWeightHotsetSummary> {
        if max_promote_bytes == 0 {
            return Ok(ResidentWeightHotsetSummary {
                promoted_blocks: 0,
                promoted_bytes: 0,
                dram_used_bytes: table.registry.used_bytes(MemoryTier::Dram),
                vram_used_bytes: table.registry.used_bytes(MemoryTier::Vram),
                residency_decisions: 0,
                first_promoted_tensor: None,
                last_promoted_tensor: None,
                hot_path_allocations: table.ledger.hot_path_allocations,
            });
        }

        let mut promoted_blocks = 0usize;
        let mut promoted_bytes = 0usize;
        let mut first_promoted_tensor = None;
        let mut last_promoted_tensor = None;
        let decision_start = table.ledger.residency_decisions.len();

        for (index, entry) in table.entries.iter_mut().enumerate() {
            if entry.tier == MemoryTier::Vram {
                continue;
            }
            let next_promoted_bytes = promoted_bytes.checked_add(entry.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: entry.bytes,
                    reason: "resident weight hotset byte count overflow".to_string(),
                }
            })?;
            if next_promoted_bytes > max_promote_bytes {
                break;
            }
            if table
                .registry
                .remaining_bytes(MemoryTier::Vram)
                .unwrap_or(0)
                < entry.bytes
            {
                break;
            }

            let allocation = AllocationId(10_000 + index as u64);
            table.registry.move_block(
                entry.block_id,
                MemoryTier::Vram,
                allocation,
                promoted_bytes as u64,
            )?;
            table.registry.mark_ready(entry.block_id)?;
            table.ledger.record_residency_decision(ResidencyDecision {
                block_id: entry.block_id,
                old_tier: entry.tier,
                new_tier: MemoryTier::Vram,
                executor_selected: ExecutionOwner::Gpu(self.config.device),
                candidate_costs: vec![
                    CandidateCost::estimated("keep-dram", entry.bytes as u64),
                    CandidateCost::estimated("promote-vram-hotset", 0),
                ],
                reason: "promote bounded exact weight hotset to VRAM",
                predicted_overlap_ns: 0,
                actual_visible_ns: Some(0),
                metric_source: MetricSource::EstimatedModel,
            });
            entry.tier = MemoryTier::Vram;
            promoted_blocks += 1;
            promoted_bytes = next_promoted_bytes;
            if first_promoted_tensor.is_none() {
                first_promoted_tensor = Some(entry.name.clone());
            }
            last_promoted_tensor = Some(entry.name.clone());
        }

        table.ledger.require_zero_hot_path_allocations()?;

        Ok(ResidentWeightHotsetSummary {
            promoted_blocks,
            promoted_bytes,
            dram_used_bytes: table.registry.used_bytes(MemoryTier::Dram),
            vram_used_bytes: table.registry.used_bytes(MemoryTier::Vram),
            residency_decisions: (table.ledger.residency_decisions.len() - decision_start) as u64,
            first_promoted_tensor,
            last_promoted_tensor,
            hot_path_allocations: table.ledger.hot_path_allocations,
        })
    }
}
