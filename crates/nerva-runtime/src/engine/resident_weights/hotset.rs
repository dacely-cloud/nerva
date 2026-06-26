use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::allocation::AllocationId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ResidencyDecision};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::runtime::Runtime;
use crate::weights::block::ResidentWeightTable;
use crate::weights::hotset::ResidentWeightHotsetSummary;

impl Runtime {
    pub fn promote_resident_weight_hotset(
        &self,
        table: &mut ResidentWeightTable,
        max_promote_bytes: usize,
    ) -> Result<ResidentWeightHotsetSummary> {
        let mut considered_blocks = 0usize;
        let mut promoted_blocks = 0usize;
        let mut promoted_bytes = 0usize;
        let mut kept_dram_blocks = 0usize;
        let mut budget_limited_blocks = 0usize;
        let mut capacity_limited_blocks = 0usize;
        let mut already_hot_blocks = 0usize;
        let mut first_promoted_tensor = None;
        let mut last_promoted_tensor = None;
        let mut last_keep_reason = None;
        let mut hotset_closed = false;
        let decision_start = table.ledger.residency_decisions.len();

        for (index, entry) in table.entries.iter_mut().enumerate() {
            if entry.tier == MemoryTier::Vram {
                already_hot_blocks += 1;
                continue;
            }
            considered_blocks += 1;
            if hotset_closed {
                kept_dram_blocks += 1;
                last_keep_reason = Some("keep weight in DRAM outside bounded hotset prefix");
                record_keep_dram_decision(
                    &mut table.ledger,
                    entry.block_id,
                    entry.tier,
                    entry.bytes,
                    last_keep_reason.unwrap(),
                );
                continue;
            }
            let next_promoted_bytes = promoted_bytes.checked_add(entry.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: entry.bytes,
                    reason: "resident weight hotset byte count overflow".to_string(),
                }
            })?;
            if next_promoted_bytes > max_promote_bytes {
                hotset_closed = true;
                kept_dram_blocks += 1;
                budget_limited_blocks += 1;
                last_keep_reason =
                    Some("keep weight in DRAM because hotset byte budget is exhausted");
                record_keep_dram_decision(
                    &mut table.ledger,
                    entry.block_id,
                    entry.tier,
                    entry.bytes,
                    last_keep_reason.unwrap(),
                );
                continue;
            }
            if table
                .registry
                .remaining_bytes(MemoryTier::Vram)
                .unwrap_or(0)
                < entry.bytes
            {
                hotset_closed = true;
                kept_dram_blocks += 1;
                capacity_limited_blocks += 1;
                last_keep_reason =
                    Some("keep weight in DRAM because VRAM hotset capacity is exhausted");
                record_keep_dram_decision(
                    &mut table.ledger,
                    entry.block_id,
                    entry.tier,
                    entry.bytes,
                    last_keep_reason.unwrap(),
                );
                continue;
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
            considered_blocks,
            promoted_blocks,
            promoted_bytes,
            kept_dram_blocks,
            budget_limited_blocks,
            capacity_limited_blocks,
            already_hot_blocks,
            dram_used_bytes: table.registry.used_bytes(MemoryTier::Dram),
            vram_used_bytes: table.registry.used_bytes(MemoryTier::Vram),
            residency_decisions: (table.ledger.residency_decisions.len() - decision_start) as u64,
            first_promoted_tensor,
            last_promoted_tensor,
            last_keep_reason,
            hot_path_allocations: table.ledger.hot_path_allocations,
        })
    }
}

fn record_keep_dram_decision(
    ledger: &mut TokenLedger,
    block_id: nerva_core::types::id::block::ResidentBlockId,
    tier: MemoryTier,
    bytes: usize,
    reason: &'static str,
) {
    ledger.record_residency_decision(ResidencyDecision {
        block_id,
        old_tier: tier,
        new_tier: tier,
        executor_selected: ExecutionOwner::Cpu,
        candidate_costs: vec![
            CandidateCost::estimated("keep-dram", bytes as u64),
            CandidateCost::estimated("promote-vram-hotset", bytes as u64 + 1),
        ],
        reason,
        predicted_overlap_ns: 0,
        actual_visible_ns: Some(bytes as u64),
        metric_source: MetricSource::EstimatedModel,
    });
}
