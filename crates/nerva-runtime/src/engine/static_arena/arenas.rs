use nerva_core::types::memory::tier::MemoryTier;
use nerva_memory::arena::set::static_set::StaticArenaSet;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::engine::residency::ResidencyBudget;
use crate::engine::runtime::Runtime;

impl Runtime {
    pub fn block_registry(&self, budget: ResidencyBudget) -> BlockRegistry {
        let _ = self.config;
        BlockRegistry::new([
            (MemoryTier::Vram, budget.vram_bytes),
            (MemoryTier::PinnedDram, budget.pinned_dram_bytes),
            (MemoryTier::Dram, budget.dram_bytes),
        ])
    }

    pub fn static_arenas(&self, budget: ResidencyBudget) -> StaticArenaSet {
        let _ = self.config;
        StaticArenaSet::new(
            budget.vram_bytes,
            budget.pinned_dram_bytes,
            budget.dram_bytes,
        )
    }
}
