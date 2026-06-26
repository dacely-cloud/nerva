use nerva_core::types::block::ResidencyState;
use nerva_core::types::error::Result;
use nerva_core::types::memory::MemoryTier;
use nerva_memory::arena::kind::ArenaKind;
use nerva_memory::arena::set::{StaticArenaBootstrapSpec, StaticArenaSet};
use nerva_memory::registry::BlockRegistry;

use crate::engine::residency::ResidencyBudget;
use crate::engine::runtime::Runtime;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticArenaProbeSummary {
    pub device_capacity_bytes: usize,
    pub pinned_host_capacity_bytes: usize,
    pub host_capacity_bytes: usize,
    pub device_used_bytes: usize,
    pub pinned_host_used_bytes: usize,
    pub host_used_bytes: usize,
    pub bootstrap_blocks: usize,
    pub ready_blocks: usize,
    pub hot_path_rejections: u64,
    pub hot_path_allocation_attempts: u64,
    pub usage_preserved_after_rejections: bool,
}

impl StaticArenaProbeSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"device_capacity_bytes\":{},\"pinned_host_capacity_bytes\":{},\"host_capacity_bytes\":{},\"device_used_bytes\":{},\"pinned_host_used_bytes\":{},\"host_used_bytes\":{},\"bootstrap_blocks\":{},\"ready_blocks\":{},\"hot_path_rejections\":{},\"hot_path_allocation_attempts\":{},\"usage_preserved_after_rejections\":{}}}",
            self.device_capacity_bytes,
            self.pinned_host_capacity_bytes,
            self.host_capacity_bytes,
            self.device_used_bytes,
            self.pinned_host_used_bytes,
            self.host_used_bytes,
            self.bootstrap_blocks,
            self.ready_blocks,
            self.hot_path_rejections,
            self.hot_path_allocation_attempts,
            self.usage_preserved_after_rejections,
        )
    }
}

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

    pub fn static_arena_probe(&self, budget: ResidencyBudget) -> Result<StaticArenaProbeSummary> {
        let mut registry = self.block_registry(budget);
        let mut arenas = self.static_arenas(budget);
        let bootstrap = arenas
            .preallocate_decode_bootstrap(&mut registry, StaticArenaBootstrapSpec::default())?;
        let bootstrap_blocks = [
            bootstrap.device_token_state,
            bootstrap.pinned_observation,
            bootstrap.host_metadata,
        ];
        let ready_blocks = bootstrap_blocks
            .iter()
            .filter(|id| {
                registry
                    .block(**id)
                    .is_some_and(|block| block.state == ResidencyState::Ready)
            })
            .count();

        let device_used_bytes = arenas.device().used();
        let pinned_host_used_bytes = arenas.pinned_host().used();
        let host_used_bytes = arenas.host().used();

        let mut ledger = self.empty_token_ledger(0);
        let mut hot_path_rejections = 0u64;
        for (kind, name) in [
            (ArenaKind::Device, "guard-device"),
            (ArenaKind::PinnedHost, "guard-pinned-host"),
            (ArenaKind::Host, "guard-host"),
        ] {
            if arenas
                .reject_hot_path_reservation_with_ledger(kind, name, 64, 64, &mut ledger)
                .is_err()
            {
                hot_path_rejections += 1;
            }
        }
        let usage_preserved_after_rejections = arenas.device().used() == device_used_bytes
            && arenas.pinned_host().used() == pinned_host_used_bytes
            && arenas.host().used() == host_used_bytes;

        Ok(StaticArenaProbeSummary {
            device_capacity_bytes: arenas.device().capacity(),
            pinned_host_capacity_bytes: arenas.pinned_host().capacity(),
            host_capacity_bytes: arenas.host().capacity(),
            device_used_bytes,
            pinned_host_used_bytes,
            host_used_bytes,
            bootstrap_blocks: bootstrap_blocks.len(),
            ready_blocks,
            hot_path_rejections,
            hot_path_allocation_attempts: ledger.hot_path_allocations,
            usage_preserved_after_rejections,
        })
    }
}
