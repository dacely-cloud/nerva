use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::Result;
use nerva_memory::arena::kind::ArenaKind;
use nerva_memory::arena::set::bootstrap::StaticArenaBootstrapSpec;

use crate::engine::runtime::Runtime;
use crate::engine::static_arena::summary::StaticArenaProbeSummary;
use crate::residency::budget::ResidencyBudget;

impl Runtime {
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
