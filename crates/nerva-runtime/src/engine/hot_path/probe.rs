use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::event::LedgerEventKind;
use nerva_memory::arena::kind::ArenaKind;
use nerva_memory::arena::set::StaticArenaBootstrapSpec;

use crate::engine::hot_path::guard::{HotPathGuard, allocation_event_count};
use crate::engine::hot_path::status::HotPathGuardStatus;
use crate::engine::hot_path::summary::HotPathGuardSummary;
use crate::engine::residency::ResidencyBudget;
use crate::engine::runtime::Runtime;

impl Runtime {
    pub fn run_hot_path_guard_probe(&self, budget: ResidencyBudget) -> Result<HotPathGuardSummary> {
        let mut registry = self.block_registry(budget);
        let mut arenas = self.static_arenas(budget);
        let bootstrap = arenas
            .preallocate_decode_bootstrap(&mut registry, StaticArenaBootstrapSpec::default())?;
        for block_id in [
            bootstrap.device_token_state,
            bootstrap.pinned_observation,
            bootstrap.host_metadata,
        ] {
            let block = registry
                .block(block_id)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: format!("bootstrap block {} is missing", block_id.0),
                })?;
            if block.state != ResidencyState::Ready {
                return Err(NervaError::InvalidArgument {
                    reason: format!("bootstrap block {} is not ready", block_id.0),
                });
            }
        }

        let baseline_device_used = arenas.device().used();
        let baseline_pinned_used = arenas.pinned_host().used();
        let baseline_host_used = arenas.host().used();
        let mut guard = HotPathGuard::new(0);

        {
            let clean_scope = guard.enter("clean_decode_hot_path")?;
            let _ = clean_scope.label();
        }
        let clean_scope_allocation_attempts = allocation_event_count(&guard);

        let mut deliberate_rejections = 0u64;
        {
            let mut violation_scope = guard.enter("deliberate_hot_path_violation_check")?;
            for (kind, name) in [
                (ArenaKind::Device, "hot-path-device-reserve"),
                (ArenaKind::PinnedHost, "hot-path-pinned-reserve"),
                (ArenaKind::Host, "hot-path-host-reserve"),
            ] {
                if violation_scope
                    .reject_arena_reservation(&mut arenas, kind, name, 64, 64)
                    .is_err()
                {
                    deliberate_rejections += 1;
                }
            }
        }

        let usage_preserved_after_rejections = guard.usage_preserved_after_rejections()
            && arenas.device().used() == baseline_device_used
            && arenas.pinned_host().used() == baseline_pinned_used
            && arenas.host().used() == baseline_host_used;
        let ledger_allocation_events = guard.ledger().event_count(LedgerEventKind::Allocation);
        let status = if clean_scope_allocation_attempts == 0
            && deliberate_rejections == guard.forbidden_allocation_attempts()
            && ledger_allocation_events == guard.ledger().hot_path_allocations
            && guard.active_scopes() == 0
            && usage_preserved_after_rejections
        {
            HotPathGuardStatus::Ok
        } else {
            HotPathGuardStatus::Failed
        };

        Ok(HotPathGuardSummary {
            status,
            token_index: guard.ledger().token_index,
            entered_scopes: guard.entered_scopes(),
            exited_scopes: guard.exited_scopes(),
            active_scopes_after_probe: guard.active_scopes(),
            clean_scope_allocation_attempts,
            deliberate_allocation_attempts: guard.forbidden_allocation_attempts(),
            deliberate_rejections,
            ledger_allocation_events,
            ledger_hot_path_allocations: guard.ledger().hot_path_allocations,
            attempted_bytes: guard.attempted_bytes(),
            release_to_system_calls: guard.release_to_system_calls(),
            usage_preserved_after_rejections,
            error: None,
        })
    }
}
