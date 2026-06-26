use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::allocation::AllocationId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::arena::kind::ArenaKind;
use nerva_memory::arena::set::static_set::StaticArenaSet;
use nerva_memory::kv::page::{KvPageSpec, KvPrefixKey};
use nerva_memory::kv::pool::table::KvPagePool;
use nerva_memory::kv::residency::types::{
    KvResidencyAction, KvResidencyPlanner, KvResidencyPolicy,
};

use crate::engine::kv_probe::config::KvResidencyProbeConfig;
use crate::engine::kv_probe::summary::{KvResidencyProbeStatus, KvResidencyProbeSummary};
use crate::engine::residency::ResidencyBudget;
use crate::engine::runtime::Runtime;

impl Runtime {
    pub fn run_kv_residency_probe(
        &self,
        config: KvResidencyProbeConfig,
    ) -> Result<KvResidencyProbeSummary> {
        validate_config(config)?;

        let total_bytes = config
            .page_bytes
            .checked_mul(config.pages as usize)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: config.page_bytes,
                reason: "KV residency probe byte count overflow".to_string(),
            })?;
        let mut arenas = StaticArenaSet::new(0, 0, total_bytes);
        let mut registry = self.block_registry(ResidencyBudget::new(total_bytes, 0, total_bytes));
        let mut pool = KvPagePool::preallocate(
            &mut arenas,
            &mut registry,
            config.pages,
            KvPageSpec::new(
                0,
                0,
                16,
                config.page_bytes,
                MemoryTier::Dram,
                ArenaKind::Host,
                64,
            ),
        )?;

        seed_kv_pool(&mut pool, &mut registry, config)?;

        let policy = KvResidencyPolicy::new(
            config.hot_page_limit,
            config.prefetch_distance,
            config.evict_after_idle,
        );
        let plan = KvResidencyPlanner::plan(&pool, &registry, config.current_step, policy)?;
        let mut ledger = TokenLedger::new(config.current_step);
        plan.record_to_ledger(&mut ledger);
        plan.apply(&mut registry)?;
        ledger.require_zero_hot_path_allocations()?;

        let copy_bytes = ledger
            .events
            .iter()
            .filter(|event| event.kind == LedgerEventKind::Copy)
            .map(|event| event.bytes)
            .sum();

        Ok(KvResidencyProbeSummary {
            status: KvResidencyProbeStatus::Ok,
            pages: config.pages,
            page_bytes: config.page_bytes,
            current_step: config.current_step,
            hot_page_limit: config.hot_page_limit,
            decisions: ledger.residency_decisions.len() as u64,
            keep_hot: plan.action_count(KvResidencyAction::KeepHot),
            keep_warm: plan.action_count(KvResidencyAction::KeepWarm),
            prefetches: plan.action_count(KvResidencyAction::PrefetchToHot),
            demotions: plan.action_count(KvResidencyAction::DemoteToWarm),
            evictions: plan.action_count(KvResidencyAction::EvictCold),
            copy_events: ledger.event_count(LedgerEventKind::Copy),
            prefetch_events: ledger.event_count(LedgerEventKind::Prefetch),
            eviction_events: ledger.event_count(LedgerEventKind::Eviction),
            stall_events: ledger.event_count(LedgerEventKind::Stall),
            copy_bytes,
            changed_bytes: plan.changed_bytes(),
            visible_stall_ns: ledger.latency_ns_for(LedgerEventKind::Stall),
            total_latency_ns: ledger.total_latency_ns(),
            hot_path_allocations: ledger.hot_path_allocations,
            vram_used_bytes: registry.used_bytes(MemoryTier::Vram),
            dram_used_bytes: registry.used_bytes(MemoryTier::Dram),
            error: None,
        })
    }
}

fn validate_config(config: KvResidencyProbeConfig) -> Result<()> {
    if config.pages < 4 {
        return Err(NervaError::InvalidArgument {
            reason: "KV residency probe requires at least four pages".to_string(),
        });
    }
    if config.page_bytes == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "KV residency probe page size must be non-zero".to_string(),
        });
    }
    Ok(())
}

fn seed_kv_pool(
    pool: &mut KvPagePool,
    registry: &mut nerva_memory::registry::table::registry::BlockRegistry,
    config: KvResidencyProbeConfig,
) -> Result<()> {
    let active = pool.allocate_page(0, 16, config.current_step.saturating_sub(1))?;
    pool.set_next_use(active.page_index, Some(config.current_step))?;

    let soon = pool.allocate_page(16, 16, config.current_step.saturating_sub(2))?;
    pool.cache_page(
        soon.page_index,
        KvPrefixKey {
            hash: [2; 32],
            group_id: 0,
        },
        16,
    )?;
    pool.set_next_use(soon.page_index, Some(config.current_step.saturating_add(1)))?;

    let cold = pool.allocate_page(32, 16, 0)?;
    pool.cache_page(
        cold.page_index,
        KvPrefixKey {
            hash: [3; 32],
            group_id: 0,
        },
        16,
    )?;

    let warm_vram = pool.allocate_page(48, 16, config.current_step.saturating_sub(1))?;
    pool.cache_page(
        warm_vram.page_index,
        KvPrefixKey {
            hash: [4; 32],
            group_id: 0,
        },
        16,
    )?;

    pool.release_page(soon.page_index, config.current_step.saturating_sub(2))?;
    pool.release_page(cold.page_index, 0)?;
    pool.release_page(warm_vram.page_index, config.current_step.saturating_sub(1))?;

    registry.move_block(
        warm_vram.block_id,
        MemoryTier::Vram,
        AllocationId(warm_vram.block_id.0),
        0,
    )?;
    registry.mark_ready(warm_vram.block_id)
}
