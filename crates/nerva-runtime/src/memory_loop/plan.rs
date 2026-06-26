use nerva_core::types::error::Result;
use nerva_ledger::types::token::TokenLedger;
use nerva_memory::registry::BlockRegistry;

use crate::memory_loop::ledger::record_task_plan;
use crate::memory_loop::planned::{MemoryLoopPlan, MemoryLoopTask};
use crate::memory_loop::summarize::summarize_plan;
use crate::memory_loop::types::MemoryLoopConfig;
use crate::memory_loop::validate::{initial_tiers, validate_config, validate_task};

pub fn plan_memory_loop(
    config: MemoryLoopConfig,
    registry: &BlockRegistry,
) -> Result<MemoryLoopPlan> {
    validate_config(&config)?;
    let mut simulated_tiers = initial_tiers(&config, registry)?;
    let mut ledger = TokenLedger::new(0);
    let mut tasks = Vec::with_capacity(config.tasks.len());

    for (index, spec) in config.tasks.iter().copied().enumerate() {
        validate_task(index, spec, &mut simulated_tiers, registry)?;
        let actual_visible_ns = spec.visible_after_overlap();
        record_task_plan(&mut ledger, spec, actual_visible_ns);
        tasks.push(MemoryLoopTask {
            task_index: index as u64,
            spec,
            actual_visible_ns,
        });
    }

    ledger.require_classified_syncs()?;
    ledger.require_zero_hot_path_allocations()?;
    let summary = summarize_plan(
        config.queue_capacity,
        config.max_inflight,
        &tasks,
        &ledger,
        0,
    );
    Ok(MemoryLoopPlan {
        queue_capacity: config.queue_capacity,
        max_inflight: config.max_inflight,
        tasks,
        ledger,
        summary,
    })
}
