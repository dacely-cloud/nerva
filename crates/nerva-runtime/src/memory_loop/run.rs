use std::collections::BTreeMap;

use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::AllocationId;
use nerva_memory::registry::table::BlockRegistry;

use crate::memory_loop::planned::MemoryLoopPlan;
use crate::memory_loop::summarize::summarize_plan;
use crate::memory_loop::summary::MemoryLoopSummary;

pub fn execute_memory_loop_plan(
    registry: &mut BlockRegistry,
    plan: &MemoryLoopPlan,
) -> Result<MemoryLoopSummary> {
    if plan.tasks.len() > plan.queue_capacity {
        return Err(NervaError::InvalidArgument {
            reason: "memory loop plan exceeds queue capacity".to_string(),
        });
    }

    let mut touched_blocks = BTreeMap::new();

    for task in &plan.tasks {
        let block =
            registry
                .block(task.spec.block_id)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: format!(
                        "memory loop execution references unknown block {}",
                        task.spec.block_id.0
                    ),
                })?;
        if block.tier != task.spec.from_tier {
            return Err(NervaError::ResidencyViolation {
                block_id: block.id,
                reason: format!(
                    "memory loop execution task {} expected {:?}, observed {:?}",
                    task.task_index, task.spec.from_tier, block.tier
                ),
            });
        }
        if task.spec.kind.is_eviction() {
            registry.transition(task.spec.block_id, ResidencyState::Evicting)?;
        } else if task.spec.kind.is_prefetch_like() {
            registry.transition(task.spec.block_id, ResidencyState::Prefetching)?;
        }
        if task.spec.from_tier != task.spec.to_tier {
            registry.move_block(
                task.spec.block_id,
                task.spec.to_tier,
                AllocationId(task.spec.block_id.0 + task.task_index + 1),
                0,
            )?;
        }
        registry.mark_ready(task.spec.block_id)?;
        touched_blocks.insert(task.spec.block_id, ());
    }

    let ledger = plan.ledger.clone();
    ledger.require_classified_syncs()?;
    ledger.require_zero_hot_path_allocations()?;
    let ready_blocks = touched_blocks
        .keys()
        .filter(|block_id| {
            registry
                .block(**block_id)
                .is_some_and(|block| block.state == ResidencyState::Ready)
        })
        .count() as u64;
    Ok(summarize_plan(
        plan.queue_capacity,
        plan.max_inflight,
        &plan.tasks,
        &ledger,
        ready_blocks,
    ))
}
