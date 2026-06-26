use std::collections::BTreeMap;

use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::ResidentBlockId;
use nerva_core::types::memory::MemoryTier;
use nerva_memory::registry::table::BlockRegistry;

use crate::memory_loop::types::{MemoryLoopConfig, MemoryLoopTaskKind, MemoryLoopTaskSpec};

pub(crate) fn validate_config(config: &MemoryLoopConfig) -> Result<()> {
    if config.queue_capacity == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "memory loop queue capacity must be non-zero".to_string(),
        });
    }
    if config.max_inflight == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "memory loop max_inflight must be non-zero".to_string(),
        });
    }
    if config.max_inflight > config.queue_capacity {
        return Err(NervaError::InvalidArgument {
            reason: "memory loop max_inflight cannot exceed queue capacity".to_string(),
        });
    }
    if config.tasks.is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: "memory loop requires at least one task".to_string(),
        });
    }
    if config.tasks.len() > config.queue_capacity {
        return Err(NervaError::InvalidArgument {
            reason: "memory loop task list exceeds bounded queue capacity".to_string(),
        });
    }
    Ok(())
}

pub(crate) fn initial_tiers(
    config: &MemoryLoopConfig,
    registry: &BlockRegistry,
) -> Result<BTreeMap<ResidentBlockId, MemoryTier>> {
    let mut tiers = BTreeMap::new();
    for task in &config.tasks {
        let block = registry
            .block(task.block_id)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!(
                    "memory loop task references unknown block {}",
                    task.block_id.0
                ),
            })?;
        tiers.entry(task.block_id).or_insert(block.tier);
    }
    Ok(tiers)
}

pub(crate) fn validate_task(
    index: usize,
    spec: MemoryLoopTaskSpec,
    simulated_tiers: &mut BTreeMap<ResidentBlockId, MemoryTier>,
    registry: &BlockRegistry,
) -> Result<()> {
    if spec.bytes == 0 {
        return Err(NervaError::InvalidArgument {
            reason: format!("memory loop task {index} has zero bytes"),
        });
    }
    if spec.predicted_visible_ns == 0 {
        return Err(NervaError::InvalidArgument {
            reason: format!("memory loop task {index} has zero predicted cost"),
        });
    }
    if spec.kind == MemoryLoopTaskKind::PrepareTransportBuffer && spec.from_tier != spec.to_tier {
        return Err(NervaError::InvalidArgument {
            reason: format!("memory loop transport task {index} must stay in one tier"),
        });
    }

    let block = registry
        .block(spec.block_id)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!(
                "memory loop task references unknown block {}",
                spec.block_id.0
            ),
        })?;
    if spec.bytes > block.bytes {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "memory loop task {index} bytes exceed block {} size",
                spec.block_id.0
            ),
        });
    }
    let Some(current_tier) = simulated_tiers.get_mut(&spec.block_id) else {
        return Err(NervaError::InvalidArgument {
            reason: format!("memory loop task {index} has no simulated tier"),
        });
    };
    if *current_tier != spec.from_tier {
        return Err(NervaError::ResidencyViolation {
            block_id: spec.block_id,
            reason: format!(
                "memory loop task {index} expected {:?}, simulated {:?}",
                spec.from_tier, current_tier
            ),
        });
    }
    *current_tier = spec.to_tier;
    Ok(())
}
