use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::decision::BlockVersionDependency;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::execution::types::TransactionOperation;

pub fn validate_and_record_block_uses(
    ledger: &mut TokenLedger,
    registry: &BlockRegistry,
    operation: &TransactionOperation,
) -> Result<()> {
    for block_use in &operation.block_uses {
        let block =
            registry
                .block(block_use.block_id)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: format!(
                        "transaction operation {} references unknown block {}",
                        operation.name, block_use.block_id.0
                    ),
                })?;
        if block.tier != block_use.expected_tier {
            return Err(NervaError::ResidencyViolation {
                block_id: block.id,
                reason: format!(
                    "transaction block use '{}' expected {:?}, observed {:?}",
                    block_use.label, block_use.expected_tier, block.tier
                ),
            });
        }
        block.require_ready(block.authoritative_copy, block_use.required_version)?;
        ledger.record_block_version_dependency(BlockVersionDependency {
            block_id: block.id,
            required_version: block_use.required_version,
            observed_version: block.version,
            label: block_use.label,
        });
        if block_use.access.writes() && block.owner != block_use.owner {
            ledger.record_sync(
                SyncClass::PhaseHandoff,
                Some(block.id),
                Some(block.tier),
                Some(block_use.expected_tier),
                block.bytes,
                1,
                MetricSource::EstimatedModel,
                "transaction_phase_handoff",
            );
        }
    }
    Ok(())
}
