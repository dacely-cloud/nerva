use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::registry::table::registry::BlockRegistry;

use crate::execution::plan::block_uses::validate_and_record_block_uses;
use crate::execution::plan::decisions::record_execution_decision;
use crate::execution::plan::events::record_operation_events;
use crate::execution::plan::summary::summarize_transaction;
use crate::execution::plan::validation::validate_operation;
use crate::execution::summary::ExecutionTransactionSummary;
use crate::execution::types::ExecutionTransactionSpec;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionTransactionPlan {
    pub spec: ExecutionTransactionSpec,
    pub ledger: TokenLedger,
    pub summary: ExecutionTransactionSummary,
}

pub fn plan_execution_transaction(
    spec: ExecutionTransactionSpec,
    registry: &BlockRegistry,
) -> Result<ExecutionTransactionPlan> {
    if spec.operations.is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: "execution transaction requires at least one operation".to_string(),
        });
    }

    let mut ledger = TokenLedger::new(spec.token_index);
    let mut clock_ns = 0u64;

    for operation in &spec.operations {
        validate_operation(operation)?;
        record_execution_decision(&mut ledger, operation);
        record_operation_events(&mut ledger, operation, clock_ns)?;
        validate_and_record_block_uses(&mut ledger, registry, operation)?;
        clock_ns = clock_ns.saturating_add(operation.predicted_visible_ns);
    }

    ledger.require_satisfied_block_versions()?;
    ledger.require_classified_syncs()?;
    ledger.require_zero_hot_path_allocations()?;

    let summary = summarize_transaction(&spec, &ledger)?;
    Ok(ExecutionTransactionPlan {
        spec,
        ledger,
        summary,
    })
}
