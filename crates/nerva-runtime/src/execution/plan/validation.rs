use nerva_core::types::error::{NervaError, Result};

use crate::execution::types::TransactionOperation;

pub fn validate_operation(operation: &TransactionOperation) -> Result<()> {
    if operation.name.is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: "transaction operation name must be non-empty".to_string(),
        });
    }
    if operation.predicted_visible_ns == 0 {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "transaction operation {} must have non-zero predicted cost",
                operation.name
            ),
        });
    }
    if operation.block_uses.is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "transaction operation {} must declare block uses",
                operation.name
            ),
        });
    }
    Ok(())
}
