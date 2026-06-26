use nerva_core::types::error::{NervaError, Result};

use crate::types::event::LedgerEventKind;
use crate::types::token::ledger::TokenLedger;

impl TokenLedger {
    pub fn require_satisfied_block_versions(&self) -> Result<()> {
        for dependency in &self.block_version_dependencies {
            if dependency.observed_version < dependency.required_version {
                return Err(NervaError::ResidencyViolation {
                    block_id: dependency.block_id,
                    reason: format!(
                        "block version dependency '{}' requires {}, observed {}",
                        dependency.label, dependency.required_version, dependency.observed_version
                    ),
                });
            }
        }
        Ok(())
    }

    pub fn require_zero_hot_path_allocations(&self) -> Result<()> {
        if self.hot_path_allocations == 0 {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: format!(
                    "hot path allocation counter is {}",
                    self.hot_path_allocations
                ),
            })
        }
    }

    pub fn require_classified_syncs(&self) -> Result<()> {
        for event in &self.events {
            match (event.kind, event.sync_class) {
                (LedgerEventKind::Sync, Some(_)) => {}
                (LedgerEventKind::Sync, None) => {
                    return Err(NervaError::InvalidArgument {
                        reason: format!("sync event '{}' is missing SyncClass", event.label),
                    });
                }
                (_, Some(_)) => {
                    return Err(NervaError::InvalidArgument {
                        reason: format!(
                            "non-sync event '{}' carries an invalid SyncClass",
                            event.label
                        ),
                    });
                }
                (_, None) => {}
            }
        }
        Ok(())
    }
}
