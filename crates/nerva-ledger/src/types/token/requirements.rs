use nerva_core::types::error::{NervaError, Result};

use crate::types::event::LedgerEventKind;
use crate::types::fallback::FallbackClass;
use crate::types::sync::SyncClass;
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

    pub fn require_production_runtime_invariants(&self) -> Result<()> {
        self.require_classified_syncs()?;
        for event in &self.events {
            if event.kind == LedgerEventKind::Sync && event.sync_class == Some(SyncClass::DebugSync)
            {
                return Err(NervaError::InvalidArgument {
                    reason: format!("debug sync '{}' is forbidden in production", event.label),
                });
            }
        }
        for fallback in &self.fallback_decisions {
            if fallback.class == FallbackClass::DebugOnly {
                return Err(NervaError::InvalidArgument {
                    reason: format!(
                        "debug fallback '{}' is forbidden in production",
                        fallback.label
                    ),
                });
            }
            if fallback.visible_ns.is_none() {
                return Err(NervaError::InvalidArgument {
                    reason: format!(
                        "fallback '{}' has no visible-cost measurement or estimate",
                        fallback.label
                    ),
                });
            }
            if fallback.label.is_empty()
                || fallback.requested.is_empty()
                || fallback.selected.is_empty()
                || fallback.reason.is_empty()
            {
                return Err(NervaError::InvalidArgument {
                    reason: "fallback decision is missing a required production label".to_string(),
                });
            }
        }
        Ok(())
    }
}
