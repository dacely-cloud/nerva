use std::collections::BTreeMap;

use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::transport::contract::types::{TransferCompletion, TransferCompletionStatus, TransferId};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportVisibilityState {
    CompletionObserved,
    ExecutionVisible,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct TransportVisibilityRecord {
    completion: TransferCompletion,
    state: TransportVisibilityState,
}

#[derive(Clone, Debug, Default)]
pub struct TransportVisibilityTracker {
    records: BTreeMap<TransferId, TransportVisibilityRecord>,
}

impl TransportVisibilityTracker {
    pub fn observe_completion(&mut self, completion: TransferCompletion) -> Result<()> {
        if completion.status != TransferCompletionStatus::Complete {
            return Err(NervaError::InvalidArgument {
                reason: "transport visibility requires a completed transfer".to_string(),
            });
        }
        if completion.bytes == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "transport visibility requires a non-empty completion".to_string(),
            });
        }
        if self.records.contains_key(&completion.transfer_id) {
            return Err(NervaError::InvalidArgument {
                reason: "transport completion visibility was already observed".to_string(),
            });
        }
        self.records.insert(
            completion.transfer_id,
            TransportVisibilityRecord {
                completion,
                state: TransportVisibilityState::CompletionObserved,
            },
        );
        Ok(())
    }

    pub fn publish_visibility_fence(
        &mut self,
        transfer_id: TransferId,
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let record =
            self.records
                .get_mut(&transfer_id)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: "transport visibility fence references unknown completion".to_string(),
                })?;
        if record.state == TransportVisibilityState::ExecutionVisible {
            return Err(NervaError::InvalidArgument {
                reason: "transport visibility fence was already published".to_string(),
            });
        }
        ledger.record_sync(
            SyncClass::PhaseHandoff,
            Some(record.completion.destination_block),
            Some(MemoryTier::PinnedDram),
            Some(MemoryTier::PinnedDram),
            record.completion.bytes,
            1,
            MetricSource::EstimatedModel,
            "transport_completion_visibility_fence",
        );
        record.state = TransportVisibilityState::ExecutionVisible;
        Ok(())
    }

    pub fn consume_visible(&self, transfer_id: TransferId) -> Result<TransferCompletion> {
        let record = self
            .records
            .get(&transfer_id)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "transport consume references unknown completion".to_string(),
            })?;
        if record.state != TransportVisibilityState::ExecutionVisible {
            return Err(NervaError::InvalidArgument {
                reason: "transport completion is not execution-visible before visibility fence"
                    .to_string(),
            });
        }
        Ok(record.completion)
    }
}
