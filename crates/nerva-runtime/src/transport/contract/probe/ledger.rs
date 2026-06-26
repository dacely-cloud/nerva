use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::transport::contract::types::{TransferCompletion, TransferCompletionStatus, TransferId};
use crate::transport::path::types::TransferMode;

pub(super) fn record_completion(ledger: &mut TokenLedger, completion: TransferCompletion) {
    if completion.status != TransferCompletionStatus::Complete {
        return;
    }
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Transport,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(completion.source_block),
        from_tier: Some(MemoryTier::PinnedDram),
        to_tier: Some(MemoryTier::PinnedDram),
        bytes: completion.bytes,
        latency_ns: completion.bytes as u64,
        label: "transport_contract_loopback_send",
    });
}

pub(super) fn empty_completion() -> TransferCompletion {
    TransferCompletion {
        transfer_id: TransferId(0),
        source_block: ResidentBlockId(0),
        destination_block: ResidentBlockId(0),
        block_version: 0,
        bytes: 0,
        mode: TransferMode::Decode,
        status: TransferCompletionStatus::Complete,
    }
}
