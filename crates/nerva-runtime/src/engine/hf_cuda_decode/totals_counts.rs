use nerva_core::types::id::device::DeviceOrdinal;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::critical::TokenCriticalPathReport;
use nerva_ledger::types::token::ledger::TokenLedger;

pub(super) fn critical_paths(ledgers: &[TokenLedger]) -> Vec<TokenCriticalPathReport> {
    ledgers
        .iter()
        .map(|ledger| {
            TokenCriticalPathReport::from_ledger(ledger, DeviceOrdinal(0))
                .expect("HF CUDA token ledgers have valid device timelines")
        })
        .collect()
}

pub(super) fn event_count(ledgers: &[TokenLedger], kind: LedgerEventKind) -> u64 {
    ledgers.iter().map(|ledger| ledger.event_count(kind)).sum()
}

pub(super) fn sync_count(ledgers: &[TokenLedger], class: SyncClass) -> u64 {
    ledgers
        .iter()
        .map(|ledger| ledger.sync_count_for(class))
        .sum()
}

pub(super) fn execution_decisions(ledgers: &[TokenLedger]) -> u64 {
    ledgers
        .iter()
        .map(|ledger| ledger.execution_decisions.len() as u64)
        .sum()
}

pub(super) fn hot_path_allocations(ledgers: &[TokenLedger]) -> u64 {
    ledgers
        .iter()
        .map(|ledger| ledger.hot_path_allocations)
        .sum()
}
