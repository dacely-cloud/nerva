use nerva_ledger::types::token::critical::TokenCriticalPathReport;
use nerva_ledger::types::token::ledger::TokenLedger;

pub(super) fn critical_paths_json(paths: &[TokenCriticalPathReport]) -> String {
    let mut out = String::from("[");
    for (index, path) in paths.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&path.to_json());
    }
    out.push(']');
    out
}

pub(super) fn token_ledgers_json(ledgers: &[TokenLedger]) -> String {
    let mut out = String::from("[");
    for (index, ledger) in ledgers.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&ledger.to_json());
    }
    out.push(']');
    out
}
