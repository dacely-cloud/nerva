use nerva_cuda::smoke::status::SmokeStatus;
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

pub(super) fn status_json(status: &SmokeStatus) -> &'static str {
    match status {
        SmokeStatus::Ok => "ok",
        SmokeStatus::Unavailable => "unavailable",
        SmokeStatus::Failed => "failed",
    }
}

pub(super) fn json_string(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\"")),
        None => "null".to_string(),
    }
}

pub(super) fn json_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

pub(super) fn json_opt_bool(value: Option<bool>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}
