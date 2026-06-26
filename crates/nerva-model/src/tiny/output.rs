use nerva_core::types::id::TokenId;
use nerva_ledger::types::token::TokenLedger;

use crate::common::token::token_ids_to_json;

#[derive(Clone, Debug, PartialEq)]
pub struct TinyGreedyDecodeOutput {
    pub tokens: Vec<TokenId>,
    pub ledgers: Vec<TokenLedger>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TinyGreedyDecodeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TinyGreedyDecodeSummary {
    pub status: TinyGreedyDecodeStatus,
    pub seed_token: TokenId,
    pub steps: usize,
    pub vocab_size: usize,
    pub tokens: Vec<TokenId>,
    pub expected_tokens: Vec<TokenId>,
    pub parity: bool,
    pub ledger_count: u64,
    pub device_events: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub output_hash: u64,
}

impl TinyGreedyDecodeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            TinyGreedyDecodeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"seed_token\":{},\"steps\":{},\"vocab_size\":{},\"tokens\":{},\"expected_tokens\":{},\"parity\":{},\"ledger_count\":{},\"device_events\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"output_hash\":{}}}",
            status,
            self.seed_token.0,
            self.steps,
            self.vocab_size,
            token_ids_to_json(&self.tokens),
            token_ids_to_json(&self.expected_tokens),
            self.parity,
            self.ledger_count,
            self.device_events,
            self.total_latency_ns,
            self.hot_path_allocations,
            self.output_hash,
        )
    }
}
