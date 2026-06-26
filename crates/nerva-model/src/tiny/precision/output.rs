use nerva_core::types::dtype::DType;
use nerva_core::types::id::TokenId;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::precision::bits::dtype_label;

#[derive(Clone, Debug, PartialEq)]
pub struct TinyPrecisionGreedyDecodeOutput {
    pub tokens: Vec<TokenId>,
    pub ledgers: Vec<TokenLedger>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TinyPrecisionGreedyDecodeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TinyPrecisionGreedyDecodeSummary {
    pub status: TinyPrecisionGreedyDecodeStatus,
    pub dtype: DType,
    pub seed_token: TokenId,
    pub steps: usize,
    pub vocab_size: usize,
    pub tokens: Vec<TokenId>,
    pub expected_tokens: Vec<TokenId>,
    pub parity: bool,
    pub ledger_count: u64,
    pub cpu_events: u64,
    pub execution_decisions: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub output_hash: u64,
}

impl TinyPrecisionGreedyDecodeSummary {
    pub fn passed(&self) -> bool {
        self.parity
            && self.ledger_count == self.steps as u64
            && self.cpu_events == self.steps as u64
            && self.execution_decisions == self.steps as u64
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            TinyPrecisionGreedyDecodeStatus::Ok => "ok",
        };
        let dtype = dtype_label(self.dtype).unwrap_or("unsupported");
        format!(
            "{{\"status\":\"{}\",\"dtype\":\"{}\",\"seed_token\":{},\"steps\":{},\"vocab_size\":{},\"tokens\":{},\"expected_tokens\":{},\"parity\":{},\"ledger_count\":{},\"cpu_events\":{},\"execution_decisions\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"output_hash\":{}}}",
            status,
            dtype,
            self.seed_token.0,
            self.steps,
            self.vocab_size,
            crate::common::token::token_ids_to_json(&self.tokens),
            crate::common::token::token_ids_to_json(&self.expected_tokens),
            self.parity,
            self.ledger_count,
            self.cpu_events,
            self.execution_decisions,
            self.total_latency_ns,
            self.hot_path_allocations,
            self.output_hash,
        )
    }
}
