use nerva_core::types::id::token::TokenId;

use crate::common::json::format::json_escape;
use crate::common::token::token_ids_to_json;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TinyPromptDecodeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TinyPromptDecodeSummary {
    pub status: TinyPromptDecodeStatus,
    pub prompt: String,
    pub prompt_tokens: Vec<TokenId>,
    pub seed_token: TokenId,
    pub steps: usize,
    pub generated_tokens: Vec<TokenId>,
    pub full_sequence: Vec<TokenId>,
    pub generated_text: String,
    pub prompt_text_roundtrip: String,
    pub seed_from_prompt: bool,
    pub vocabulary_covered: bool,
    pub ledger_count: u64,
    pub device_events: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub output_hash: u64,
}

impl TinyPromptDecodeSummary {
    pub fn passed(&self) -> bool {
        self.seed_from_prompt
            && self.vocabulary_covered
            && self.ledger_count == self.steps as u64
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            TinyPromptDecodeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"prompt\":\"{}\",\"prompt_tokens\":{},\"seed_token\":{},\"steps\":{},\"generated_tokens\":{},\"full_sequence\":{},\"generated_text\":\"{}\",\"prompt_text_roundtrip\":\"{}\",\"seed_from_prompt\":{},\"vocabulary_covered\":{},\"ledger_count\":{},\"device_events\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"output_hash\":{}}}",
            status,
            json_escape(&self.prompt),
            token_ids_to_json(&self.prompt_tokens),
            self.seed_token.0,
            self.steps,
            token_ids_to_json(&self.generated_tokens),
            token_ids_to_json(&self.full_sequence),
            json_escape(&self.generated_text),
            json_escape(&self.prompt_text_roundtrip),
            self.seed_from_prompt,
            self.vocabulary_covered,
            self.ledger_count,
            self.device_events,
            self.total_latency_ns,
            self.hot_path_allocations,
            self.output_hash,
        )
    }
}
