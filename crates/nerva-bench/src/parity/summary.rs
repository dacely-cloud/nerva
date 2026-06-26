use nerva_core::types::id::TokenId;

use crate::json::json_escape;
use crate::parity::json::{json_opt_usize, token_ids_to_json};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TokenIdentityParitySummary {
    pub status: TokenIdentityParityStatus,
    pub source_format: &'static str,
    pub steps: usize,
    pub seed_token: TokenId,
    pub vllm_tokens: Vec<TokenId>,
    pub nerva_tokens: Vec<TokenId>,
    pub matched_tokens: usize,
    pub mismatched_tokens: usize,
    pub missing_tokens: usize,
    pub extra_tokens: usize,
    pub first_mismatch_index: Option<usize>,
    pub vllm_token_hash: u64,
    pub nerva_token_hash: u64,
    pub hot_path_allocations: u64,
}

impl TokenIdentityParitySummary {
    pub fn passed(&self) -> bool {
        matches!(self.status, TokenIdentityParityStatus::Ok)
            && self.mismatched_tokens == 0
            && self.missing_tokens == 0
            && self.extra_tokens == 0
            && self.vllm_token_hash == self.nerva_token_hash
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            TokenIdentityParityStatus::Ok => "ok",
            TokenIdentityParityStatus::Mismatch => "mismatch",
        };
        format!(
            "{{\"status\":\"{}\",\"source_format\":\"{}\",\"steps\":{},\"seed_token\":{},\"vllm_tokens\":{},\"nerva_tokens\":{},\"matched_tokens\":{},\"mismatched_tokens\":{},\"missing_tokens\":{},\"extra_tokens\":{},\"first_mismatch_index\":{},\"vllm_token_hash\":{},\"nerva_token_hash\":{},\"hot_path_allocations\":{}}}",
            status,
            json_escape(self.source_format),
            self.steps,
            self.seed_token.0,
            token_ids_to_json(&self.vllm_tokens),
            token_ids_to_json(&self.nerva_tokens),
            self.matched_tokens,
            self.mismatched_tokens,
            self.missing_tokens,
            self.extra_tokens,
            json_opt_usize(self.first_mismatch_index),
            self.vllm_token_hash,
            self.nerva_token_hash,
            self.hot_path_allocations,
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum TokenIdentityParityStatus {
    Ok,
    Mismatch,
}
