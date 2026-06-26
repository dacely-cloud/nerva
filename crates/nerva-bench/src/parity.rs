mod compare;
mod hash;
mod json;
mod parser;

#[cfg(test)]
mod tests;

use std::fs;

use nerva_core::types::id::TokenId;

use crate::json::json_escape;
use crate::parity::compare::compare_token_slices;
use crate::parity::hash::hash_tokens;
use crate::parity::json::{json_opt_usize, token_ids_to_json};
use crate::parity::parser::parse_vllm_token_ids;

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

pub(crate) fn compare_vllm_token_identity(
    vllm_json: &str,
    steps: usize,
) -> Result<TokenIdentityParitySummary, String> {
    let (source_format, vllm_tokens) = parse_vllm_token_ids(vllm_json)?;
    let nerva_summary = nerva_model::tiny::smoke::tiny_greedy_decode_smoke(steps)
        .map_err(|err| format!("NERVA tiny greedy decode failed: {err:?}"))?;
    let nerva_tokens = nerva_summary.tokens;
    let comparison = compare_token_slices(&vllm_tokens, &nerva_tokens);
    let vllm_token_hash = hash_tokens(&vllm_tokens);
    let nerva_token_hash = hash_tokens(&nerva_tokens);
    let status = if comparison.mismatched_tokens == 0
        && comparison.missing_tokens == 0
        && comparison.extra_tokens == 0
        && vllm_token_hash == nerva_token_hash
        && nerva_summary.hot_path_allocations == 0
    {
        TokenIdentityParityStatus::Ok
    } else {
        TokenIdentityParityStatus::Mismatch
    };

    Ok(TokenIdentityParitySummary {
        status,
        source_format,
        steps,
        seed_token: nerva_summary.seed_token,
        vllm_tokens,
        nerva_tokens,
        matched_tokens: comparison.matched_tokens,
        mismatched_tokens: comparison.mismatched_tokens,
        missing_tokens: comparison.missing_tokens,
        extra_tokens: comparison.extra_tokens,
        first_mismatch_index: comparison.first_mismatch_index,
        vllm_token_hash,
        nerva_token_hash,
        hot_path_allocations: nerva_summary.hot_path_allocations,
    })
}

pub(crate) fn run_vllm_token_identity_parity(
    path: Option<String>,
    steps: usize,
) -> Result<String, String> {
    load_vllm_token_identity_parity(path, steps).map(|summary| summary.to_json())
}

pub(crate) fn load_vllm_token_identity_parity(
    path: Option<String>,
    steps: usize,
) -> Result<TokenIdentityParitySummary, String> {
    let path = path.ok_or_else(|| "vllm-parity requires a vLLM token JSON path".to_string())?;
    let contents =
        fs::read_to_string(&path).map_err(|err| format!("failed to read {path}: {err}"))?;
    compare_vllm_token_identity(&contents, steps)
}
