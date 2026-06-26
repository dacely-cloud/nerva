use std::fs;

use crate::parity::compare::compare_token_slices;
use crate::parity::hash::hash_tokens;
use crate::parity::parser::parse_vllm_token_ids;
use crate::parity::summary::{TokenIdentityParityStatus, TokenIdentityParitySummary};

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
