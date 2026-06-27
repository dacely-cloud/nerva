use std::fs;

use nerva_core::types::id::token::TokenId;

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
        candidate_source_format: "nerva_tiny_greedy",
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

pub(crate) fn compare_token_identity_artifacts(
    baseline_json: &str,
    candidate_json: &str,
) -> Result<TokenIdentityParitySummary, String> {
    let (source_format, baseline_tokens) = parse_vllm_token_ids(baseline_json)?;
    let (candidate_source_format, candidate_tokens) = parse_vllm_token_ids(candidate_json)?;
    let comparison = compare_token_slices(&baseline_tokens, &candidate_tokens);
    let vllm_token_hash = hash_tokens(&baseline_tokens);
    let nerva_token_hash = hash_tokens(&candidate_tokens);
    let hot_path_allocations =
        parse_first_u64_field(candidate_json, "hot_path_allocations").unwrap_or(1);
    let status = if comparison.mismatched_tokens == 0
        && comparison.missing_tokens == 0
        && comparison.extra_tokens == 0
        && vllm_token_hash == nerva_token_hash
        && hot_path_allocations == 0
    {
        TokenIdentityParityStatus::Ok
    } else {
        TokenIdentityParityStatus::Mismatch
    };

    Ok(TokenIdentityParitySummary {
        status,
        source_format,
        candidate_source_format,
        steps: candidate_tokens.len(),
        seed_token: TokenId(0),
        vllm_tokens: baseline_tokens,
        nerva_tokens: candidate_tokens,
        matched_tokens: comparison.matched_tokens,
        mismatched_tokens: comparison.mismatched_tokens,
        missing_tokens: comparison.missing_tokens,
        extra_tokens: comparison.extra_tokens,
        first_mismatch_index: comparison.first_mismatch_index,
        vllm_token_hash,
        nerva_token_hash,
        hot_path_allocations,
    })
}

pub(crate) fn run_vllm_token_identity_parity(
    path: Option<String>,
    steps: usize,
) -> Result<String, String> {
    load_vllm_token_identity_parity(path, steps).map(|summary| summary.to_json())
}

pub(crate) fn run_token_identity_artifact_parity(
    baseline_path: Option<String>,
    candidate_path: Option<String>,
) -> Result<String, String> {
    load_token_identity_artifact_parity(baseline_path, candidate_path)
        .map(|summary| summary.to_json())
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

pub(crate) fn load_token_identity_artifact_parity(
    baseline_path: Option<String>,
    candidate_path: Option<String>,
) -> Result<TokenIdentityParitySummary, String> {
    let baseline_path =
        baseline_path.ok_or_else(|| "token-parity requires baseline JSON path".to_string())?;
    let candidate_path =
        candidate_path.ok_or_else(|| "token-parity requires candidate JSON path".to_string())?;
    let baseline = fs::read_to_string(&baseline_path)
        .map_err(|err| format!("failed to read {baseline_path}: {err}"))?;
    let candidate = fs::read_to_string(&candidate_path)
        .map_err(|err| format!("failed to read {candidate_path}: {err}"))?;
    compare_token_identity_artifacts(&baseline, &candidate)
}

fn parse_first_u64_field(source: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\":");
    let index = source.find(&needle)? + needle.len();
    let bytes = source.as_bytes();
    let mut start = index;
    while matches!(bytes.get(start), Some(b' ' | b'\n' | b'\r' | b'\t')) {
        start += 1;
    }
    let mut end = start;
    while matches!(bytes.get(end), Some(b'0'..=b'9')) {
        end += 1;
    }
    (end > start)
        .then(|| source[start..end].parse().ok())
        .flatten()
}
