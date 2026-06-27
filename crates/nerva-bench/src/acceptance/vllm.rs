use crate::parity::run::{compare_token_identity_artifacts, compare_vllm_token_identity};

pub(crate) fn vllm_token_identity_acceptance() -> Result<(bool, String), String> {
    let vllm_style_json =
        r#"{"request_id":"nerva-m4-parity","outputs":[{"token_ids":[1,2,3,0,1,2,3,0]}]}"#;
    let summary = compare_vllm_token_identity(vllm_style_json, 8)?;
    Ok((
        summary.passed(),
        format!(
            "source_format={} steps={} matched={} mismatched={} missing={} extra={} first_mismatch={} vllm_hash={} nerva_hash={} hot_path_allocations={}",
            summary.source_format,
            summary.steps,
            summary.matched_tokens,
            summary.mismatched_tokens,
            summary.missing_tokens,
            summary.extra_tokens,
            summary
                .first_mismatch_index
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            summary.vllm_token_hash,
            summary.nerva_token_hash,
            summary.hot_path_allocations,
        ),
    ))
}

pub(crate) fn qwen3_vllm_nerva_token_acceptance() -> Result<(bool, String), String> {
    let vllm = include_str!("../../../../docs/source/parity/qwen3_8b_vllm_tokens.json");
    let nerva = include_str!("../../../../docs/source/parity/qwen3_8b_nerva_tokens.json");
    let summary = compare_token_identity_artifacts(vllm, nerva)?;
    Ok((
        summary.passed(),
        format!(
            "workload=qwen3_8b_bf16_greedy_decode source_format={} candidate_source_format={} steps={} matched={} mismatched={} missing={} extra={} first_mismatch={} vllm_hash={} nerva_hash={} hot_path_allocations={}",
            summary.source_format,
            summary.candidate_source_format,
            summary.steps,
            summary.matched_tokens,
            summary.mismatched_tokens,
            summary.missing_tokens,
            summary.extra_tokens,
            summary
                .first_mismatch_index
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            summary.vllm_token_hash,
            summary.nerva_token_hash,
            summary.hot_path_allocations,
        ),
    ))
}
