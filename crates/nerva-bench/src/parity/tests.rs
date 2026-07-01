use crate::parity::run::{compare_token_identity_artifacts, compare_vllm_token_identity};
use crate::parity::summary::TokenIdentityParityStatus;
use nerva_core::types::id::token::TokenId;

#[test]
fn accepts_vllm_nested_token_ids_for_exact_identity() {
    let summary =
        compare_vllm_token_identity(r#"{"outputs":[{"token_ids":[1,2,3,0]}]}"#, 4).unwrap();

    assert!(summary.passed());
    assert_eq!(summary.source_format, "token_ids");
    assert_eq!(summary.matched_tokens, 4);
    assert_eq!(summary.mismatched_tokens, 0);
    assert_eq!(summary.missing_tokens, 0);
    assert_eq!(summary.extra_tokens, 0);
    assert_eq!(summary.first_mismatch_index, None);
    assert_eq!(summary.vllm_token_hash, summary.nerva_token_hash);
    assert!(summary.to_json().contains("\"status\":\"ok\""));
}

#[test]
fn reports_mismatch_and_first_mismatch_index() {
    let summary = compare_vllm_token_identity(r#"{"output_token_ids":[1,2,99,0]}"#, 4).unwrap();

    assert!(!summary.passed());
    assert_eq!(summary.status, TokenIdentityParityStatus::Mismatch);
    assert_eq!(summary.matched_tokens, 3);
    assert_eq!(summary.mismatched_tokens, 1);
    assert_eq!(summary.first_mismatch_index, Some(2));
    assert_ne!(summary.vllm_token_hash, summary.nerva_token_hash);
}

#[test]
fn reports_missing_and_extra_tokens() {
    let missing = compare_vllm_token_identity(r#"{"tokens":[1,2]}"#, 4).unwrap();
    assert_eq!(missing.missing_tokens, 2);
    assert_eq!(missing.extra_tokens, 0);
    assert_eq!(missing.first_mismatch_index, Some(2));

    let extra = compare_vllm_token_identity(r#"{"generated_token_ids":[1,2,3,0,1]}"#, 4).unwrap();
    assert_eq!(extra.missing_tokens, 0);
    assert_eq!(extra.extra_tokens, 1);
    assert_eq!(extra.first_mismatch_index, Some(4));
}

#[test]
fn token_parser_skips_matching_string_values() {
    let summary = compare_token_identity_artifacts(
        r#"{"prompt_mode":"token_ids","prompt_token_ids":[10,11],"tokens":[1,2]}"#,
        r#"{"tokens":[1,2],"hot_path_allocations":0}"#,
    )
    .unwrap();

    assert!(summary.passed());
    assert_eq!(summary.source_format, "tokens");
    assert_eq!(summary.vllm_tokens, vec![TokenId(1), TokenId(2)]);
}

#[test]
fn rejects_missing_or_invalid_token_arrays() {
    assert!(compare_vllm_token_identity(r#"{"text":"hello"}"#, 4).is_err());
    assert!(compare_vllm_token_identity(r#"{"token_ids":[1,-2]}"#, 4).is_err());
    assert!(compare_vllm_token_identity(r#"{"token_ids":["1"]}"#, 4).is_err());
}

#[test]
fn compares_external_token_artifacts_without_tiny_model_substitution() {
    let baseline = r#"{"engine":"vllm","outputs":[{"token_ids":[50994,67]}]}"#;
    let candidate = r#"{"engine":"nerva","tokens":[50994,67],"hot_path_allocations":0}"#;

    let summary = compare_token_identity_artifacts(baseline, candidate).unwrap();

    assert!(summary.passed());
    assert_eq!(summary.source_format, "token_ids");
    assert_eq!(summary.candidate_source_format, "tokens");
    assert_eq!(summary.vllm_tokens, summary.nerva_tokens);
    assert_eq!(summary.steps, 2);
    assert!(
        summary
            .to_json()
            .contains("\"candidate_source_format\":\"tokens\"")
    );
}

#[test]
fn external_token_artifact_parity_rejects_mismatch_or_hot_path_allocations() {
    let baseline = r#"{"token_ids":[1,2]}"#;
    let mismatch = r#"{"tokens":[1,3],"hot_path_allocations":0}"#;
    let allocated = r#"{"tokens":[1,2],"hot_path_allocations":1}"#;

    let mismatch = compare_token_identity_artifacts(baseline, mismatch).unwrap();
    assert_eq!(mismatch.status, TokenIdentityParityStatus::Mismatch);
    assert_eq!(mismatch.first_mismatch_index, Some(1));

    let allocated = compare_token_identity_artifacts(baseline, allocated).unwrap();
    assert_eq!(allocated.status, TokenIdentityParityStatus::Mismatch);
    assert_eq!(allocated.hot_path_allocations, 1);
}
