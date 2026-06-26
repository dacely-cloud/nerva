use crate::parity::run::compare_vllm_token_identity;
use crate::parity::summary::TokenIdentityParityStatus;

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
fn rejects_missing_or_invalid_token_arrays() {
    assert!(compare_vllm_token_identity(r#"{"text":"hello"}"#, 4).is_err());
    assert!(compare_vllm_token_identity(r#"{"token_ids":[1,-2]}"#, 4).is_err());
    assert!(compare_vllm_token_identity(r#"{"token_ids":["1"]}"#, 4).is_err());
}
