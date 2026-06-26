use crate::artifact::run::run_artifact;
use crate::parity::run::run_vllm_token_identity_parity;

#[test]
fn vllm_token_identity_parity_reads_vllm_style_json() {
    let dir = std::env::temp_dir().join(format!("nerva-bench-parity-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vllm_tokens.json");
    std::fs::write(
        &path,
        r#"{"request_id":"test","outputs":[{"token_ids":[1,2,3,0,1,2,3,0]}]}"#,
    )
    .unwrap();

    let json =
        run_vllm_token_identity_parity(Some(path.to_string_lossy().into_owned()), 8).unwrap();

    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"source_format\":\"token_ids\""));
    assert!(json.contains("\"matched_tokens\":8"));
    assert!(json.contains("\"mismatched_tokens\":0"));
    assert!(json.contains("\"missing_tokens\":0"));
    assert!(json.contains("\"extra_tokens\":0"));
    assert!(json.contains("\"hot_path_allocations\":0"));

    let artifact = run_artifact(
        Some("vllm-parity".to_string()),
        vec![path.to_string_lossy().into_owned(), "8".to_string()],
    )
    .unwrap();
    assert!(artifact.contains("\"artifact_schema\":\"nerva-bench-v1\""));
    assert!(artifact.contains("\"command\":\"vllm-parity\""));
    assert!(artifact.contains("\"summary\":{\"status\":\"ok\""));

    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn vllm_token_identity_parity_reports_mismatch() {
    let dir = std::env::temp_dir().join(format!(
        "nerva-bench-parity-mismatch-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vllm_tokens.json");
    std::fs::write(&path, r#"{"outputs":[{"token_ids":[1,2,99,0]}]}"#).unwrap();

    let json =
        run_vllm_token_identity_parity(Some(path.to_string_lossy().into_owned()), 4).unwrap();

    assert!(json.contains("\"status\":\"mismatch\""));
    assert!(json.contains("\"mismatched_tokens\":1"));
    assert!(json.contains("\"first_mismatch_index\":2"));

    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_dir(dir);
}
