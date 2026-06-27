use crate::cli::model::causal_lm::hf_causal_lm_decode_input_json;
use crate::tests::support::{synthetic_header_for_entries, write_safetensors_header};

#[test]
fn hf_decode_cli_loads_checkpoint_dir_and_reports_ledgers() {
    let dir = write_checkpoint_dir("nerva-hf-decode-cli");

    let json = hf_causal_lm_decode_input_json(
        Some(dir.to_string_lossy().into_owned()),
        Some("0".to_string()),
        3,
    )
    .unwrap();

    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"input_mode\":\"token_id\""));
    assert!(json.contains("\"context_mode\":\"last_token_seed_only\""));
    assert!(json.contains("\"prompt_token_ids\":[0]"));
    assert!(json.contains("\"tokens\":[0,0,0]"));
    assert!(json.contains("\"manifest_entries\":12"));
    assert!(json.contains("\"shard_plan_entries\":12"));
    assert!(json.contains("\"tensors_loaded\":12"));
    assert!(json.contains("\"final_norm_manifest\":true"));
    assert!(json.contains("\"ledger_count\":3"));
    assert!(json.contains("\"ledger_events\":3"));
    assert!(json.contains("\"execution_decisions\":3"));
    assert!(json.contains("\"hot_path_allocations\":0"));

    remove_checkpoint_dir(&dir);
}

#[test]
fn hf_decode_cli_accepts_token_id_prompt_sequence() {
    let dir = write_checkpoint_dir("nerva-hf-decode-ids-cli");
    let json = hf_causal_lm_decode_input_json(
        Some(dir.to_string_lossy().into_owned()),
        Some("ids:1,2".to_string()),
        2,
    )
    .unwrap();

    assert!(json.contains("\"input_mode\":\"token_ids\""));
    assert!(json.contains("\"context_mode\":\"last_token_seed_only\""));
    assert!(json.contains("\"prompt_token_ids\":[1,2]"));
    assert!(json.contains("\"prompt_tokens\":2"));
    assert!(json.contains("\"seed_token\":2"));
    assert!(json.contains("\"ledger_count\":2"));

    remove_checkpoint_dir(&dir);
}

#[test]
fn hf_decode_cli_uses_hf_tokenizer_json_for_text_prompt() {
    let dir = write_checkpoint_dir("nerva-hf-decode-text-cli");
    write_wordlevel_tokenizer(&dir);

    let json = hf_causal_lm_decode_input_json(
        Some(dir.to_string_lossy().into_owned()),
        Some("one two".to_string()),
        2,
    )
    .unwrap();

    assert!(json.contains("\"input_mode\":\"tokenizer_json\""));
    assert!(json.contains("\"context_mode\":\"last_token_seed_only\""));
    assert!(json.contains("\"prompt_text\":\"one two\""));
    assert!(json.contains("\"prompt_token_ids\":[1,2]"));
    assert!(json.contains("\"seed_token\":2"));
    assert!(json.contains("\"ledger_count\":2"));

    remove_checkpoint_dir(&dir);
}

#[test]
fn hf_decode_cli_requires_checkpoint_dir() {
    let err = hf_causal_lm_decode_input_json(None, Some("0".to_string()), 1).unwrap_err();

    assert_eq!(err, "hf-decode requires checkpoint_dir");
}

#[test]
fn hf_decode_cli_rejects_prompt_token_outside_vocab() {
    let dir = write_checkpoint_dir("nerva-hf-decode-bad-token-cli");
    let err = hf_causal_lm_decode_input_json(
        Some(dir.to_string_lossy().into_owned()),
        Some("ids:1,99".to_string()),
        1,
    )
    .unwrap_err();

    assert!(err.contains("token id 99 is outside model vocabulary 4"));

    remove_checkpoint_dir(&dir);
}

fn write_checkpoint_dir(prefix: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = r#"{
            "model_type": "llama",
            "hidden_size": 2,
            "intermediate_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 4,
            "torch_dtype": "float16"
        }"#;
    std::fs::write(dir.join("config.json"), config).unwrap();
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(config).unwrap();
    let layout = nerva_model::weights::layout::plan::plan_hf_weight_layout(&metadata).unwrap();
    let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&layout).unwrap();
    let header = synthetic_header_for_entries(manifest.architecture, &manifest.entries);
    write_safetensors_header(
        &dir.join("model.safetensors"),
        &header,
        manifest.total_weight_bytes,
    );
    dir
}

fn write_wordlevel_tokenizer(dir: &std::path::Path) {
    let tokenizer = r#"{
        "version":"1.0",
        "truncation":null,
        "padding":null,
        "added_tokens":[],
        "normalizer":null,
        "pre_tokenizer":{"type":"Whitespace"},
        "post_processor":null,
        "decoder":null,
        "model":{
            "type":"WordLevel",
            "vocab":{"zero":0,"one":1,"two":2,"three":3},
            "unk_token":"zero"
        }
    }"#;
    std::fs::write(dir.join("tokenizer.json"), tokenizer).unwrap();
}

fn remove_checkpoint_dir(dir: &std::path::Path) {
    let _ = std::fs::remove_file(dir.join("tokenizer.json"));
    let _ = std::fs::remove_file(dir.join("model.safetensors"));
    let _ = std::fs::remove_file(dir.join("config.json"));
    let _ = std::fs::remove_dir(dir);
}
