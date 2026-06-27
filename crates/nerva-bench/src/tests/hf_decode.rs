use crate::cli::model::causal_lm::hf_causal_lm_decode_json;
use crate::tests::support::{synthetic_header_for_entries, write_safetensors_header};

#[test]
fn hf_decode_cli_loads_checkpoint_dir_and_reports_ledgers() {
    let dir = std::env::temp_dir().join(format!("nerva-hf-decode-cli-{}", std::process::id()));
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

    let json = hf_causal_lm_decode_json(Some(dir.to_string_lossy().into_owned()), 0, 3).unwrap();

    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"tokens\":[0,0,0]"));
    assert!(json.contains("\"manifest_entries\":12"));
    assert!(json.contains("\"shard_plan_entries\":12"));
    assert!(json.contains("\"tensors_loaded\":12"));
    assert!(json.contains("\"final_norm_manifest\":true"));
    assert!(json.contains("\"ledger_count\":3"));
    assert!(json.contains("\"ledger_events\":3"));
    assert!(json.contains("\"execution_decisions\":3"));
    assert!(json.contains("\"hot_path_allocations\":0"));

    let _ = std::fs::remove_file(dir.join("model.safetensors"));
    let _ = std::fs::remove_file(dir.join("config.json"));
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn hf_decode_cli_requires_checkpoint_dir() {
    let err = hf_causal_lm_decode_json(None, 0, 1).unwrap_err();

    assert_eq!(err, "hf-decode requires checkpoint_dir");
}
