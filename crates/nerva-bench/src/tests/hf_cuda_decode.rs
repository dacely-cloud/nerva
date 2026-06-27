use crate::cli::model::causal_lm_cuda::hf_causal_lm_cuda_decode_json;
use crate::tests::support::{synthetic_header_for_entries, write_safetensors_header};

#[test]
fn hf_cuda_decode_cli_loads_checkpoint_dir_and_reports_cuda_ledgers() {
    let dir = write_checkpoint_dir("nerva-hf-cuda-decode-cli");
    let json =
        hf_causal_lm_cuda_decode_json(Some(dir.to_string_lossy().into_owned()), 0, 2).unwrap();

    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"backend\":\"cuda\""));
    assert!(json.contains("\"input_mode\":\"seed_token\""));
    assert!(json.contains("\"seed_token\":0"));
    assert!(json.contains("\"layers\":1"));
    assert!(json.contains("\"tokens\":[0,0]"));
    assert!(json.contains("\"expected_tokens\":[0,0]"));
    assert!(json.contains("\"parity\":true"));
    assert!(json.contains("\"ledger_count\":2"));
    assert!(json.contains("\"device_events\":2"));
    assert!(json.contains("\"kernel_launches\":2"));
    assert!(json.contains("\"sync_calls\":1"));
    assert!(json.contains("\"host_causality_edges\":0"));
    assert!(json.contains("\"hot_path_allocations\":0"));

    remove_checkpoint_dir(&dir);
}

#[test]
fn hf_cuda_decode_cli_requires_checkpoint_dir() {
    let err = hf_causal_lm_cuda_decode_json(None, 0, 1).unwrap_err();

    assert_eq!(err, "hf-cuda-decode requires checkpoint_dir");
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

fn remove_checkpoint_dir(dir: &std::path::Path) {
    let _ = std::fs::remove_file(dir.join("model.safetensors"));
    let _ = std::fs::remove_file(dir.join("config.json"));
    let _ = std::fs::remove_dir(dir);
}
