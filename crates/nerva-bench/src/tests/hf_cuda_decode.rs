use crate::cli::model::causal_lm_cuda::{
    hf_causal_lm_cuda_decode_input_json, hf_causal_lm_cuda_decode_json,
};
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
    assert!(json.contains("\"resident_kv_bytes\":"));
    assert!(json.contains("\"cuda_footprint\":{"));
    assert!(json.contains("\"context_tokens\":2"));
    assert!(json.contains("\"cuda_device_free_memory_bytes\":"));
    assert!(json.contains("\"cuda_fits_device_free_memory\":true"));
    assert!(json.contains("\"device_arena_bytes\":"));
    assert!(json.contains("\"pinned_host_bytes\":"));
    assert!(json.contains("\"resident_weight_plan\""));
    assert!(json.contains("\"plan_steps\":12"));
    assert!(json.contains("\"run_steps\":12"));
    assert!(json.contains("\"hotset_promoted_blocks\":"));
    assert!(json.contains("\"plan_gpu_resident_steps\":"));
    assert!(json.contains("\"plan_gpu_resident_weight_bytes\":"));
    assert!(json.contains("\"plan_gpu_staged_weight_bytes\":"));
    assert!(json.contains("\"plan_descriptor_blocks\":12"));
    assert!(json.contains("\"cuda_contract_descriptor_blocks\":12"));
    assert!(json.contains("\"cuda_contract_descriptor_hash\":"));
    assert!(json.contains("\"cuda_contract_matched\":true"));
    assert!(json.contains("\"kv_tokens\":2"));
    assert!(json.contains("\"graph_replays\":2"));
    assert!(json.contains("\"graph_launches\":2"));
    assert!(json.contains("\"graph_replay_events\":2"));
    assert!(json.contains("\"kernel_launches\":2"));
    assert!(json.contains("\"sync_calls\":1"));
    assert!(json.contains("\"hard_syncs\":0"));
    assert!(json.contains("\"soft_visibility_syncs\":1"));
    assert!(json.contains("\"host_causality_edges\":0"));
    assert!(json.contains("\"hot_path_allocations\":0"));
    assert!(json.contains("\"critical_paths\":["));
    assert!(json.contains("\"proves_host_wait_not_gpu_idle\":true"));
    assert!(json.contains("\"token_ledgers\":["));
    assert!(json.contains("\"device_timeline\":["));

    remove_checkpoint_dir(&dir);
}

#[test]
fn hf_cuda_decode_cli_requires_checkpoint_dir() {
    let err = hf_causal_lm_cuda_decode_json(None, 0, 1).unwrap_err();

    assert_eq!(err, "hf-cuda-decode requires checkpoint_dir");
}

#[test]
fn hf_cuda_decode_cli_accepts_token_id_prompt_sequence() {
    let dir = write_checkpoint_dir("nerva-hf-cuda-decode-ids-cli");
    let json = hf_causal_lm_cuda_decode_input_json(
        Some(dir.to_string_lossy().into_owned()),
        Some("ids:0,1".to_string()),
        2,
    )
    .unwrap();

    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"input_mode\":\"token_ids\""));
    assert!(json.contains("\"prompt_token_ids\":[0,1]"));
    assert!(json.contains("\"prompt_tokens\":2"));
    assert!(json.contains("\"seed_token\":1"));
    assert!(json.contains("\"kv_tokens\":3"));
    assert!(json.contains("\"context_tokens\":3"));
    assert!(json.contains("\"cuda_fits_device_free_memory\":true"));
    assert!(json.contains("\"resident_kv_bytes\":"));
    assert!(json.contains("\"hotset_kept_dram_blocks\":"));
    assert!(json.contains("\"cuda_contract_blocks\":12"));
    assert!(json.contains("\"plan_descriptor_hash\":"));
    assert!(json.contains("\"graph_replays\":3"));
    assert!(json.contains("\"parity\":true"));
    assert!(json.contains("\"critical_paths\":["));
    assert!(json.contains("\"token_ledgers\":["));

    remove_checkpoint_dir(&dir);
}

#[test]
fn hf_cuda_decode_cli_uses_hf_tokenizer_json_for_text_prompt() {
    let dir = write_checkpoint_dir("nerva-hf-cuda-decode-text-cli");
    write_wordlevel_tokenizer(&dir);
    let json = hf_causal_lm_cuda_decode_input_json(
        Some(dir.to_string_lossy().into_owned()),
        Some("one two".to_string()),
        2,
    )
    .unwrap();

    assert!(json.contains("\"input_mode\":\"tokenizer_json\""));
    assert!(json.contains("\"prompt_text\":\"one two\""));
    assert!(json.contains("\"prompt_token_ids\":[1,2]"));
    assert!(json.contains("\"generated_text\":\"zero zero\""));
    assert!(json.contains("\"seed_token\":2"));
    assert!(json.contains("\"kv_tokens\":3"));
    assert!(json.contains("\"cuda_footprint\":{"));
    assert!(json.contains("\"context_tokens\":3"));
    assert!(json.contains("\"run_gpu_staged_steps\":"));
    assert!(json.contains("\"cuda_contract_weight_bytes\":"));
    assert!(json.contains("\"parity\":true"));

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

fn remove_checkpoint_dir(dir: &std::path::Path) {
    let _ = std::fs::remove_file(dir.join("tokenizer.json"));
    let _ = std::fs::remove_file(dir.join("model.safetensors"));
    let _ = std::fs::remove_file(dir.join("config.json"));
    let _ = std::fs::remove_dir(dir);
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
