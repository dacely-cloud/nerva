use crate::model_io::resident::{
    run_hotset_probe, run_resident_shard_probe, run_weight_execution_probe,
};
use crate::model_io::safetensors::{run_safetensors_probe, run_safetensors_shard_probe};
use crate::tests::support::{
    SHARD_ONE, SHARD_TWO, synthetic_header_for_entries, synthetic_index_json,
    write_safetensors_header,
};

#[test]
fn safetensors_probe_reads_file_header_and_validates_manifest() {
    let dir = std::env::temp_dir().join(format!("nerva-bench-header-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config_path = dir.join("config.json");
    let path = dir.join("model.safetensors");
    let config = r#"{
            "model_type": "llama",
            "hidden_size": 2,
            "intermediate_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 4,
            "torch_dtype": "float16"
        }"#;
    std::fs::write(&config_path, config).unwrap();
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(config).unwrap();
    let layout = nerva_model::weights::layout::plan_hf_weight_layout(&metadata).unwrap();
    let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&layout).unwrap();
    let header = synthetic_header_for_entries(manifest.architecture, &manifest.entries);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(8 + header.len() + manifest.total_weight_bytes, 0);
    std::fs::write(&path, bytes).unwrap();

    let json = run_safetensors_probe(
        Some(config_path.to_string_lossy().into_owned()),
        Some(path.to_string_lossy().into_owned()),
    )
    .unwrap();

    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"file_header\""));
    assert!(json.contains("\"validation\""));
    assert!(json.contains("\"validated_tensors\":11"));
    assert!(json.contains("\"payload_bytes\""));

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&config_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn safetensors_shard_probe_reads_index_and_headers() {
    let dir = std::env::temp_dir().join(format!("nerva-bench-shard-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config_path = dir.join("config.json");
    let index_path = dir.join("model.safetensors.index.json");
    let config = r#"{
            "model_type": "llama",
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 10,
            "torch_dtype": "float16"
        }"#;
    std::fs::write(&config_path, config).unwrap();

    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(config).unwrap();
    let layout = nerva_model::weights::layout::plan_hf_weight_layout(&metadata).unwrap();
    let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&layout).unwrap();
    let index = synthetic_index_json(&manifest, 10);
    std::fs::write(&index_path, index).unwrap();
    write_safetensors_header(
        &dir.join(SHARD_ONE),
        &synthetic_header_for_entries(manifest.architecture, &manifest.entries[..10]),
        manifest.entries[..10].iter().map(|entry| entry.bytes).sum(),
    );
    write_safetensors_header(
        &dir.join(SHARD_TWO),
        &synthetic_header_for_entries(manifest.architecture, &manifest.entries[10..]),
        manifest.entries[10..].iter().map(|entry| entry.bytes).sum(),
    );

    let json = run_safetensors_shard_probe(
        Some(config_path.to_string_lossy().into_owned()),
        Some(index_path.to_string_lossy().into_owned()),
        Some(dir.to_string_lossy().into_owned()),
    )
    .unwrap();

    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"entries\":20"));
    assert!(json.contains("\"shards\":2"));

    let resident_json = run_resident_shard_probe(
        Some(config_path.to_string_lossy().into_owned()),
        Some(index_path.to_string_lossy().into_owned()),
        Some(dir.to_string_lossy().into_owned()),
        128,
    )
    .unwrap();
    assert!(resident_json.contains("\"status\":\"ok\""));
    assert!(resident_json.contains("\"blocks\":20"));
    assert!(resident_json.contains("\"prefetch\""));
    assert!(resident_json.contains("\"execution\""));
    assert!(resident_json.contains("\"tasks\":20"));
    assert!(resident_json.contains("\"disk_read_events\""));
    assert!(resident_json.contains("\"data_hash\""));

    let hotset_json =
        run_hotset_probe(Some(config_path.to_string_lossy().into_owned()), 256, 200).unwrap();
    assert!(hotset_json.contains("\"status\":\"ok\""));
    assert!(hotset_json.contains("\"hotset\""));
    assert!(hotset_json.contains("\"promoted_blocks\":7"));

    let execution_json = run_weight_execution_probe(
        Some(config_path.to_string_lossy().into_owned()),
        128,
        100,
        3,
        Some(89),
    )
    .unwrap();
    assert!(execution_json.contains("\"status\":\"ok\""));
    assert!(execution_json.contains("\"execution\""));
    assert!(execution_json.contains("\"run\""));
    assert!(execution_json.contains("\"gpu_resident_steps\":2"));
    assert!(execution_json.contains("\"gpu_staged_steps\":1"));

    let _ = std::fs::remove_file(dir.join(SHARD_ONE));
    let _ = std::fs::remove_file(dir.join(SHARD_TWO));
    let _ = std::fs::remove_file(config_path);
    let _ = std::fs::remove_file(index_path);
    let _ = std::fs::remove_dir(dir);
}
