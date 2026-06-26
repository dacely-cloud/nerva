use std::path::Path;

use crate::{
    acceptance::run_acceptance_probe,
    artifact::run_artifact,
    json::json_string_array,
    model_io::{
        run_hotset_probe, run_resident_shard_probe, run_safetensors_probe,
        run_safetensors_shard_probe, run_weight_execution_probe,
    },
    parity::run_vllm_token_identity_parity,
};

const SHARD_ONE: &str = "model-00001-of-00002.safetensors";
const SHARD_TWO: &str = "model-00002-of-00002.safetensors";

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

#[test]
fn artifact_wraps_probe_with_reproducibility_metadata() {
    let json = run_artifact(
        Some("synthetic".to_string()),
        vec!["2".to_string(), "4".to_string()],
    )
    .unwrap();

    assert!(json.contains("\"artifact_schema\":\"nerva-bench-v1\""));
    assert!(json.contains("\"command\":\"synthetic\""));
    assert!(json.contains("\"args\":[\"2\",\"4\"]"));
    assert!(json.contains("\"git_commit\""));
    assert!(json.contains("\"package_version\""));
    assert!(json.contains("\"rustc_version\""));
    assert!(json.contains("\"cargo_version\""));
    assert!(json.contains("\"rustflags\""));
    assert!(json.contains("\"cargo_encoded_rustflags\""));
    assert!(json.contains("\"capabilities\""));
    assert!(json.contains("\"target_os\":\"linux\""));
    assert!(json.contains("\"cuda_compute_capability\""));
    assert!(json.contains("\"cuda_device_total_memory_bytes\""));
    assert!(json.contains("\"cuda_pci_bus_id\""));
    assert!(json.contains("\"rdma_core_loaded\""));
    assert!(json.contains("\"mlx5_core_loaded\""));
    assert!(json.contains("\"nvidia_peer_memory_module\""));
    assert!(json.contains("\"topology\""));
    assert!(json.contains("\"summary\""));
    assert!(json.contains("\"observed_token_hash\""));
    assert!(json.contains("\"token_ring_reuses\""));
    assert!(json.contains("\"device_timeline_idle_ns\":0"));
}

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

#[test]
fn acceptance_probe_reports_current_invariants() {
    let json = run_acceptance_probe().unwrap();

    assert!(json.contains("\"acceptance_schema\":\"nerva-acceptance-v1\""));
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"failed\":0"));
    assert!(json.contains("\"vllm_rvllm_audit\""));
    assert!(json.contains("\"cuda_runtime_smoke\""));
    assert!(json.contains("\"cuda_graph_transaction\""));
    assert!(json.contains("\"static_arenas\""));
    assert!(json.contains("\"topology_snapshot\""));
    assert!(json.contains("\"synthetic_transaction\""));
    assert!(json.contains("\"synthetic_device_token\""));
    assert!(json.contains("\"fp16_bf16_precision_block\""));
    assert!(json.contains("\"safetensors_precision_block\""));
    assert!(json.contains("\"hf_model_manifest\""));
    assert!(json.contains("\"safetensors_file_header\""));
    assert!(json.contains("\"safetensors_file_prefetch\""));
    assert!(json.contains("\"vllm_token_identity_parity\""));
    assert!(json.contains("\"kv_residency_tiering\""));
    assert!(json.contains("\"transport_pinned_fallback\""));
    assert!(json.contains("\"transport_capability_matrix\""));
    assert!(json.contains("\"resident_weight_execution\""));
}

#[test]
fn json_string_array_escapes_probe_args() {
    let args = vec!["quote\"".to_string(), "line\nbreak".to_string()];
    assert_eq!(json_string_array(&args), "[\"quote\\\"\",\"line\\nbreak\"]");
}

fn synthetic_header_for_entries(
    architecture: nerva_model::hf::architecture::HfArchitectureKind,
    entries: &[nerva_model::weights::manifest::HfTensorManifestEntry],
) -> String {
    let total_weight_bytes = entries.iter().map(|entry| entry.bytes).sum();
    let manifest = nerva_model::weights::manifest::HfTensorManifest {
        architecture,
        entries: entries.to_vec(),
        total_weight_bytes,
        manifest_hash: 0,
    };
    nerva_model::weights::safetensors::synthetic_safetensors_header_for_manifest(&manifest).unwrap()
}

fn synthetic_index_json(
    manifest: &nerva_model::weights::manifest::HfTensorManifest,
    split_at: usize,
) -> String {
    let mut out = format!(
        "{{\"metadata\":{{\"total_size\":{}}},\"weight_map\":{{",
        manifest.total_weight_bytes
    );
    for (index, entry) in manifest.entries.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&entry.name);
        out.push_str("\":\"");
        out.push_str(if index < split_at {
            SHARD_ONE
        } else {
            SHARD_TWO
        });
        out.push('"');
    }
    out.push_str("}}");
    out
}

fn write_safetensors_header(path: &Path, header: &str, payload_bytes: usize) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(8 + header.len() + payload_bytes, 0);
    std::fs::write(path, bytes).unwrap();
}
