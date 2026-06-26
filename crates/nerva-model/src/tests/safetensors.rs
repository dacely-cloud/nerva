use crate::hf::parser::parse_hf_config_metadata;
use crate::tests::support::{
    SHARD_ONE, SHARD_TWO, synthetic_header_for_entries, synthetic_sharded_index_json,
    tiny_llama_manifest,
};
use crate::weights::file::{read_safetensors_header_file, read_safetensors_header_file_with_limit};
use crate::weights::layout::plan_hf_weight_layout;
use crate::weights::manifest::build_hf_tensor_manifest;
use crate::weights::safetensors::planner::{
    plan_safetensors_shards_for_manifest, required_safetensors_shards_for_manifest,
};
use crate::weights::safetensors::{
    SafetensorsShardHeader, SafetensorsValidationStatus, safetensors_header_from_bytes,
    safetensors_header_probe, synthetic_safetensors_header_for_manifest,
    validate_safetensors_header_for_manifest,
};

#[test]
fn validates_synthetic_safetensors_header_against_manifest() {
    let manifest = crate::weights::manifest::hf_tensor_manifest_probe()
        .unwrap()
        .manifest;
    let header = synthetic_safetensors_header_for_manifest(&manifest).unwrap();
    let validation = validate_safetensors_header_for_manifest(&header, &manifest).unwrap();

    assert_eq!(validation.status, SafetensorsValidationStatus::Ok);
    assert_eq!(validation.manifest_entries, manifest.entries.len());
    assert_eq!(validation.validated_tensors, manifest.entries.len());
    assert_eq!(validation.total_data_bytes, manifest.total_weight_bytes);
    assert_eq!(validation.manifest_hash, manifest.manifest_hash);
    assert_ne!(validation.header_hash, 0);
}

#[test]
fn extracts_safetensors_header_from_file_bytes() {
    let header = "{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0xaa, 0xbb]);

    assert_eq!(safetensors_header_from_bytes(&bytes).unwrap(), header);
    assert!(safetensors_header_from_bytes(&bytes[..4]).is_err());
}

#[test]
fn reads_safetensors_file_header_without_payload_scan() {
    let dir = std::env::temp_dir().join(format!("nerva-model-header-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let header = "{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd]);
    std::fs::write(&path, bytes).unwrap();

    let file_header = read_safetensors_header_file(&path).unwrap();

    assert_eq!(file_header.header_json, header);
    assert_eq!(file_header.header_bytes, header.len());
    assert_eq!(file_header.data_start, 8 + header.len());
    assert_eq!(file_header.payload_bytes, 4);
    assert!(file_header.require_payload_bytes(4).is_ok());
    assert!(file_header.require_payload_bytes(5).is_err());
    assert!(
        file_header
            .require_file_offset_end(8 + header.len() + 4)
            .is_ok()
    );
    assert!(
        file_header
            .require_file_offset_end(8 + header.len() + 5)
            .is_err()
    );
    assert!(file_header.to_json().contains("\"payload_bytes\":4"));

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn safetensors_file_header_rejects_oversized_header_limit() {
    let dir = std::env::temp_dir().join(format!(
        "nerva-model-header-limit-test-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let header = "{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    std::fs::write(&path, bytes).unwrap();

    assert!(read_safetensors_header_file_with_limit(&path, header.len() - 1).is_err());

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn safetensors_validation_rejects_missing_and_mismatched_tensors() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();
    let plan = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    let valid = synthetic_safetensors_header_for_manifest(&manifest).unwrap();

    assert!(validate_safetensors_header_for_manifest("{}", &manifest).is_err());

    let first = &manifest.entries[0];
    let bad_dtype = format!(
        "{{\"{}\":{{\"dtype\":\"F32\",\"shape\":[{},{}],\"data_offsets\":[0,{}]}}}}",
        first.name, first.rows, first.cols, first.bytes
    );
    assert!(validate_safetensors_header_for_manifest(&bad_dtype, &manifest).is_err());

    let bad_shape = valid.replacen(
        &format!("\"shape\":[{},{}]", first.rows, first.cols),
        "\"shape\":[1,1]",
        1,
    );
    assert!(validate_safetensors_header_for_manifest(&bad_shape, &manifest).is_err());
}

#[test]
fn safetensors_header_probe_reports_manifest_parity() {
    let summary = safetensors_header_probe().unwrap();

    assert_eq!(summary.status, SafetensorsValidationStatus::Ok);
    assert_eq!(summary.validation.manifest_entries, 290);
    assert_eq!(summary.validation.validated_tensors, 290);
    assert_eq!(summary.validation.total_data_bytes, 11_866_210_304);
    assert_ne!(summary.validation.header_hash, 0);
    assert!(summary.to_json().contains("\"validated_tensors\":290"));
}

#[test]
fn plans_safetensors_shards_from_index_and_headers() {
    let manifest = tiny_llama_manifest(false);
    let index = synthetic_sharded_index_json(&manifest, 10);
    let header_one = synthetic_header_for_entries(manifest.architecture, &manifest.entries[..10]);
    let header_two = synthetic_header_for_entries(manifest.architecture, &manifest.entries[10..]);

    let required = required_safetensors_shards_for_manifest(&index, &manifest).unwrap();
    let plan = plan_safetensors_shards_for_manifest(
        &index,
        &[
            SafetensorsShardHeader::new(SHARD_ONE, &header_one),
            SafetensorsShardHeader::new(SHARD_TWO, &header_two),
        ],
        &manifest,
    )
    .unwrap();

    assert_eq!(required, vec![SHARD_ONE.to_string(), SHARD_TWO.to_string()]);
    assert_eq!(plan.entries.len(), 20);
    assert_eq!(plan.shards.len(), 2);
    assert_eq!(plan.total_weight_bytes, 768);
    assert_eq!(plan.index_total_size, Some(768));
    assert_eq!(plan.shards[0].file_name, SHARD_ONE);
    assert_eq!(plan.shards[0].tensor_count, 10);
    assert_eq!(plan.shards[0].payload_bytes, 384);
    assert_eq!(plan.entries[0].tensor_name, "model.embed_tokens.weight");
    assert_eq!(plan.entries[0].file_offset_begin, 8 + header_one.len());
    assert_eq!(
        plan.entries[0].file_offset_end,
        8 + header_one.len() + plan.entries[0].bytes
    );
    assert_eq!(plan.entries[10].shard_file, SHARD_TWO);
    assert_ne!(plan.plan_hash, 0);
    assert!(plan.to_json().contains("\"shards\":2"));
}

#[test]
fn safetensors_shard_plan_supports_tied_embedding_manifest() {
    let manifest = tiny_llama_manifest(true);
    let index = synthetic_sharded_index_json(&manifest, 10);
    let header_one = synthetic_header_for_entries(manifest.architecture, &manifest.entries[..10]);
    let header_two = synthetic_header_for_entries(manifest.architecture, &manifest.entries[10..]);
    let plan = plan_safetensors_shards_for_manifest(
        &index,
        &[
            SafetensorsShardHeader::new(SHARD_ONE, &header_one),
            SafetensorsShardHeader::new(SHARD_TWO, &header_two),
        ],
        &manifest,
    )
    .unwrap();

    assert_eq!(plan.entries.len(), 19);
    assert_eq!(plan.total_weight_bytes, 688);
    assert_eq!(
        plan.entries.last().unwrap().tensor_name,
        "model.layers.1.mlp.down_proj.weight"
    );
    assert!(
        !plan
            .entries
            .iter()
            .any(|entry| entry.tensor_name == "lm_head.weight")
    );
}

#[test]
fn safetensors_shard_plan_rejects_missing_index_or_header() {
    let manifest = tiny_llama_manifest(false);
    let index = synthetic_sharded_index_json(&manifest, 10);
    let header_one = synthetic_header_for_entries(manifest.architecture, &manifest.entries[..10]);
    let missing_lm_head_index = index.replace(
        "\"lm_head.weight\":\"model-00002-of-00002.safetensors\"",
        "\"unused.weight\":\"model-00002-of-00002.safetensors\"",
    );

    assert!(required_safetensors_shards_for_manifest(&missing_lm_head_index, &manifest).is_err());
    assert!(
        plan_safetensors_shards_for_manifest(
            &index,
            &[SafetensorsShardHeader::new(SHARD_ONE, &header_one)],
            &manifest,
        )
        .is_err()
    );
    assert!(
        plan_safetensors_shards_for_manifest(
            &index,
            &[
                SafetensorsShardHeader::new(SHARD_ONE, &header_one),
                SafetensorsShardHeader::new(SHARD_ONE, &header_one),
            ],
            &manifest,
        )
        .is_err()
    );
}
