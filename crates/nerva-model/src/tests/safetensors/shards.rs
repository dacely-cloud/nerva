use crate::tests::support::{
    SHARD_ONE, SHARD_TWO, synthetic_header_for_entries, synthetic_sharded_index_json,
    tiny_llama_manifest,
};
use crate::weights::safetensors::planner::{
    plan_safetensors_shards_for_manifest, required_safetensors_shards_for_manifest,
};
use crate::weights::safetensors::shard::SafetensorsShardHeader;

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
    assert_eq!(plan.entries.len(), 21);
    assert_eq!(plan.shards.len(), 2);
    assert_eq!(plan.total_weight_bytes, 776);
    assert_eq!(plan.index_total_size, Some(776));
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
fn safetensors_shard_plan_ignores_extra_non_manifest_tensors() {
    let manifest = tiny_llama_manifest(false);
    let base_index = synthetic_sharded_index_json(&manifest, 10);
    let index = base_index
        .replace(
            &format!("\"total_size\":{}", manifest.total_weight_bytes),
            &format!("\"total_size\":{}", manifest.total_weight_bytes + 2),
        )
        .replace(
            "}}",
            &format!(",\"model.visual.blocks.0.attn.qkv.weight\":\"{SHARD_ONE}\"}}"),
        );
    let mut header_one =
        synthetic_header_for_entries(manifest.architecture, &manifest.entries[..10]);
    header_one.insert_str(
        header_one.len() - 1,
        ",\"model.visual.blocks.0.attn.qkv.weight\":{\"dtype\":\"F16\",\"shape\":[1,1],\"data_offsets\":[0,2]}",
    );
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
    assert_eq!(plan.entries.len(), manifest.entries.len());
    assert_eq!(plan.total_weight_bytes, manifest.total_weight_bytes);
    assert_eq!(plan.index_total_size, Some(manifest.total_weight_bytes + 2));
    assert_eq!(plan.shards[0].tensor_count, 10);
    assert!(
        !plan
            .entries
            .iter()
            .any(|entry| entry.tensor_name.starts_with("model.visual."))
    );
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

    assert_eq!(plan.entries.len(), 20);
    assert_eq!(plan.total_weight_bytes, 696);
    assert_eq!(
        plan.entries.last().unwrap().tensor_name,
        "model.norm.weight"
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
