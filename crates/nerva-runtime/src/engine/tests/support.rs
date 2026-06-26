pub(crate) const SHARD_ONE: &str = "model-00001-of-00001.safetensors";

pub(crate) fn tiny_llama_manifest() -> nerva_model::weights::manifest::HfTensorManifest {
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(
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
    let layout = nerva_model::weights::layout::plan_hf_weight_layout(&metadata).unwrap();
    nerva_model::weights::manifest::build_hf_tensor_manifest(&layout).unwrap()
}

fn single_shard_index_json(manifest: &nerva_model::weights::manifest::HfTensorManifest) -> String {
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
        out.push_str(SHARD_ONE);
        out.push('"');
    }
    out.push_str("}}");
    out
}

pub(crate) fn tiny_shard_plan() -> (
    nerva_model::weights::safetensors::SafetensorsShardPlan,
    usize,
) {
    let (plan, header) = tiny_shard_plan_with_header();
    let header_len = header.len();
    (plan, header_len)
}

pub(crate) fn tiny_shard_plan_with_header() -> (
    nerva_model::weights::safetensors::SafetensorsShardPlan,
    String,
) {
    let manifest = tiny_llama_manifest();
    let index = single_shard_index_json(&manifest);
    let header =
        nerva_model::weights::safetensors::synthetic_safetensors_header_for_manifest(&manifest)
            .unwrap();
    let plan = nerva_model::weights::safetensors::planner::plan_safetensors_shards_for_manifest(
        &index,
        &[nerva_model::weights::safetensors::SafetensorsShardHeader::new(SHARD_ONE, &header)],
        &manifest,
    )
    .unwrap();
    (plan, header)
}

pub(crate) fn write_tiny_shard_file(dir: &std::path::Path, header: &str, payload_bytes: usize) {
    let path = dir.join(SHARD_ONE);
    let mut bytes = Vec::with_capacity(8 + header.len() + payload_bytes);
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for index in 0..payload_bytes {
        bytes.push(((index * 31 + 7) % 251) as u8);
    }
    std::fs::write(path, bytes).unwrap();
}
