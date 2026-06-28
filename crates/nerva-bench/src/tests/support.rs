use std::path::Path;

pub(crate) const SHARD_ONE: &str = "model-00001-of-00002.safetensors";
pub(crate) const SHARD_TWO: &str = "model-00002-of-00002.safetensors";

pub(crate) fn synthetic_header_for_entries(
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
    nerva_model::weights::safetensors::header::synthetic_safetensors_header_for_manifest(&manifest)
        .unwrap()
}

pub(crate) fn synthetic_index_json(
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

pub(crate) fn write_safetensors_header(path: &Path, header: &str, payload_bytes: usize) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(8 + header.len() + payload_bytes, 0);
    std::fs::write(path, bytes).unwrap();
}

pub(crate) fn write_tiny_hf_checkpoint_dir(prefix: &str) -> std::path::PathBuf {
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

pub(crate) fn remove_tiny_hf_checkpoint_dir(dir: &std::path::Path) {
    let _ = std::fs::remove_file(dir.join("tokenizer.json"));
    let _ = std::fs::remove_file(dir.join("model.safetensors"));
    let _ = std::fs::remove_file(dir.join("config.json"));
    let _ = std::fs::remove_dir(dir);
}

pub(crate) fn write_tiny_wordlevel_tokenizer(dir: &std::path::Path) {
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
