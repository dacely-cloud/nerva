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
