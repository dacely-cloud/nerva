use std::path::Path;

use nerva_core::types::error::{NervaError, Result};

use crate::common::json::format::json_escape;
use crate::weights::file::read_safetensors_header_file;
use crate::weights::manifest::HfTensorManifest;
use crate::weights::safetensors::planner::required_safetensors_shards_for_manifest;

pub(crate) fn load_or_synthesize_index(dir: &Path, manifest: &HfTensorManifest) -> Result<String> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        return std::fs::read_to_string(&index_path).map_err(|err| NervaError::InvalidArgument {
            reason: format!("failed to read {}: {err}", index_path.display()),
        });
    }
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(single_shard_index_json(manifest, "model.safetensors"));
    }
    Err(NervaError::InvalidArgument {
        reason: format!(
            "HF model directory {} has no model.safetensors.index.json or model.safetensors",
            dir.display()
        ),
    })
}

pub(crate) fn read_required_headers(
    dir: &Path,
    index_json: &str,
    manifest: &HfTensorManifest,
) -> Result<Vec<(String, String)>> {
    let shards = required_safetensors_shards_for_manifest(index_json, manifest)?;
    let mut headers = Vec::with_capacity(shards.len());
    for shard in shards {
        let path = dir.join(&shard);
        let file = read_safetensors_header_file(&path)?;
        headers.push((shard, file.header_json));
    }
    Ok(headers)
}

fn single_shard_index_json(manifest: &HfTensorManifest, shard: &str) -> String {
    let mut out = format!(
        "{{\"metadata\":{{\"total_size\":{}}},\"weight_map\":{{",
        manifest.total_weight_bytes
    );
    for (index, entry) in manifest.entries.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(&entry.name));
        out.push_str("\":\"");
        out.push_str(&json_escape(shard));
        out.push('"');
    }
    out.push_str("}}");
    out
}
