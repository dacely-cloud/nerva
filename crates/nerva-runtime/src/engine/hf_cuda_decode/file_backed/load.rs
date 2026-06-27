use std::path::Path;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_model::hf::metadata::HfModelMetadata;
use nerva_model::hf::parser::parse_hf_config_metadata;
use nerva_model::weights::file::read_safetensors_header_file;
use nerva_model::weights::layout::plan::plan_hf_weight_layout;
use nerva_model::weights::manifest::{HfTensorManifest, build_hf_tensor_manifest};
use nerva_model::weights::safetensors::planner::{
    plan_safetensors_shards_for_manifest, required_safetensors_shards_for_manifest,
};
use nerva_model::weights::safetensors::shard::{
    SafetensorsShardHeader, SafetensorsShardPlan, SafetensorsShardPlanEntry,
};

pub(super) struct ShardBackedWeights {
    pub metadata: HfModelMetadata,
    pub dtype: DType,
    pub manifest: HfTensorManifest,
    pub shard_plan: SafetensorsShardPlan,
    pub buffers: Vec<ShardBuffer>,
}

pub(super) struct ShardBuffer {
    pub file_name: String,
    pub bytes: Vec<u8>,
}

impl ShardBackedWeights {
    pub fn source_bytes(&self, entry: &SafetensorsShardPlanEntry) -> Result<&[u8]> {
        if entry.bytes % 2 != 0 {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "safetensors tensor {} byte count is not u16 aligned",
                    entry.tensor_name
                ),
            });
        }
        let buffer = self
            .buffers
            .iter()
            .find(|buffer| buffer.file_name == entry.shard_file)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("safetensors shard {} was not loaded", entry.shard_file),
            })?;
        if entry.file_offset_end > buffer.bytes.len() {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "safetensors tensor {} exceeds shard bounds",
                    entry.tensor_name
                ),
            });
        }
        let bytes = &buffer.bytes[entry.file_offset_begin..entry.file_offset_end];
        if bytes.len() != entry.bytes {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "safetensors tensor {} byte length changed",
                    entry.tensor_name
                ),
            });
        }
        Ok(bytes)
    }
}

pub(super) fn load_shard_backed_weights(dir: &Path) -> Result<ShardBackedWeights> {
    let config_path = dir.join("config.json");
    let config =
        std::fs::read_to_string(&config_path).map_err(|err| NervaError::InvalidArgument {
            reason: format!(
                "failed to read HF config from {}: {err}",
                config_path.display()
            ),
        })?;
    let metadata = parse_hf_config_metadata(&config)?;
    if metadata.tie_word_embeddings {
        return Err(NervaError::InvalidArgument {
            reason: "CUDA descriptor device-only path requires a materialized lm_head".to_string(),
        });
    }
    let dtype = metadata
        .torch_dtype
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "HF CUDA descriptor load requires torch_dtype".to_string(),
        })?;
    let layout = plan_hf_weight_layout(&metadata)?;
    let manifest = build_hf_tensor_manifest(&layout)?;
    let index_json = load_or_synthesize_index(dir, &manifest)?;
    let header_store = read_required_headers(dir, &index_json, &manifest)?;
    let shard_headers = header_store
        .iter()
        .map(|(name, header)| SafetensorsShardHeader::new(name.as_str(), header.as_str()))
        .collect::<Vec<_>>();
    let shard_plan = plan_safetensors_shards_for_manifest(&index_json, &shard_headers, &manifest)?;
    let buffers = read_required_buffers(dir, &shard_plan)?;
    Ok(ShardBackedWeights {
        metadata,
        dtype,
        manifest,
        shard_plan,
        buffers,
    })
}

fn load_or_synthesize_index(dir: &Path, manifest: &HfTensorManifest) -> Result<String> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        return std::fs::read_to_string(&index_path).map_err(|err| NervaError::InvalidArgument {
            reason: format!("failed to read {}: {err}", index_path.display()),
        });
    }
    if dir.join("model.safetensors").exists() {
        return Ok(single_shard_index_json(manifest));
    }
    Err(NervaError::InvalidArgument {
        reason: format!("HF directory {} has no safetensors model", dir.display()),
    })
}

fn read_required_headers(
    dir: &Path,
    index_json: &str,
    manifest: &HfTensorManifest,
) -> Result<Vec<(String, String)>> {
    required_safetensors_shards_for_manifest(index_json, manifest)?
        .into_iter()
        .map(|shard| {
            let file = read_safetensors_header_file(dir.join(&shard))?;
            Ok((shard, file.header_json))
        })
        .collect()
}

fn read_required_buffers(dir: &Path, plan: &SafetensorsShardPlan) -> Result<Vec<ShardBuffer>> {
    plan.shards
        .iter()
        .map(|shard| {
            let path = dir.join(&shard.file_name);
            let bytes = std::fs::read(&path).map_err(|err| NervaError::InvalidArgument {
                reason: format!("failed to read safetensors shard {}: {err}", path.display()),
            })?;
            Ok(ShardBuffer {
                file_name: shard.file_name.clone(),
                bytes,
            })
        })
        .collect()
}

fn single_shard_index_json(manifest: &HfTensorManifest) -> String {
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
        out.push_str("\":\"model.safetensors\"");
    }
    out.push_str("}}");
    out
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}
