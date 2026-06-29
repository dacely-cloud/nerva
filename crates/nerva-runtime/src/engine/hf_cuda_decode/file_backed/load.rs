use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_model::hf::contract::validate_exact_runtime_contract;
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

const STREAM_HASH_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const STREAM_HASH_CHUNK_BYTES: usize = 8 * 1024 * 1024;

pub(super) struct ShardBackedWeights {
    pub dir: PathBuf,
    pub metadata: HfModelMetadata,
    pub dtype: DType,
    pub manifest: HfTensorManifest,
    pub shard_plan: SafetensorsShardPlan,
    pub data_hash: u64,
    pub data_hash_available: bool,
}

impl ShardBackedWeights {
    pub fn source_path(&self, entry: &SafetensorsShardPlanEntry) -> Result<PathBuf> {
        if entry.bytes % 2 != 0 {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "safetensors tensor {} byte count is not u16 aligned",
                    entry.tensor_name
                ),
            });
        }
        let path = self.dir.join(&entry.shard_file);
        let len = std::fs::metadata(&path)
            .map_err(|err| NervaError::InvalidArgument {
                reason: format!("failed to stat safetensors shard {}: {err}", path.display()),
            })?
            .len();
        if entry.file_offset_end as u64 > len {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "safetensors tensor {} exceeds shard bounds",
                    entry.tensor_name
                ),
            });
        }
        if entry.file_offset_end - entry.file_offset_begin != entry.bytes {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "safetensors tensor {} byte length changed",
                    entry.tensor_name
                ),
            });
        }
        Ok(path)
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
    validate_exact_runtime_contract(&metadata)?;
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
    let (data_hash, data_hash_available) = maybe_hash_required_buffers(dir, &manifest)?;
    Ok(ShardBackedWeights {
        dir: dir.to_path_buf(),
        metadata,
        dtype,
        manifest,
        shard_plan,
        data_hash,
        data_hash_available,
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

fn maybe_hash_required_buffers(dir: &Path, manifest: &HfTensorManifest) -> Result<(u64, bool)> {
    if manifest.total_weight_bytes > STREAM_HASH_LIMIT_BYTES {
        return Ok((0, false));
    }
    let index_json = load_or_synthesize_index(dir, manifest)?;
    let required_shards = required_safetensors_shards_for_manifest(&index_json, manifest)?;
    let mut data_hash = 0xcbf2_9ce4_8422_2325u64;
    let mut buffer = vec![0u8; STREAM_HASH_CHUNK_BYTES];
    for shard in required_shards {
        let path = dir.join(&shard);
        let mut file = File::open(&path).map_err(|err| NervaError::InvalidArgument {
            reason: format!("failed to open safetensors shard {}: {err}", path.display()),
        })?;
        data_hash = hash_bytes(data_hash, shard.as_bytes());
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|err| NervaError::InvalidArgument {
                    reason: format!("failed to read safetensors shard {}: {err}", path.display()),
                })?;
            if read == 0 {
                break;
            }
            data_hash = hash_bytes(data_hash, &buffer[..read]);
        }
    }
    Ok((data_hash, true))
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

fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
