use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_model::hf::contract::{validate_exact_runtime_contract, validate_weight_layout_contract};
use nerva_model::hf::metadata::HfModelMetadata;
use nerva_model::hf::parser::parse_hf_config_metadata;
use nerva_model::weights::file::read_safetensors_header_file;
use nerva_model::weights::layout::plan::{
    plan_hf_weight_layout, plan_hf_weight_layout_for_safetensors_index,
};
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
    validate_shard_backed_weight_contract(&metadata)?;
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
    let default_layout = plan_hf_weight_layout(&metadata)?;
    let default_manifest = build_hf_tensor_manifest(&default_layout)?;
    let index_json = load_or_synthesize_index(dir, &default_manifest)?;
    let layout = plan_hf_weight_layout_for_safetensors_index(&metadata, &index_json)?;
    let manifest = build_hf_tensor_manifest(&layout)?;
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

fn validate_shard_backed_weight_contract(metadata: &HfModelMetadata) -> Result<()> {
    if metadata.architecture.is_deepseek() {
        return validate_weight_layout_contract(metadata);
    }
    validate_exact_runtime_contract(metadata)
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

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use nerva_model::hf::architecture::HfArchitectureKind;
    use nerva_model::weights::safetensors::header::synthetic_safetensors_header_for_manifest;

    use super::*;

    #[test]
    fn load_shard_backed_weights_accepts_deepseek_layout_contract() {
        let dir = temp_checkpoint_dir("nerva-runtime-deepseek-load");
        write_deepseek_v3_checkpoint(&dir);

        let weights = load_shard_backed_weights(&dir).unwrap();

        assert_eq!(
            weights.metadata.architecture,
            HfArchitectureKind::DeepSeekV3
        );
        assert!(weights.manifest.total_weight_bytes > 0);
        assert_eq!(weights.shard_plan.shards.len(), 1);
        assert!(weights.data_hash_available);

        let _ = std::fs::remove_file(dir.join("model.safetensors"));
        let _ = std::fs::remove_file(dir.join("config.json"));
        let _ = std::fs::remove_dir(dir);
    }

    fn temp_checkpoint_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_deepseek_v3_checkpoint(dir: &Path) {
        std::fs::write(dir.join("config.json"), deepseek_v3_fixture_config()).unwrap();
        let metadata = parse_hf_config_metadata(deepseek_v3_fixture_config()).unwrap();
        let layout = plan_hf_weight_layout(&metadata).unwrap();
        let manifest = build_hf_tensor_manifest(&layout).unwrap();
        let header = synthetic_safetensors_header_for_manifest(&manifest).unwrap();
        let mut bytes = Vec::with_capacity(8 + header.len() + manifest.total_weight_bytes);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + manifest.total_weight_bytes, 0);
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    }

    fn deepseek_v3_fixture_config() -> &'static str {
        r#"{
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 4,
            "intermediate_size": 8,
            "moe_intermediate_size": 2,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "n_group": 1,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "q_lora_rank": 2,
            "kv_lora_rank": 2,
            "qk_nope_head_dim": 1,
            "qk_rope_head_dim": 1,
            "v_head_dim": 1,
            "vocab_size": 8,
            "max_position_embeddings": 128,
            "rms_norm_eps": 0.000001,
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16"
        }"#
    }
}
