use std::collections::BTreeMap;

use nerva_core::types::error::{NervaError, Result};

use crate::common::hash::hash_bytes;
use crate::hf::architecture::HfArchitectureKind;
use crate::weights::hash::hash_safetensors_shard_plan;
use crate::weights::layout::entry::WeightBlockRole;
use crate::weights::manifest::{HfTensorManifest, HfTensorManifestEntry};
use crate::weights::safetensors::shard::{
    SafetensorsShardHeader, SafetensorsShardPlan, SafetensorsShardPlanEntry,
    SafetensorsShardPlanShard,
};
use crate::weights::safetensors::tensor::{
    safetensors_file_data_start, safetensors_tensor_data_offsets,
};
use crate::weights::safetensors::weight_map::parse_safetensors_weight_map;

struct IndexedSafetensorsShardHeader<'a> {
    header_json: &'a str,
    tensor_json_by_name: BTreeMap<String, String>,
}

pub fn required_safetensors_shards_for_manifest(
    index_json: &str,
    manifest: &HfTensorManifest,
) -> Result<Vec<String>> {
    let weight_map = parse_safetensors_weight_map(index_json)?;
    let mut shards = BTreeMap::new();
    for entry in &manifest.entries {
        let (tensor_name, shard_file) =
            resolve_index_tensor(manifest.architecture, entry, &weight_map.tensor_to_shard)?;
        let _ = tensor_name;
        shards.insert(shard_file.clone(), ());
    }
    Ok(shards.into_keys().collect())
}

pub fn plan_safetensors_shards_for_manifest(
    index_json: &str,
    shard_headers: &[SafetensorsShardHeader<'_>],
    manifest: &HfTensorManifest,
) -> Result<SafetensorsShardPlan> {
    let weight_map = parse_safetensors_weight_map(index_json)?;
    let mut header_by_file = BTreeMap::new();
    for header in shard_headers {
        if header.file_name.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "safetensors shard header file name cannot be empty".to_string(),
            });
        }
        if header_by_file
            .insert(header.file_name, index_safetensors_header(header)?)
            .is_some()
        {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "duplicate safetensors shard header for {}",
                    header.file_name
                ),
            });
        }
    }

    let mut entries = Vec::with_capacity(manifest.entries.len());
    let mut shard_stats: BTreeMap<String, SafetensorsShardPlanShard> = BTreeMap::new();
    let mut total_weight_bytes = 0usize;

    for entry in &manifest.entries {
        let (tensor_name, shard_file) =
            resolve_index_tensor(manifest.architecture, entry, &weight_map.tensor_to_shard)?;
        let header =
            header_by_file
                .get(shard_file.as_str())
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: format!("safetensors shard header for {shard_file} was not provided"),
                })?;
        let tensor_json = header
            .tensor_json_by_name
            .get(&tensor_name)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!(
                    "safetensors shard {shard_file} is missing tensor {}",
                    tensor_name
                ),
            })?;
        let (data_offset_begin, data_offset_end) =
            safetensors_tensor_data_offsets(tensor_json, entry)?;
        let file_data_start = safetensors_file_data_start(header.header_json)?;
        let file_offset_begin =
            file_data_start
                .checked_add(data_offset_begin)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes: data_offset_begin,
                    reason: "safetensors file offset overflow".to_string(),
                })?;
        let file_offset_end = file_data_start
            .checked_add(data_offset_end)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: data_offset_end,
                reason: "safetensors file offset overflow".to_string(),
            })?;

        total_weight_bytes = total_weight_bytes.checked_add(entry.bytes).ok_or_else(|| {
            NervaError::AllocationFailed {
                bytes: entry.bytes,
                reason: "safetensors shard plan byte count overflow".to_string(),
            }
        })?;
        let shard =
            shard_stats
                .entry(shard_file.clone())
                .or_insert_with(|| SafetensorsShardPlanShard {
                    file_name: shard_file.clone(),
                    tensor_count: 0,
                    payload_bytes: 0,
                    header_bytes: header.header_json.len(),
                });
        shard.tensor_count =
            shard
                .tensor_count
                .checked_add(1)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes: 1,
                    reason: "safetensors shard tensor count overflow".to_string(),
                })?;
        shard.payload_bytes = shard
            .payload_bytes
            .checked_add(entry.bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: entry.bytes,
                reason: "safetensors shard payload byte count overflow".to_string(),
            })?;

        entries.push(SafetensorsShardPlanEntry {
            tensor_name,
            shard_file: shard_file.clone(),
            role: entry.role,
            layer: entry.layer,
            expert: entry.expert,
            dtype: entry.dtype,
            tier: entry.tier,
            bytes: entry.bytes,
            data_offset_begin,
            data_offset_end,
            file_offset_begin,
            file_offset_end,
        });
    }

    if total_weight_bytes != manifest.total_weight_bytes {
        return Err(NervaError::InvalidArgument {
            reason: "safetensors shard plan byte count does not match manifest".to_string(),
        });
    }
    if let Some(index_total_size) = weight_map.total_size {
        if index_total_size < total_weight_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "safetensors index total_size is smaller than required manifest bytes"
                    .to_string(),
            });
        }
    }

    let mut plan = SafetensorsShardPlan {
        entries,
        shards: shard_stats.into_values().collect(),
        total_weight_bytes,
        index_total_size: weight_map.total_size,
        manifest_hash: manifest.manifest_hash,
        index_hash: hash_bytes(index_json.as_bytes()),
        plan_hash: 0,
    };
    plan.plan_hash = hash_safetensors_shard_plan(&plan);
    Ok(plan)
}

pub fn safetensors_index_has_tensor(index_json: &str, tensor_name: &str) -> Result<bool> {
    Ok(parse_safetensors_weight_map(index_json)?
        .tensor_to_shard
        .contains_key(tensor_name))
}

fn resolve_index_tensor(
    architecture: HfArchitectureKind,
    entry: &HfTensorManifestEntry,
    tensor_to_shard: &BTreeMap<String, String>,
) -> Result<(String, String)> {
    for tensor_name in safetensors_tensor_name_candidates(architecture, entry) {
        if let Some(shard_file) = tensor_to_shard.get(&tensor_name) {
            return Ok((tensor_name, shard_file.clone()));
        }
    }
    Err(NervaError::InvalidArgument {
        reason: format!("safetensors index is missing tensor {}", entry.name),
    })
}

fn index_safetensors_header<'a>(
    header: &SafetensorsShardHeader<'a>,
) -> Result<IndexedSafetensorsShardHeader<'a>> {
    let value: serde_json::Value =
        serde_json::from_str(header.header_json).map_err(|err| NervaError::InvalidArgument {
            reason: format!(
                "safetensors shard {} header is not valid JSON: {err}",
                header.file_name
            ),
        })?;
    let object = value
        .as_object()
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!(
                "safetensors shard {} header must be a JSON object",
                header.file_name
            ),
        })?;
    let mut tensor_json_by_name = BTreeMap::new();
    for (name, tensor) in object {
        if name == "__metadata__" {
            continue;
        }
        tensor_json_by_name.insert(name.clone(), tensor.to_string());
    }
    Ok(IndexedSafetensorsShardHeader {
        header_json: header.header_json,
        tensor_json_by_name,
    })
}

fn safetensors_tensor_name_candidates(
    architecture: HfArchitectureKind,
    entry: &HfTensorManifestEntry,
) -> Vec<String> {
    let mut names = vec![entry.name.clone()];
    if architecture != HfArchitectureKind::DeepSeekV4 {
        return names;
    }

    match entry.role {
        WeightBlockRole::DeepSeekV4WqAScale
        | WeightBlockRole::DeepSeekV4WqBScale
        | WeightBlockRole::DeepSeekV4WkvScale
        | WeightBlockRole::DeepSeekV4WoAScale
        | WeightBlockRole::DeepSeekV4WoBScale
        | WeightBlockRole::DeepSeekV4CompressorWkvScale
        | WeightBlockRole::DeepSeekV4CompressorWgateScale
        | WeightBlockRole::DeepSeekV4IndexerWqBScale
        | WeightBlockRole::DeepSeekV4IndexerCompressorWkvScale
        | WeightBlockRole::DeepSeekV4IndexerCompressorWgateScale
        | WeightBlockRole::DeepSeekV4IndexerWeightsScale
        | WeightBlockRole::DeepSeekV4SharedExpertGateScale
        | WeightBlockRole::DeepSeekV4SharedExpertUpScale
        | WeightBlockRole::DeepSeekV4SharedExpertDownScale
        | WeightBlockRole::DeepSeekV4ExpertGateScale
        | WeightBlockRole::DeepSeekV4ExpertUpScale
        | WeightBlockRole::DeepSeekV4ExpertDownScale => {
            push_suffix_alias(&mut names, &entry.name, ".scale", ".weight_scale");
        }
        WeightBlockRole::SharedExpertGateProjection
        | WeightBlockRole::SharedExpertUpProjection
        | WeightBlockRole::SharedExpertDownProjection
        | WeightBlockRole::ExpertGateProjection
        | WeightBlockRole::ExpertUpProjection
        | WeightBlockRole::ExpertDownProjection => {
            push_suffix_alias(&mut names, &entry.name, ".weight", ".weight_packed");
        }
        WeightBlockRole::RouterCorrectionBias => {
            push_suffix_alias(&mut names, &entry.name, ".bias", ".e_score_correction_bias");
        }
        _ => {}
    }
    names
}

fn push_suffix_alias(names: &mut Vec<String>, name: &str, suffix: &str, alias_suffix: &str) {
    if let Some(prefix) = name.strip_suffix(suffix) {
        names.push(format!("{prefix}{alias_suffix}"));
    }
}
