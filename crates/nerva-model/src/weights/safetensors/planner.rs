use std::collections::BTreeMap;

use nerva_core::types::error::{NervaError, Result};

use crate::common::hash::hash_bytes;
use crate::common::json::fields::optional_usize;
use crate::common::json::parse::find_top_level_json_value;
use crate::weights::hash::hash_safetensors_shard_plan;
use crate::weights::manifest::HfTensorManifest;
use crate::weights::safetensors::parse::parse_json_string_map_value;
use crate::weights::safetensors::shard::{
    SafetensorsShardHeader, SafetensorsShardPlan, SafetensorsShardPlanEntry,
    SafetensorsShardPlanShard,
};
use crate::weights::safetensors::tensor::{
    safetensors_file_data_start, safetensors_tensor_data_offsets,
};

pub fn required_safetensors_shards_for_manifest(
    index_json: &str,
    manifest: &HfTensorManifest,
) -> Result<Vec<String>> {
    let weight_map = parse_safetensors_weight_map(index_json)?;
    let mut shards = BTreeMap::new();
    for entry in &manifest.entries {
        let shard_file = weight_map.tensor_to_shard.get(&entry.name).ok_or_else(|| {
            NervaError::InvalidArgument {
                reason: format!("safetensors index is missing tensor {}", entry.name),
            }
        })?;
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
            .insert(header.file_name, header.header_json)
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
        let shard_file = weight_map.tensor_to_shard.get(&entry.name).ok_or_else(|| {
            NervaError::InvalidArgument {
                reason: format!("safetensors index is missing tensor {}", entry.name),
            }
        })?;
        let header_json =
            header_by_file
                .get(shard_file.as_str())
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: format!("safetensors shard header for {shard_file} was not provided"),
                })?;
        let tensor_json =
            find_top_level_json_value(header_json, &entry.name)?.ok_or_else(|| {
                NervaError::InvalidArgument {
                    reason: format!(
                        "safetensors shard {shard_file} is missing tensor {}",
                        entry.name
                    ),
                }
            })?;
        let (data_offset_begin, data_offset_end) =
            safetensors_tensor_data_offsets(tensor_json, entry)?;
        let file_data_start = safetensors_file_data_start(header_json)?;
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
                    header_bytes: header_json.len(),
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
            tensor_name: entry.name.clone(),
            shard_file: shard_file.clone(),
            role: entry.role,
            layer: entry.layer,
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

#[derive(Clone, Debug, PartialEq)]
struct SafetensorsWeightMap {
    tensor_to_shard: BTreeMap<String, String>,
    total_size: Option<usize>,
}

fn parse_safetensors_weight_map(index_json: &str) -> Result<SafetensorsWeightMap> {
    let weight_map_json =
        find_top_level_json_value(index_json, "weight_map")?.ok_or_else(|| {
            NervaError::InvalidArgument {
                reason: "safetensors index is missing weight_map".to_string(),
            }
        })?;
    let mut tensor_to_shard = BTreeMap::new();
    for (tensor_name, shard_file) in parse_json_string_map_value(weight_map_json, "weight_map")? {
        if tensor_name.is_empty() || shard_file.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "safetensors weight_map entries cannot be empty".to_string(),
            });
        }
        if tensor_to_shard
            .insert(tensor_name.clone(), shard_file)
            .is_some()
        {
            return Err(NervaError::InvalidArgument {
                reason: format!("duplicate safetensors weight_map entry for {tensor_name}"),
            });
        }
    }
    let total_size = match find_top_level_json_value(index_json, "metadata")? {
        Some(metadata_json) => optional_usize(metadata_json, "total_size")?,
        None => None,
    };
    Ok(SafetensorsWeightMap {
        tensor_to_shard,
        total_size,
    })
}
