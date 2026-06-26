use std::collections::BTreeMap;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::MemoryTier;

use crate::common::dtype::dtype_to_str;
use crate::common::hash::hash_bytes;
use crate::common::json::{
    find_json_value_end, find_top_level_json_value, json_escape, json_opt_str, json_opt_usize,
    optional_string, optional_usize, parse_json_string_at, parse_json_string_value, skip_json_ws,
};
use crate::weights::hash::hash_safetensors_shard_plan;
use crate::weights::layout::WeightBlockRole;
use crate::weights::manifest::{HfTensorManifest, HfTensorManifestEntry, hf_tensor_manifest_probe};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SafetensorsValidationStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SafetensorsManifestValidationSummary {
    pub status: SafetensorsValidationStatus,
    pub manifest_entries: usize,
    pub validated_tensors: usize,
    pub total_data_bytes: usize,
    pub header_bytes: usize,
    pub manifest_hash: u64,
    pub header_hash: u64,
}

impl SafetensorsManifestValidationSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SafetensorsValidationStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"manifest_entries\":{},\"validated_tensors\":{},\"total_data_bytes\":{},\"header_bytes\":{},\"manifest_hash\":{},\"header_hash\":{}}}",
            status,
            self.manifest_entries,
            self.validated_tensors,
            self.total_data_bytes,
            self.header_bytes,
            self.manifest_hash,
            self.header_hash,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SafetensorsHeaderProbeSummary {
    pub status: SafetensorsValidationStatus,
    pub validation: SafetensorsManifestValidationSummary,
}

impl SafetensorsHeaderProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SafetensorsValidationStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"validation\":{}}}",
            status,
            self.validation.to_json(),
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SafetensorsShardHeader<'a> {
    pub file_name: &'a str,
    pub header_json: &'a str,
}

impl<'a> SafetensorsShardHeader<'a> {
    pub const fn new(file_name: &'a str, header_json: &'a str) -> Self {
        Self {
            file_name,
            header_json,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SafetensorsShardPlanEntry {
    pub tensor_name: String,
    pub shard_file: String,
    pub role: WeightBlockRole,
    pub layer: Option<u32>,
    pub dtype: DType,
    pub tier: MemoryTier,
    pub bytes: usize,
    pub data_offset_begin: usize,
    pub data_offset_end: usize,
    pub file_offset_begin: usize,
    pub file_offset_end: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SafetensorsShardPlanShard {
    pub file_name: String,
    pub tensor_count: usize,
    pub payload_bytes: usize,
    pub header_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SafetensorsShardPlan {
    pub entries: Vec<SafetensorsShardPlanEntry>,
    pub shards: Vec<SafetensorsShardPlanShard>,
    pub total_weight_bytes: usize,
    pub index_total_size: Option<usize>,
    pub manifest_hash: u64,
    pub index_hash: u64,
    pub plan_hash: u64,
}

impl SafetensorsShardPlan {
    pub fn to_json(&self) -> String {
        let first = self.entries.first().map(|entry| entry.tensor_name.as_str());
        let last = self.entries.last().map(|entry| entry.tensor_name.as_str());
        format!(
            "{{\"entries\":{},\"shards\":{},\"total_weight_bytes\":{},\"index_total_size\":{},\"first_tensor\":{},\"last_tensor\":{},\"manifest_hash\":{},\"index_hash\":{},\"plan_hash\":{}}}",
            self.entries.len(),
            self.shards.len(),
            self.total_weight_bytes,
            json_opt_usize(self.index_total_size),
            json_opt_str(first),
            json_opt_str(last),
            self.manifest_hash,
            self.index_hash,
            self.plan_hash,
        )
    }
}

pub fn safetensors_header_from_bytes(bytes: &[u8]) -> Result<&str> {
    if bytes.len() < 8 {
        return Err(NervaError::InvalidArgument {
            reason: "safetensors file is too small to contain a header length".to_string(),
        });
    }
    let mut header_len_bytes = [0u8; 8];
    header_len_bytes.copy_from_slice(&bytes[..8]);
    let header_len = u64::from_le_bytes(header_len_bytes);
    let header_len = usize::try_from(header_len).map_err(|_| NervaError::InvalidArgument {
        reason: "safetensors header length does not fit usize".to_string(),
    })?;
    let header_end = 8usize
        .checked_add(header_len)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "safetensors header length overflows file offset".to_string(),
        })?;
    if header_end > bytes.len() {
        return Err(NervaError::InvalidArgument {
            reason: "safetensors header length exceeds available bytes".to_string(),
        });
    }
    core::str::from_utf8(&bytes[8..header_end]).map_err(|_| NervaError::InvalidArgument {
        reason: "safetensors header is not valid UTF-8".to_string(),
    })
}

pub fn validate_safetensors_header_for_manifest(
    header_json: &str,
    manifest: &HfTensorManifest,
) -> Result<SafetensorsManifestValidationSummary> {
    let mut validated_tensors = 0usize;
    let mut total_data_bytes = 0usize;
    for entry in &manifest.entries {
        let tensor_json =
            find_top_level_json_value(header_json, &entry.name)?.ok_or_else(|| {
                NervaError::InvalidArgument {
                    reason: format!("safetensors header is missing tensor {}", entry.name),
                }
            })?;
        validate_safetensors_tensor_header(tensor_json, entry)?;
        validated_tensors =
            validated_tensors
                .checked_add(1)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes: 1,
                    reason: "validated tensor count overflow".to_string(),
                })?;
        total_data_bytes = total_data_bytes.checked_add(entry.bytes).ok_or_else(|| {
            NervaError::AllocationFailed {
                bytes: entry.bytes,
                reason: "safetensors validated data byte count overflow".to_string(),
            }
        })?;
    }
    if total_data_bytes != manifest.total_weight_bytes {
        return Err(NervaError::InvalidArgument {
            reason: "safetensors validated byte count does not match manifest".to_string(),
        });
    }
    Ok(SafetensorsManifestValidationSummary {
        status: SafetensorsValidationStatus::Ok,
        manifest_entries: manifest.entries.len(),
        validated_tensors,
        total_data_bytes,
        header_bytes: header_json.len(),
        manifest_hash: manifest.manifest_hash,
        header_hash: hash_bytes(header_json.as_bytes()),
    })
}

pub fn synthetic_safetensors_header_for_manifest(manifest: &HfTensorManifest) -> Result<String> {
    let mut header = String::from("{");
    for (index, entry) in manifest.entries.iter().enumerate() {
        if index > 0 {
            header.push(',');
        }
        let begin = manifest.entries[..index]
            .iter()
            .try_fold(0usize, |acc, item| {
                acc.checked_add(item.bytes)
                    .ok_or_else(|| NervaError::AllocationFailed {
                        bytes: item.bytes,
                        reason: "synthetic safetensors offset overflow".to_string(),
                    })
            })?;
        let end = begin
            .checked_add(entry.bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: entry.bytes,
                reason: "synthetic safetensors tensor end overflow".to_string(),
            })?;
        header.push('"');
        header.push_str(&json_escape(&entry.name));
        header.push_str("\":{\"dtype\":\"");
        header.push_str(safetensors_dtype(entry.dtype)?);
        header.push_str("\",\"shape\":");
        push_safetensors_shape_json(entry, &mut header);
        header.push_str(",\"data_offsets\":[");
        header.push_str(&begin.to_string());
        header.push(',');
        header.push_str(&end.to_string());
        header.push_str("]}");
    }
    header.push('}');
    Ok(header)
}

pub fn safetensors_header_probe() -> Result<SafetensorsHeaderProbeSummary> {
    let manifest = hf_tensor_manifest_probe()?.manifest;
    let header = synthetic_safetensors_header_for_manifest(&manifest)?;
    let validation = validate_safetensors_header_for_manifest(&header, &manifest)?;
    Ok(SafetensorsHeaderProbeSummary {
        status: SafetensorsValidationStatus::Ok,
        validation,
    })
}

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

pub(crate) fn validate_safetensors_tensor_header(
    tensor_json: &str,
    entry: &HfTensorManifestEntry,
) -> Result<()> {
    let _ = safetensors_tensor_data_offsets(tensor_json, entry)?;
    Ok(())
}

pub(crate) fn safetensors_tensor_data_offsets(
    tensor_json: &str,
    entry: &HfTensorManifestEntry,
) -> Result<(usize, usize)> {
    let dtype =
        optional_string(tensor_json, "dtype")?.ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("safetensors tensor {} is missing dtype", entry.name),
        })?;
    if dtype != safetensors_dtype(entry.dtype)? {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "safetensors tensor {} dtype {} does not match expected {}",
                entry.name,
                dtype,
                safetensors_dtype(entry.dtype)?
            ),
        });
    }

    let shape = required_usize_array(tensor_json, "shape")?;
    if shape != expected_safetensors_shape(entry) {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "safetensors tensor {} shape {:?} does not match expected {:?}",
                entry.name,
                shape,
                expected_safetensors_shape(entry)
            ),
        });
    }

    let offsets = required_usize_array(tensor_json, "data_offsets")?;
    if offsets.len() != 2 || offsets[1] < offsets[0] {
        return Err(NervaError::InvalidArgument {
            reason: format!("safetensors tensor {} has invalid offsets", entry.name),
        });
    }
    let span = offsets[1] - offsets[0];
    if span != entry.bytes {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "safetensors tensor {} offset span {} does not match expected bytes {}",
                entry.name, span, entry.bytes
            ),
        });
    }
    Ok((offsets[0], offsets[1]))
}

pub(crate) fn safetensors_file_data_start(header_json: &str) -> Result<usize> {
    8usize
        .checked_add(header_json.len())
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: header_json.len(),
            reason: "safetensors header file offset overflow".to_string(),
        })
}

pub(crate) fn required_usize_array(source: &str, key: &'static str) -> Result<Vec<usize>> {
    let value =
        find_top_level_json_value(source, key)?.ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("JSON object is missing required array {key}"),
        })?;
    parse_usize_array_value(value, key)
}

pub(crate) fn parse_usize_array_value(value: &str, key: &'static str) -> Result<Vec<usize>> {
    let value = value.trim();
    if !value.starts_with('[') || !value.ends_with(']') {
        return Err(NervaError::InvalidArgument {
            reason: format!("JSON field {key} must be an unsigned integer array"),
        });
    }
    let inner = value[1..value.len() - 1].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|part| {
            let part = part.trim();
            if part.starts_with('-') || part.is_empty() {
                return Err(NervaError::InvalidArgument {
                    reason: format!("JSON field {key} must contain unsigned integers"),
                });
            }
            let parsed = part
                .parse::<u64>()
                .map_err(|_| NervaError::InvalidArgument {
                    reason: format!("JSON field {key} must contain unsigned integers"),
                })?;
            usize::try_from(parsed).map_err(|_| NervaError::InvalidArgument {
                reason: format!("JSON field {key} value does not fit usize"),
            })
        })
        .collect()
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

pub(crate) fn parse_json_string_map_value(
    value: &str,
    key: &'static str,
) -> Result<Vec<(String, String)>> {
    let value = value.trim();
    let bytes = value.as_bytes();
    let mut index = skip_json_ws(bytes, 0);
    if index >= bytes.len() || bytes[index] != b'{' {
        return Err(NervaError::InvalidArgument {
            reason: format!("JSON field {key} must be an object"),
        });
    }
    index += 1;
    let mut entries = Vec::new();
    loop {
        index = skip_json_ws(bytes, index);
        if index >= bytes.len() {
            return Err(NervaError::InvalidArgument {
                reason: format!("JSON field {key} object is not closed"),
            });
        }
        if bytes[index] == b'}' {
            return Ok(entries);
        }
        if bytes[index] == b',' {
            index += 1;
            continue;
        }
        if bytes[index] != b'"' {
            return Err(NervaError::InvalidArgument {
                reason: format!("JSON field {key} object key must be a string"),
            });
        }
        let (field, after_key) = parse_json_string_at(value, index)?;
        index = skip_json_ws(bytes, after_key);
        if index >= bytes.len() || bytes[index] != b':' {
            return Err(NervaError::InvalidArgument {
                reason: format!("JSON field {key} object key is missing ':'"),
            });
        }
        index = skip_json_ws(bytes, index + 1);
        let value_start = index;
        let value_end = find_json_value_end(value, value_start)?;
        let mapped = parse_json_string_value(&value[value_start..value_end])?;
        entries.push((field, mapped));
        index = value_end;
    }
}

pub(crate) fn safetensors_dtype(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F16 => Ok("F16"),
        DType::BF16 => Ok("BF16"),
        DType::F32 => Ok("F32"),
        _ => Err(NervaError::InvalidArgument {
            reason: format!(
                "dtype {} is not supported in exact safetensors manifest validation",
                dtype_to_str(dtype)
            ),
        }),
    }
}

pub(crate) fn expected_safetensors_shape(entry: &HfTensorManifestEntry) -> Vec<usize> {
    match entry.rank {
        1 => vec![entry.rows],
        2 => vec![entry.rows, entry.cols],
        _ => Vec::new(),
    }
}

pub(crate) fn push_safetensors_shape_json(entry: &HfTensorManifestEntry, out: &mut String) {
    match entry.rank {
        1 => {
            out.push('[');
            out.push_str(&entry.rows.to_string());
            out.push(']');
        }
        2 => {
            out.push('[');
            out.push_str(&entry.rows.to_string());
            out.push(',');
            out.push_str(&entry.cols.to_string());
            out.push(']');
        }
        _ => out.push_str("[]"),
    }
}
