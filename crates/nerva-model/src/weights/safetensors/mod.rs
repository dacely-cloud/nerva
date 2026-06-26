pub mod planner;

mod parse;
mod tensor;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::MemoryTier;

use crate::common::hash::hash_bytes;
use crate::common::json::{json_escape, json_opt_str, json_opt_usize};
use crate::weights::manifest::{HfTensorManifest, hf_tensor_manifest_probe};

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
    pub role: crate::weights::layout::WeightBlockRole,
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
        let tensor_json = crate::common::json::find_top_level_json_value(header_json, &entry.name)?
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("safetensors header is missing tensor {}", entry.name),
            })?;
        tensor::validate_safetensors_tensor_header(tensor_json, entry)?;
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
        header.push_str(tensor::safetensors_dtype(entry.dtype)?);
        header.push_str("\",\"shape\":");
        tensor::push_safetensors_shape_json(entry, &mut header);
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
