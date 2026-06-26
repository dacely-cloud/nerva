use nerva_core::types::error::{NervaError, Result};

use crate::common::hash::hash_bytes;
use crate::common::json::parse::find_top_level_json_value;
use crate::weights::manifest::HfTensorManifest;
use crate::weights::safetensors::tensor;

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
