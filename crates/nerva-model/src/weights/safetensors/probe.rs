use nerva_core::types::error::Result;

use crate::weights::manifest::hf_tensor_manifest_probe;
use crate::weights::safetensors::header::synthetic_safetensors_header_for_manifest;
use crate::weights::safetensors::validation::{
    SafetensorsManifestValidationSummary, SafetensorsValidationStatus,
    validate_safetensors_header_for_manifest,
};

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

pub fn safetensors_header_probe() -> Result<SafetensorsHeaderProbeSummary> {
    let manifest = hf_tensor_manifest_probe()?.manifest;
    let header = synthetic_safetensors_header_for_manifest(&manifest)?;
    let validation = validate_safetensors_header_for_manifest(&header, &manifest)?;
    Ok(SafetensorsHeaderProbeSummary {
        status: SafetensorsValidationStatus::Ok,
        validation,
    })
}
