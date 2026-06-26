use crate::weights::file::{read_safetensors_header_file, read_safetensors_header_file_with_limit};
use crate::weights::safetensors::header::{
    safetensors_header_from_bytes, synthetic_safetensors_header_for_manifest,
};
use crate::weights::safetensors::validation::{
    SafetensorsValidationStatus, validate_safetensors_header_for_manifest,
};

#[test]
fn validates_synthetic_safetensors_header_against_manifest() {
    let manifest = crate::weights::manifest::hf_tensor_manifest_probe()
        .unwrap()
        .manifest;
    let header = synthetic_safetensors_header_for_manifest(&manifest).unwrap();
    let validation = validate_safetensors_header_for_manifest(&header, &manifest).unwrap();

    assert_eq!(validation.status, SafetensorsValidationStatus::Ok);
    assert_eq!(validation.manifest_entries, manifest.entries.len());
    assert_eq!(validation.validated_tensors, manifest.entries.len());
    assert_eq!(validation.total_data_bytes, manifest.total_weight_bytes);
    assert_eq!(validation.manifest_hash, manifest.manifest_hash);
    assert_ne!(validation.header_hash, 0);
}

#[test]
fn extracts_safetensors_header_from_file_bytes() {
    let header = "{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0xaa, 0xbb]);

    assert_eq!(safetensors_header_from_bytes(&bytes).unwrap(), header);
    assert!(safetensors_header_from_bytes(&bytes[..4]).is_err());
}

#[test]
fn reads_safetensors_file_header_without_payload_scan() {
    let dir = std::env::temp_dir().join(format!("nerva-model-header-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let header = "{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd]);
    std::fs::write(&path, bytes).unwrap();

    let file_header = read_safetensors_header_file(&path).unwrap();

    assert_eq!(file_header.header_json, header);
    assert_eq!(file_header.header_bytes, header.len());
    assert_eq!(file_header.data_start, 8 + header.len());
    assert_eq!(file_header.payload_bytes, 4);
    assert!(file_header.require_payload_bytes(4).is_ok());
    assert!(file_header.require_payload_bytes(5).is_err());
    assert!(
        file_header
            .require_file_offset_end(8 + header.len() + 4)
            .is_ok()
    );
    assert!(
        file_header
            .require_file_offset_end(8 + header.len() + 5)
            .is_err()
    );
    assert!(file_header.to_json().contains("\"payload_bytes\":4"));

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn safetensors_file_header_rejects_oversized_header_limit() {
    let dir = std::env::temp_dir().join(format!(
        "nerva-model-header-limit-test-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let header = "{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    std::fs::write(&path, bytes).unwrap();

    assert!(read_safetensors_header_file_with_limit(&path, header.len() - 1).is_err());

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}
