use crate::hf::parser::parse_hf_config_metadata;
use crate::weights::layout::plan::plan_hf_weight_layout;
use crate::weights::manifest::build_hf_tensor_manifest;
use crate::weights::safetensors::header::synthetic_safetensors_header_for_manifest;
use crate::weights::safetensors::probe::safetensors_header_probe;
use crate::weights::safetensors::validation::{
    SafetensorsValidationStatus, validate_safetensors_header_for_manifest,
};

#[test]
fn safetensors_validation_rejects_missing_and_mismatched_tensors() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();
    let plan = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    let valid = synthetic_safetensors_header_for_manifest(&manifest).unwrap();

    assert!(validate_safetensors_header_for_manifest("{}", &manifest).is_err());

    let first = &manifest.entries[0];
    let bad_dtype = format!(
        "{{\"{}\":{{\"dtype\":\"F32\",\"shape\":[{},{}],\"data_offsets\":[0,{}]}}}}",
        first.name, first.rows, first.cols, first.bytes
    );
    assert!(validate_safetensors_header_for_manifest(&bad_dtype, &manifest).is_err());

    let bad_shape = valid.replacen(
        &format!("\"shape\":[{},{}]", first.rows, first.cols),
        "\"shape\":[1,1]",
        1,
    );
    assert!(validate_safetensors_header_for_manifest(&bad_shape, &manifest).is_err());
}

#[test]
fn safetensors_header_probe_reports_manifest_parity() {
    let summary = safetensors_header_probe().unwrap();

    assert_eq!(summary.status, SafetensorsValidationStatus::Ok);
    assert_eq!(summary.validation.manifest_entries, 291);
    assert_eq!(summary.validation.validated_tensors, 291);
    assert_eq!(summary.validation.total_data_bytes, 11_866_218_496);
    assert_ne!(summary.validation.header_hash, 0);
    assert!(summary.to_json().contains("\"validated_tensors\":291"));
}
