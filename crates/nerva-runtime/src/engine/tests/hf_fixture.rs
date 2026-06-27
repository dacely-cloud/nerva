use std::path::{Path, PathBuf};

use nerva_model::hf::parser::parse_hf_config_metadata;
use nerva_model::precision::bits::f32_to_f16_bits;
use nerva_model::weights::layout::entry::WeightBlockRole;
use nerva_model::weights::layout::plan::plan_hf_weight_layout;
use nerva_model::weights::manifest::build_hf_tensor_manifest;
use nerva_model::weights::manifest::{HfTensorManifest, HfTensorManifestEntry};
use nerva_model::weights::safetensors::header::synthetic_safetensors_header_for_manifest;

pub(crate) fn write_cycle_hf_checkpoint_dir(prefix: &str, layers: usize) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = fixture_config(layers);
    std::fs::write(dir.join("config.json"), &config).unwrap();
    let metadata = parse_hf_config_metadata(&config).unwrap();
    let layout = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&layout).unwrap();
    write_safetensors(&dir, &manifest);
    dir
}

pub(crate) fn remove_hf_checkpoint_dir(dir: &Path) {
    let _ = std::fs::remove_file(dir.join("model.safetensors"));
    let _ = std::fs::remove_file(dir.join("config.json"));
    let _ = std::fs::remove_dir(dir);
}

fn fixture_config(layers: usize) -> String {
    format!(
        r#"{{
            "model_type": "llama",
            "hidden_size": 2,
            "intermediate_size": 2,
            "num_hidden_layers": {layers},
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 4,
            "rms_norm_eps": 0.00001,
            "torch_dtype": "float16"
        }}"#,
    )
}

fn write_safetensors(dir: &Path, manifest: &HfTensorManifest) {
    let header = synthetic_safetensors_header_for_manifest(manifest).unwrap();
    let payload = payload_for_manifest(manifest);
    let mut bytes = Vec::with_capacity(8 + header.len() + payload.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&payload);
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

fn payload_for_manifest(manifest: &HfTensorManifest) -> Vec<u8> {
    let mut payload = Vec::new();
    for entry in &manifest.entries {
        for value in values_for_entry(entry) {
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    payload
}

fn values_for_entry(entry: &HfTensorManifestEntry) -> Vec<u16> {
    let elements = entry.bytes / 2;
    match entry.role {
        WeightBlockRole::TokenEmbedding => {
            encode_values(&[1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0])
        }
        WeightBlockRole::LmHead => encode_values(&[0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0]),
        WeightBlockRole::AttentionNorm | WeightBlockRole::MlpNorm | WeightBlockRole::FinalNorm => {
            vec![f32_to_f16_bits(1.0); elements]
        }
        WeightBlockRole::QueryBias
        | WeightBlockRole::KeyBias
        | WeightBlockRole::ValueBias
        | WeightBlockRole::OutputBias
        | WeightBlockRole::QueryProjection
        | WeightBlockRole::KeyProjection
        | WeightBlockRole::ValueProjection
        | WeightBlockRole::OutputProjection
        | WeightBlockRole::GateProjection
        | WeightBlockRole::UpProjection
        | WeightBlockRole::DownProjection => vec![0; elements],
    }
}

fn encode_values(values: &[f32]) -> Vec<u16> {
    values.iter().copied().map(f32_to_f16_bits).collect()
}
