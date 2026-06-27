use std::path::{Path, PathBuf};

use nerva_core::types::id::token::TokenId;
use nerva_model::causal_lm::types::HfCausalLmModel;
use nerva_model::hf::parser::parse_hf_config_metadata;
use nerva_model::precision::bits::f32_to_f16_bits;
use nerva_model::weights::layout::entry::WeightBlockRole;
use nerva_model::weights::layout::plan::plan_hf_weight_layout;
use nerva_model::weights::manifest::build_hf_tensor_manifest;
use nerva_model::weights::manifest::{HfTensorManifest, HfTensorManifestEntry};
use nerva_model::weights::safetensors::header::synthetic_safetensors_header_for_manifest;
use nerva_runtime::engine::hf_cuda_decode::run::run_hf_causal_lm_cuda_seed_decode;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_loaded_hf_kv_seed_decode(report: &mut AcceptanceReport) {
    let dir = write_checkpoint_dir();
    let summary = HfCausalLmModel::load_from_hf_dir(&dir)
        .map_err(|err| format!("failed to load HF KV fixture: {err:?}"))
        .and_then(|loaded| {
            run_hf_causal_lm_cuda_seed_decode(&loaded.model, TokenId(0), 4)
                .map_err(|err| format!("failed to execute HF KV decode on CUDA: {err:?}"))
        });
    remove_checkpoint_dir(&dir);

    match summary {
        Ok(summary) => report.push(
            "cuda_loaded_hf_kv_seed_decode",
            summary.passed()
                && summary.tokens == summary.expected_tokens
                && summary.graph_replays == summary.steps_requested as u64
                && summary.graph_replay_events == summary.steps_requested as u64
                && summary.resident_kv_bytes > 0
                && summary.kv_tokens == summary.steps_requested as u64
                && summary.host_causality_edges == 0
                && summary.hot_path_allocations == 0,
            format!(
                "status={:?} steps={} parity={} tokens={} expected={} graph_replays={} graph_replay_events={} resident_kv_bytes={} kv_tokens={} host_causality_edges={} hot_path_allocations={} output_hash={} expected_hash={} error={}",
                summary.status,
                summary.steps_requested,
                summary.parity,
                summary.tokens.len(),
                summary.expected_tokens.len(),
                summary.graph_replays,
                summary.graph_replay_events,
                summary.resident_kv_bytes,
                summary.kv_tokens,
                summary.host_causality_edges,
                summary.hot_path_allocations,
                summary.output_hash,
                summary.expected_hash,
                summary.error.as_deref().unwrap_or("none"),
            ),
        ),
        Err(err) => report.push("cuda_loaded_hf_kv_seed_decode", false, err),
    }
}

pub(crate) fn write_checkpoint_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nerva-hf-kv-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = config_json();
    std::fs::write(dir.join("config.json"), config).unwrap();
    let metadata = parse_hf_config_metadata(config).unwrap();
    let layout = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&layout).unwrap();
    write_safetensors(&dir, &manifest);
    dir
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
            encode_values(&[1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0])
        }
        WeightBlockRole::LmHead => encode_values(&[0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0]),
        WeightBlockRole::AttentionNorm
        | WeightBlockRole::QueryNorm
        | WeightBlockRole::KeyNorm
        | WeightBlockRole::MlpNorm
        | WeightBlockRole::FinalNorm => vec![f32_to_f16_bits(1.0); elements],
        WeightBlockRole::QueryProjection => encoded_identity(entry.rows, entry.cols, -1.0),
        WeightBlockRole::KeyProjection
        | WeightBlockRole::ValueProjection
        | WeightBlockRole::OutputProjection => encoded_identity(entry.rows, entry.cols, 1.0),
        WeightBlockRole::GateProjection
        | WeightBlockRole::UpProjection
        | WeightBlockRole::DownProjection => vec![0; elements],
        WeightBlockRole::QueryBias
        | WeightBlockRole::KeyBias
        | WeightBlockRole::ValueBias
        | WeightBlockRole::OutputBias => vec![0; elements],
    }
}

fn encoded_identity(rows: usize, cols: usize, diagonal: f32) -> Vec<u16> {
    let mut values = vec![0u16; rows * cols];
    let encoded = f32_to_f16_bits(diagonal);
    for index in 0..rows.min(cols) {
        values[index * cols + index] = encoded;
    }
    values
}

fn encode_values(values: &[f32]) -> Vec<u16> {
    values.iter().copied().map(f32_to_f16_bits).collect()
}

pub(crate) fn remove_checkpoint_dir(dir: &Path) {
    let _ = std::fs::remove_file(dir.join("model.safetensors"));
    let _ = std::fs::remove_file(dir.join("config.json"));
    let _ = std::fs::remove_dir(dir);
}

fn config_json() -> &'static str {
    r#"{
        "model_type": "llama",
        "hidden_size": 2,
        "intermediate_size": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 4,
        "rms_norm_eps": 0.00001,
        "torch_dtype": "float16"
    }"#
}
