use std::path::Path;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_ledger::types::event::LedgerEventKind;

use crate::causal_lm::summary::{HfCausalLmSmokeStatus, HfCausalLmSmokeSummary};
use crate::causal_lm::types::{HfCausalLmDecodeScratch, HfCausalLmLoaded, HfCausalLmModel};
use crate::common::hash::hash_tokens;
use crate::common::token::expected_cycle;
use crate::hf::parser::parse_hf_config_metadata;
use crate::precision::bits::f32_to_f16_bits;
use crate::weights::layout::entry::WeightBlockRole;
use crate::weights::layout::plan::plan_hf_weight_layout;
use crate::weights::manifest::{HfTensorManifest, HfTensorManifestEntry, build_hf_tensor_manifest};
use crate::weights::safetensors::header::synthetic_safetensors_header_for_manifest;

pub fn hf_causal_lm_safetensors_smoke(steps: usize) -> Result<HfCausalLmSmokeSummary> {
    let dir = temp_model_dir();
    std::fs::create_dir_all(&dir).map_err(|err| NervaError::InvalidArgument {
        reason: format!("failed to create {}: {err}", dir.display()),
    })?;
    let result = write_fixture(&dir).and_then(|manifest| run_fixture(&dir, manifest, steps));
    let _ = std::fs::remove_file(dir.join("model.safetensors"));
    let _ = std::fs::remove_file(dir.join("config.json"));
    let _ = std::fs::remove_dir(&dir);
    result
}

pub fn load_hf_causal_lm_smoke_fixture() -> Result<HfCausalLmLoaded> {
    let dir = temp_model_dir();
    std::fs::create_dir_all(&dir).map_err(|err| NervaError::InvalidArgument {
        reason: format!("failed to create {}: {err}", dir.display()),
    })?;
    let result = write_fixture(&dir).and_then(|_| HfCausalLmModel::load_from_hf_dir(&dir));
    let _ = std::fs::remove_file(dir.join("model.safetensors"));
    let _ = std::fs::remove_file(dir.join("config.json"));
    let _ = std::fs::remove_dir(&dir);
    result
}

fn run_fixture(
    dir: &Path,
    manifest: HfTensorManifest,
    steps: usize,
) -> Result<HfCausalLmSmokeSummary> {
    let loaded = HfCausalLmModel::load_from_hf_dir(dir)?;
    let model = loaded.model;
    let mut scratch = HfCausalLmDecodeScratch::new(model.shape(), model.metadata().vocab_size)?;
    let seed = TokenId(0);
    let (tokens, ledgers) = model.decode_greedy(seed, steps, &mut scratch)?;
    let expected_tokens = expected_cycle(seed, steps, model.metadata().vocab_size);
    let hot_path_allocations = ledgers
        .iter()
        .map(|ledger| ledger.hot_path_allocations)
        .sum();
    let cpu_events = ledgers
        .iter()
        .map(|ledger| ledger.event_count(LedgerEventKind::CpuActivity))
        .sum();
    let execution_decisions = ledgers
        .iter()
        .map(|ledger| ledger.execution_decisions.len() as u64)
        .sum();

    Ok(HfCausalLmSmokeSummary {
        status: HfCausalLmSmokeStatus::Ok,
        dtype: model.dtype(),
        layers: model.layer_count(),
        hidden: model.metadata().hidden_size,
        vocab_size: model.metadata().vocab_size,
        manifest_entries: manifest.entries.len(),
        shard_plan_entries: loaded.summary.shard_plan.entries.len(),
        tensors_loaded: loaded.summary.tensors_loaded,
        bytes_loaded: loaded.summary.bytes_loaded,
        final_norm_loaded: model.final_norm.len() == model.metadata().hidden_size,
        tied_lm_head: loaded.summary.tied_lm_head,
        steps,
        parity: tokens == expected_tokens,
        output_hash: hash_tokens(&tokens),
        data_hash: loaded.summary.data_hash,
        ledger_count: ledgers.len() as u64,
        cpu_events,
        execution_decisions,
        hot_path_allocations,
        tokens,
        expected_tokens,
    })
}

fn write_fixture(dir: &Path) -> Result<HfTensorManifest> {
    let config = fixture_config();
    std::fs::write(dir.join("config.json"), config).map_err(|err| NervaError::InvalidArgument {
        reason: format!("failed to write fixture config: {err}"),
    })?;
    let metadata = parse_hf_config_metadata(config)?;
    let layout = plan_hf_weight_layout(&metadata)?;
    let manifest = build_hf_tensor_manifest(&layout)?;
    let header = synthetic_safetensors_header_for_manifest(&manifest)?;
    let payload = payload_for_manifest(&manifest)?;
    let mut bytes = Vec::with_capacity(8 + header.len() + payload.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&payload);
    std::fs::write(dir.join("model.safetensors"), bytes).map_err(|err| {
        NervaError::InvalidArgument {
            reason: format!("failed to write fixture safetensors: {err}"),
        }
    })?;
    Ok(manifest)
}

fn payload_for_manifest(manifest: &HfTensorManifest) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    for entry in &manifest.entries {
        payload.extend_from_slice(&bytes_for_entry(entry)?);
    }
    Ok(payload)
}

fn bytes_for_entry(entry: &HfTensorManifestEntry) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(entry.bytes);
    match entry.dtype {
        DType::F32 => {
            let value = match entry.role {
                WeightBlockRole::AttentionNorm
                | WeightBlockRole::QueryNorm
                | WeightBlockRole::KeyNorm
                | WeightBlockRole::LinearNorm
                | WeightBlockRole::MlpNorm
                | WeightBlockRole::FinalNorm => 1.0f32,
                _ => 0.0f32,
            };
            for _ in 0..entry.elements {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        DType::F16 | DType::BF16 => {
            for value in values_for_entry(entry)? {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        _ => {
            return Err(NervaError::InvalidArgument {
                reason: format!("fixture tensor {} has unsupported dtype", entry.name),
            });
        }
    }
    if bytes.len() == entry.bytes {
        Ok(bytes)
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!("fixture tensor {} has wrong byte count", entry.name),
        })
    }
}

fn values_for_entry(entry: &HfTensorManifestEntry) -> Result<Vec<u16>> {
    let elements = entry.bytes / 2;
    let values = match entry.role {
        WeightBlockRole::TokenEmbedding => encoded_cycle_embeddings(entry.rows, entry.cols)?,
        WeightBlockRole::LmHead => encoded_cycle_lm_head(entry.rows, entry.cols)?,
        WeightBlockRole::AttentionNorm
        | WeightBlockRole::QueryNorm
        | WeightBlockRole::KeyNorm
        | WeightBlockRole::DeepSeekQALoraNorm
        | WeightBlockRole::DeepSeekKvANorm
        | WeightBlockRole::DeepSeekIndexerKeyNorm
        | WeightBlockRole::LinearNorm
        | WeightBlockRole::MlpNorm
        | WeightBlockRole::FinalNorm => vec![f32_to_f16_bits(1.0); elements],
        WeightBlockRole::QueryBias
        | WeightBlockRole::KeyBias
        | WeightBlockRole::ValueBias
        | WeightBlockRole::OutputBias
        | WeightBlockRole::DeepSeekIndexerKeyNormBias => vec![0; elements],
        WeightBlockRole::QueryProjection
        | WeightBlockRole::KeyProjection
        | WeightBlockRole::ValueProjection
        | WeightBlockRole::OutputProjection
        | WeightBlockRole::DeepSeekQALoraProjection
        | WeightBlockRole::DeepSeekQALoraScaleInv
        | WeightBlockRole::DeepSeekQBProjection
        | WeightBlockRole::DeepSeekQBScaleInv
        | WeightBlockRole::DeepSeekKvAProjection
        | WeightBlockRole::DeepSeekKvAScaleInv
        | WeightBlockRole::DeepSeekKvBProjection
        | WeightBlockRole::DeepSeekKvBScaleInv
        | WeightBlockRole::DeepSeekOutputScaleInv
        | WeightBlockRole::DeepSeekIndexerQueryProjection
        | WeightBlockRole::DeepSeekIndexerQueryScaleInv
        | WeightBlockRole::DeepSeekIndexerKeyProjection
        | WeightBlockRole::DeepSeekIndexerKeyScaleInv
        | WeightBlockRole::DeepSeekIndexerWeightsProjection
        | WeightBlockRole::LinearConvProjection
        | WeightBlockRole::LinearQkvProjection
        | WeightBlockRole::LinearZProjection
        | WeightBlockRole::LinearBProjection
        | WeightBlockRole::LinearAProjection
        | WeightBlockRole::LinearDtBias
        | WeightBlockRole::LinearALog
        | WeightBlockRole::LinearOutputProjection
        | WeightBlockRole::GateProjection
        | WeightBlockRole::UpProjection
        | WeightBlockRole::DownProjection
        | WeightBlockRole::GateScaleInv
        | WeightBlockRole::UpScaleInv
        | WeightBlockRole::DownScaleInv
        | WeightBlockRole::RouterProjection
        | WeightBlockRole::RouterCorrectionBias
        | WeightBlockRole::ExpertGateProjection
        | WeightBlockRole::ExpertUpProjection
        | WeightBlockRole::ExpertGateUpProjection
        | WeightBlockRole::ExpertDownProjection
        | WeightBlockRole::ExpertGateScaleInv
        | WeightBlockRole::ExpertUpScaleInv
        | WeightBlockRole::ExpertDownScaleInv
        | WeightBlockRole::SharedExpertGateProjection
        | WeightBlockRole::SharedExpertUpProjection
        | WeightBlockRole::SharedExpertDownProjection
        | WeightBlockRole::SharedExpertGateScaleInv
        | WeightBlockRole::SharedExpertUpScaleInv
        | WeightBlockRole::SharedExpertDownScaleInv
        | WeightBlockRole::SharedExpertRouterProjection => vec![0; elements],
    };
    if values.len() == elements {
        Ok(values)
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!("fixture tensor {} has wrong element count", entry.name),
        })
    }
}

fn encoded_cycle_embeddings(rows: usize, cols: usize) -> Result<Vec<u16>> {
    require_cycle_shape(rows, cols, "embedding")?;
    Ok(encode_values(&[1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0]))
}

fn encoded_cycle_lm_head(rows: usize, cols: usize) -> Result<Vec<u16>> {
    require_cycle_shape(rows, cols, "lm_head")?;
    Ok(encode_values(&[0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0]))
}

fn require_cycle_shape(rows: usize, cols: usize, name: &str) -> Result<()> {
    if rows == 4 && cols == 2 {
        Ok(())
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!("HF causal LM fixture {name} expects a 4x2 tensor"),
        })
    }
}

fn encode_values(values: &[f32]) -> Vec<u16> {
    values.iter().copied().map(f32_to_f16_bits).collect()
}

fn fixture_config() -> &'static str {
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

fn temp_model_dir() -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("nerva-hf-causal-lm-{}-{nanos}", std::process::id()))
}
