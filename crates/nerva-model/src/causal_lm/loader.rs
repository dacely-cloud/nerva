use std::path::Path;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

use crate::causal_lm::files::{load_or_synthesize_index, read_required_headers};
use crate::causal_lm::types::{HfCausalLmLoadSummary, HfCausalLmLoaded, HfCausalLmModel};
use crate::common::hash::hash_bytes;
use crate::common::shape::TransformerBlockShape;
use crate::hf::parser::parse_hf_config_metadata;
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::weights::layout::entry::WeightBlockRole;
use crate::weights::layout::plan::plan_hf_weight_layout;
use crate::weights::manifest::build_hf_tensor_manifest;
use crate::weights::safetensors::planner::plan_safetensors_shards_for_manifest;
use crate::weights::safetensors::shard::{SafetensorsShardHeader, SafetensorsShardPlan};
use crate::weights::tensor::{LoadedSafetensorsTensorU16, read_safetensors_tensor_u16};

impl HfCausalLmModel {
    pub fn load_from_hf_dir(path: impl AsRef<Path>) -> Result<HfCausalLmLoaded> {
        load_from_hf_dir(path.as_ref())
    }
}

fn load_from_hf_dir(dir: &Path) -> Result<HfCausalLmLoaded> {
    let config = std::fs::read_to_string(dir.join("config.json")).map_err(|err| {
        NervaError::InvalidArgument {
            reason: format!("failed to read HF config from {}: {err}", dir.display()),
        }
    })?;
    let metadata = parse_hf_config_metadata(&config)?;
    let dtype = metadata
        .torch_dtype
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "HF causal LM requires torch_dtype".to_string(),
        })?;
    let layout = plan_hf_weight_layout(&metadata)?;
    let manifest = build_hf_tensor_manifest(&layout)?;
    let index_json = load_or_synthesize_index(dir, &manifest)?;
    let shard_headers_owned = read_required_headers(dir, &index_json, &manifest)?;
    let shard_headers = shard_headers_owned
        .iter()
        .map(|(name, header)| SafetensorsShardHeader::new(name.as_str(), header.as_str()))
        .collect::<Vec<_>>();
    let shard_plan = plan_safetensors_shards_for_manifest(&index_json, &shard_headers, &manifest)?;

    let mut layers = Vec::with_capacity(metadata.num_hidden_layers);
    let mut bytes_loaded = 0usize;
    let mut data_hash = 0xcbf2_9ce4_8422_2325u64;
    let shape = metadata.block_shape();
    let rms_eps = metadata.rms_norm_eps.unwrap_or(1e-5);
    let rope_theta = metadata.rope_theta;
    let attention_bias = metadata.attention_bias;
    for layer in 0..metadata.num_hidden_layers {
        let block = load_layer(
            dir,
            &shard_plan,
            dtype,
            shape,
            rms_eps,
            rope_theta,
            attention_bias,
            layer as u32,
            &mut bytes_loaded,
            &mut data_hash,
        )?;
        layers.push(block);
    }
    let embeddings = load_tensor(dir, &shard_plan, WeightBlockRole::TokenEmbedding, None)?;
    bytes_loaded += embeddings.bytes_read;
    data_hash = fold_hash(data_hash, embeddings.data_hash);
    let final_norm = load_tensor(dir, &shard_plan, WeightBlockRole::FinalNorm, None)?;
    bytes_loaded += final_norm.bytes_read;
    data_hash = fold_hash(data_hash, final_norm.data_hash);
    let lm_head = if metadata.tie_word_embeddings {
        embeddings.values.clone()
    } else {
        let tensor = load_tensor(dir, &shard_plan, WeightBlockRole::LmHead, None)?;
        bytes_loaded += tensor.bytes_read;
        data_hash = fold_hash(data_hash, tensor.data_hash);
        tensor.values
    };

    Ok(HfCausalLmLoaded {
        model: HfCausalLmModel {
            metadata,
            dtype,
            layers,
            embeddings: embeddings.values,
            final_norm: final_norm.values,
            lm_head,
            rms_eps,
        },
        summary: HfCausalLmLoadSummary {
            manifest,
            shard_plan,
            tensors_loaded: layout.blocks.len(),
            bytes_loaded,
            data_hash,
            tied_lm_head: layout.metadata.tie_word_embeddings,
        },
    })
}

fn load_layer(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    dtype: DType,
    shape: TransformerBlockShape,
    rms_eps: f32,
    rope_theta: Option<f32>,
    attention_bias: bool,
    layer: u32,
    bytes_loaded: &mut usize,
    data_hash: &mut u64,
) -> Result<PrecisionTransformerBlock> {
    let mut load = |role| -> Result<Vec<u16>> {
        let tensor = load_tensor(dir, plan, role, Some(layer))?;
        *bytes_loaded += tensor.bytes_read;
        *data_hash = fold_hash(*data_hash, tensor.data_hash);
        Ok(tensor.values)
    };
    let block = PrecisionTransformerBlock::new_from_encoded(
        dtype,
        shape,
        load(WeightBlockRole::AttentionNorm)?,
        load(WeightBlockRole::MlpNorm)?,
        load(WeightBlockRole::QueryProjection)?,
        load(WeightBlockRole::KeyProjection)?,
        load(WeightBlockRole::ValueProjection)?,
        load(WeightBlockRole::OutputProjection)?,
        load(WeightBlockRole::GateProjection)?,
        load(WeightBlockRole::UpProjection)?,
        load(WeightBlockRole::DownProjection)?,
        rms_eps,
    )?
    .with_rope_theta(rope_theta)?;
    if attention_bias {
        block.with_attention_biases(
            load(WeightBlockRole::QueryBias)?,
            load(WeightBlockRole::KeyBias)?,
            load(WeightBlockRole::ValueBias)?,
            load(WeightBlockRole::OutputBias)?,
        )
    } else {
        Ok(block)
    }
}

fn load_tensor(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    role: WeightBlockRole,
    layer: Option<u32>,
) -> Result<LoadedSafetensorsTensorU16> {
    let entry = plan
        .entries
        .iter()
        .find(|entry| entry.role == role && entry.layer == layer)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("HF causal LM missing tensor role {:?}", role),
        })?;
    read_safetensors_tensor_u16(dir.join(&entry.shard_file), entry)
}

fn fold_hash(hash: u64, value: u64) -> u64 {
    let mut bytes = hash.to_le_bytes().to_vec();
    bytes.extend_from_slice(&value.to_le_bytes());
    hash_bytes(&bytes)
}
