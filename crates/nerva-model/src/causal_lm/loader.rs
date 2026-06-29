use std::path::Path;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

use crate::causal_lm::files::{load_or_synthesize_index, read_required_headers};
use crate::causal_lm::load_options::HfCausalLmLoadOptions;
use crate::causal_lm::types::{
    HfCausalLmLayer, HfCausalLmLoadSummary, HfCausalLmLoaded, HfCausalLmModel,
};
use crate::common::shape::TransformerBlockShape;
use crate::hf::architecture::HfArchitectureKind;
use crate::hf::contract::validate_weight_layout_contract;
use crate::hf::metadata::{HfAttentionLayerKind, HfMlpLayerKind, HfModelMetadata};
use crate::hf::parser::parse_hf_config_metadata;
use crate::precision::block::gdn::{PrecisionGatedDeltaNetConfig, PrecisionGatedDeltaNetMoeBlock};
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::precision::block::moe::{PrecisionMoeConfig, PrecisionMoeTransformerBlock};
use crate::weights::layout::entry::WeightBlockRole;
use crate::weights::layout::plan::plan_hf_weight_layout;
use crate::weights::manifest::build_hf_tensor_manifest;
use crate::weights::safetensors::planner::plan_safetensors_shards_for_manifest;
use crate::weights::safetensors::shard::{SafetensorsShardHeader, SafetensorsShardPlan};
use crate::weights::tensor::{
    LoadedSafetensorsTensorF32, LoadedSafetensorsTensorU16, read_safetensors_tensor_f32_with_hash,
    read_safetensors_tensor_u16_with_hash,
};

mod accounting;

use self::accounting::LoadAccounting;

impl HfCausalLmModel {
    pub fn load_from_hf_dir(path: impl AsRef<Path>) -> Result<HfCausalLmLoaded> {
        load_from_hf_dir(path.as_ref(), HfCausalLmLoadOptions::default())
    }

    pub fn load_from_hf_dir_with_options(
        path: impl AsRef<Path>,
        options: HfCausalLmLoadOptions,
    ) -> Result<HfCausalLmLoaded> {
        load_from_hf_dir(path.as_ref(), options)
    }
}

fn load_from_hf_dir(dir: &Path, options: HfCausalLmLoadOptions) -> Result<HfCausalLmLoaded> {
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
    validate_weight_layout_contract(&metadata)?;
    let index_json = load_or_synthesize_index(dir, &manifest)?;
    let shard_headers_owned = read_required_headers(dir, &index_json, &manifest)?;
    let shard_headers = shard_headers_owned
        .iter()
        .map(|(name, header)| SafetensorsShardHeader::new(name.as_str(), header.as_str()))
        .collect::<Vec<_>>();
    let shard_plan = plan_safetensors_shards_for_manifest(&index_json, &shard_headers, &manifest)?;

    let mut layers = Vec::with_capacity(metadata.num_hidden_layers);
    let mut accounting = LoadAccounting::new();
    let shape = metadata.block_shape();
    let rms_eps = metadata.rms_norm_eps.unwrap_or(1e-5);
    let rope_theta = metadata.rope_theta;
    for layer in 0..metadata.num_hidden_layers {
        let block = load_layer(
            dir,
            &shard_plan,
            &metadata,
            dtype,
            shape,
            rms_eps,
            rope_theta,
            layer as u32,
            options,
            &mut accounting,
        )?;
        layers.push(block);
    }
    let embeddings = load_tensor(
        dir,
        &shard_plan,
        WeightBlockRole::TokenEmbedding,
        None,
        options,
    )?;
    accounting.record(embeddings.bytes_read, embeddings.data_hash);
    let final_norm = load_tensor(dir, &shard_plan, WeightBlockRole::FinalNorm, None, options)?;
    accounting.record(final_norm.bytes_read, final_norm.data_hash);
    let lm_head = if metadata.tie_word_embeddings {
        embeddings.values.clone()
    } else {
        let tensor = load_tensor(dir, &shard_plan, WeightBlockRole::LmHead, None, options)?;
        accounting.record(tensor.bytes_read, tensor.data_hash);
        tensor.values
    };

    let tensors_loaded = manifest.entries.len();
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
            tensors_loaded,
            bytes_loaded: accounting.bytes_loaded,
            data_hash: accounting.data_hash(options.compute_data_hash),
            data_hash_available: options.compute_data_hash,
            tied_lm_head: layout.metadata.tie_word_embeddings,
        },
    })
}

fn load_layer(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    metadata: &HfModelMetadata,
    dtype: DType,
    shape: TransformerBlockShape,
    rms_eps: f32,
    rope_theta: Option<f32>,
    layer: u32,
    options: HfCausalLmLoadOptions,
    accounting: &mut LoadAccounting,
) -> Result<HfCausalLmLayer> {
    let attention_kind = metadata
        .attention_layer_types
        .get(layer as usize)
        .copied()
        .unwrap_or(HfAttentionLayerKind::Full);
    let mlp_kind = metadata
        .mlp_layer_types
        .get(layer as usize)
        .copied()
        .unwrap_or(HfMlpLayerKind::Dense);
    match (attention_kind, mlp_kind) {
        (HfAttentionLayerKind::Linear, HfMlpLayerKind::SparseMoe) => {
            load_gated_delta_net_moe_layer(
                dir, plan, metadata, dtype, shape, rms_eps, layer, options, accounting,
            )
        }
        (HfAttentionLayerKind::Linear, HfMlpLayerKind::Dense) => Err(NervaError::InvalidArgument {
            reason: "HF Qwen3.5 linear_attention dense-MLP layers are not represented yet"
                .to_string(),
        }),
        (HfAttentionLayerKind::Full, HfMlpLayerKind::SparseMoe) => load_sparse_moe_layer(
            dir,
            plan,
            metadata,
            dtype,
            shape,
            rms_eps,
            rope_theta,
            metadata.attention_qkv_bias,
            metadata.attention_output_bias,
            layer,
            options,
            accounting,
        ),
        (HfAttentionLayerKind::Full, HfMlpLayerKind::Dense) => load_dense_layer(
            dir,
            plan,
            metadata,
            dtype,
            shape,
            rms_eps,
            rope_theta,
            metadata.attention_qkv_bias,
            metadata.attention_output_bias,
            layer,
            options,
            accounting,
        )
        .map(HfCausalLmLayer::Dense),
    }
}

fn load_dense_layer(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    metadata: &HfModelMetadata,
    dtype: DType,
    shape: TransformerBlockShape,
    rms_eps: f32,
    rope_theta: Option<f32>,
    attention_qkv_bias: bool,
    attention_output_bias: bool,
    layer: u32,
    options: HfCausalLmLoadOptions,
    accounting: &mut LoadAccounting,
) -> Result<PrecisionTransformerBlock> {
    let (w_q, w_q_gate) = load_query_projection(dir, plan, metadata, layer, options, accounting)?;
    let mut block = PrecisionTransformerBlock::new_from_encoded(
        dtype,
        shape,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::AttentionNorm,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::MlpNorm,
            layer,
            options,
            accounting,
        )?,
        w_q,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::KeyProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::ValueProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::OutputProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::GateProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::UpProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::DownProjection,
            layer,
            options,
            accounting,
        )?,
        rms_eps,
    )?
    .with_rope_theta(rope_theta)?;
    if let Some(w_q_gate) = w_q_gate {
        block = block.with_query_gate_projection(w_q_gate)?;
    }
    if plan
        .entries
        .iter()
        .any(|entry| entry.role == WeightBlockRole::QueryNorm && entry.layer == Some(layer))
    {
        block = block.with_qk_norm(
            load_layer_tensor(
                dir,
                plan,
                WeightBlockRole::QueryNorm,
                layer,
                options,
                accounting,
            )?,
            load_layer_tensor(
                dir,
                plan,
                WeightBlockRole::KeyNorm,
                layer,
                options,
                accounting,
            )?,
        )?;
    }
    if attention_qkv_bias || attention_output_bias {
        block.with_optional_attention_biases(
            load_enabled_bias_tensor(
                dir,
                plan,
                WeightBlockRole::QueryBias,
                layer,
                attention_qkv_bias,
                options,
                accounting,
            )?,
            load_enabled_bias_tensor(
                dir,
                plan,
                WeightBlockRole::KeyBias,
                layer,
                attention_qkv_bias,
                options,
                accounting,
            )?,
            load_enabled_bias_tensor(
                dir,
                plan,
                WeightBlockRole::ValueBias,
                layer,
                attention_qkv_bias,
                options,
                accounting,
            )?,
            load_enabled_bias_tensor(
                dir,
                plan,
                WeightBlockRole::OutputBias,
                layer,
                attention_output_bias,
                options,
                accounting,
            )?,
        )
    } else {
        Ok(block)
    }
}

fn load_gated_delta_net_moe_layer(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    metadata: &HfModelMetadata,
    dtype: DType,
    shape: TransformerBlockShape,
    rms_eps: f32,
    layer: u32,
    options: HfCausalLmLoadOptions,
    accounting: &mut LoadAccounting,
) -> Result<HfCausalLmLayer> {
    if metadata.architecture != HfArchitectureKind::Qwen35Moe {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "GatedDeltaNet-MoE layer loading is only implemented for Qwen3.5-MoE, got {}",
                metadata.architecture.as_str()
            ),
        });
    }
    let gdn_config = PrecisionGatedDeltaNetConfig {
        key_heads: required_metadata_usize(metadata.linear_num_key_heads, "linear_num_key_heads")?,
        value_heads: required_metadata_usize(
            metadata.linear_num_value_heads,
            "linear_num_value_heads",
        )?,
        key_head_dim: required_metadata_usize(metadata.linear_key_head_dim, "linear_key_head_dim")?,
        value_head_dim: required_metadata_usize(
            metadata.linear_value_head_dim,
            "linear_value_head_dim",
        )?,
        conv_kernel: required_metadata_usize(
            metadata.linear_conv_kernel_dim,
            "linear_conv_kernel_dim",
        )?,
    };
    let moe_config = moe_config_from_metadata(metadata)?;
    let block = PrecisionGatedDeltaNetMoeBlock::new_from_encoded(
        dtype,
        shape,
        gdn_config,
        moe_config,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::AttentionNorm,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::LinearConvProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::LinearQkvProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::LinearZProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::LinearBProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::LinearAProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::LinearDtBias,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor_f32(
            dir,
            plan,
            WeightBlockRole::LinearALog,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor_f32(
            dir,
            plan,
            WeightBlockRole::LinearNorm,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::LinearOutputProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::MlpNorm,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::RouterProjection,
            layer,
            options,
            accounting,
        )?,
        load_sparse_moe_expert_gate_up(
            dir,
            plan,
            metadata.architecture,
            layer,
            moe_config.num_experts,
            moe_config.moe_intermediate,
            metadata.hidden_size,
            options,
            accounting,
        )?,
        load_sparse_moe_expert_down(
            dir,
            plan,
            metadata.architecture,
            layer,
            moe_config.num_experts,
            moe_config.moe_intermediate,
            metadata.hidden_size,
            options,
            accounting,
        )?,
        load_optional_layer_tensor(
            dir,
            plan,
            WeightBlockRole::SharedExpertGateProjection,
            layer,
            options,
            accounting,
        )?,
        load_optional_layer_tensor(
            dir,
            plan,
            WeightBlockRole::SharedExpertUpProjection,
            layer,
            options,
            accounting,
        )?,
        load_optional_layer_tensor(
            dir,
            plan,
            WeightBlockRole::SharedExpertDownProjection,
            layer,
            options,
            accounting,
        )?,
        load_optional_layer_tensor(
            dir,
            plan,
            WeightBlockRole::SharedExpertRouterProjection,
            layer,
            options,
            accounting,
        )?,
        rms_eps,
    )?;
    Ok(HfCausalLmLayer::GatedDeltaNetMoe(block))
}

fn load_sparse_moe_layer(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    metadata: &HfModelMetadata,
    dtype: DType,
    shape: TransformerBlockShape,
    rms_eps: f32,
    rope_theta: Option<f32>,
    attention_qkv_bias: bool,
    attention_output_bias: bool,
    layer: u32,
    options: HfCausalLmLoadOptions,
    accounting: &mut LoadAccounting,
) -> Result<HfCausalLmLayer> {
    let moe_config = moe_config_from_metadata(metadata)?;
    let (w_q, w_q_gate) = load_query_projection(dir, plan, metadata, layer, options, accounting)?;
    let mut block = PrecisionMoeTransformerBlock::new_from_encoded(
        dtype,
        shape,
        moe_config,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::AttentionNorm,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::MlpNorm,
            layer,
            options,
            accounting,
        )?,
        w_q,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::KeyProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::ValueProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::OutputProjection,
            layer,
            options,
            accounting,
        )?,
        load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::RouterProjection,
            layer,
            options,
            accounting,
        )?,
        load_sparse_moe_expert_gate_up(
            dir,
            plan,
            metadata.architecture,
            layer,
            moe_config.num_experts,
            moe_config.moe_intermediate,
            metadata.hidden_size,
            options,
            accounting,
        )?,
        load_sparse_moe_expert_down(
            dir,
            plan,
            metadata.architecture,
            layer,
            moe_config.num_experts,
            moe_config.moe_intermediate,
            metadata.hidden_size,
            options,
            accounting,
        )?,
        load_optional_layer_tensor(
            dir,
            plan,
            WeightBlockRole::SharedExpertGateProjection,
            layer,
            options,
            accounting,
        )?,
        load_optional_layer_tensor(
            dir,
            plan,
            WeightBlockRole::SharedExpertUpProjection,
            layer,
            options,
            accounting,
        )?,
        load_optional_layer_tensor(
            dir,
            plan,
            WeightBlockRole::SharedExpertDownProjection,
            layer,
            options,
            accounting,
        )?,
        load_optional_layer_tensor(
            dir,
            plan,
            WeightBlockRole::SharedExpertRouterProjection,
            layer,
            options,
            accounting,
        )?,
        rms_eps,
    )?
    .with_rope_theta(rope_theta)?;
    if let Some(w_q_gate) = w_q_gate {
        block = block.with_query_gate_projection(w_q_gate)?;
    }
    if plan
        .entries
        .iter()
        .any(|entry| entry.role == WeightBlockRole::QueryNorm && entry.layer == Some(layer))
    {
        block = block.with_qk_norm(
            load_layer_tensor(
                dir,
                plan,
                WeightBlockRole::QueryNorm,
                layer,
                options,
                accounting,
            )?,
            load_layer_tensor(
                dir,
                plan,
                WeightBlockRole::KeyNorm,
                layer,
                options,
                accounting,
            )?,
        )?;
    }
    if attention_qkv_bias || attention_output_bias {
        block = block.with_optional_attention_biases(
            load_enabled_bias_tensor(
                dir,
                plan,
                WeightBlockRole::QueryBias,
                layer,
                attention_qkv_bias,
                options,
                accounting,
            )?,
            load_enabled_bias_tensor(
                dir,
                plan,
                WeightBlockRole::KeyBias,
                layer,
                attention_qkv_bias,
                options,
                accounting,
            )?,
            load_enabled_bias_tensor(
                dir,
                plan,
                WeightBlockRole::ValueBias,
                layer,
                attention_qkv_bias,
                options,
                accounting,
            )?,
            load_enabled_bias_tensor(
                dir,
                plan,
                WeightBlockRole::OutputBias,
                layer,
                attention_output_bias,
                options,
                accounting,
            )?,
        )?;
    }
    Ok(HfCausalLmLayer::SparseMoe(block))
}

fn load_query_projection(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    metadata: &HfModelMetadata,
    layer: u32,
    options: HfCausalLmLoadOptions,
    accounting: &mut LoadAccounting,
) -> Result<(Vec<u16>, Option<Vec<u16>>)> {
    let tensor = load_tensor(
        dir,
        plan,
        WeightBlockRole::QueryProjection,
        Some(layer),
        options,
    )?;
    accounting.record(tensor.bytes_read, tensor.data_hash);
    if !matches!(
        metadata.architecture,
        HfArchitectureKind::Qwen35 | HfArchitectureKind::Qwen35Moe
    ) {
        return Ok((tensor.values, None));
    }
    split_qwen35_query_gate_projection(
        tensor.values,
        metadata.num_attention_heads,
        metadata.head_dim,
        metadata.hidden_size,
    )
}

fn split_qwen35_query_gate_projection(
    packed: Vec<u16>,
    heads: usize,
    head_dim: usize,
    hidden: usize,
) -> Result<(Vec<u16>, Option<Vec<u16>>)> {
    let head_projection =
        head_dim
            .checked_mul(hidden)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: head_dim,
                reason: "Qwen3.5 query head projection size overflow".to_string(),
            })?;
    let expected = heads
        .checked_mul(head_projection)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: head_projection,
            reason: "Qwen3.5 packed query/gate projection size overflow".to_string(),
        })?;
    require_vec_len("Qwen3.5 packed q_proj", packed.len(), expected)?;
    let logical_len =
        heads
            .checked_mul(head_projection)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: head_projection,
                reason: "Qwen3.5 logical query projection size overflow".to_string(),
            })?;
    let mut query = Vec::with_capacity(logical_len);
    let mut gate = Vec::with_capacity(logical_len);
    for head in 0..heads {
        let base = head
            .checked_mul(head_projection)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: head_projection,
                reason: "Qwen3.5 query/gate projection offset overflow".to_string(),
            })?;
        let q_start = base;
        let q_end = q_start + head_projection;
        let gate_start = q_end;
        let gate_end = gate_start + head_projection;
        query.extend_from_slice(&packed[q_start..q_end]);
        gate.extend_from_slice(&packed[gate_start..gate_end]);
    }
    Ok((query, Some(gate)))
}

fn load_layer_tensor(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    role: WeightBlockRole,
    layer: u32,
    options: HfCausalLmLoadOptions,
    accounting: &mut LoadAccounting,
) -> Result<Vec<u16>> {
    let tensor = load_tensor(dir, plan, role, Some(layer), options)?;
    accounting.record(tensor.bytes_read, tensor.data_hash);
    Ok(tensor.values)
}

fn load_layer_tensor_f32(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    role: WeightBlockRole,
    layer: u32,
    options: HfCausalLmLoadOptions,
    accounting: &mut LoadAccounting,
) -> Result<Vec<f32>> {
    let tensor = load_tensor_f32(dir, plan, role, Some(layer), options)?;
    accounting.record(tensor.bytes_read, tensor.data_hash);
    Ok(tensor.values)
}

fn load_enabled_bias_tensor(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    role: WeightBlockRole,
    layer: u32,
    enabled: bool,
    options: HfCausalLmLoadOptions,
    accounting: &mut LoadAccounting,
) -> Result<Option<Vec<u16>>> {
    if enabled {
        load_layer_tensor(dir, plan, role, layer, options, accounting).map(Some)
    } else {
        Ok(None)
    }
}

fn load_optional_layer_tensor(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    role: WeightBlockRole,
    layer: u32,
    options: HfCausalLmLoadOptions,
    accounting: &mut LoadAccounting,
) -> Result<Vec<u16>> {
    if plan
        .entries
        .iter()
        .any(|entry| entry.role == role && entry.layer == Some(layer))
    {
        load_layer_tensor(dir, plan, role, layer, options, accounting)
    } else {
        Ok(Vec::new())
    }
}

fn load_tensor(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    role: WeightBlockRole,
    layer: Option<u32>,
    options: HfCausalLmLoadOptions,
) -> Result<LoadedSafetensorsTensorU16> {
    load_tensor_with_expert(dir, plan, role, layer, None, options)
}

fn load_tensor_f32(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    role: WeightBlockRole,
    layer: Option<u32>,
    options: HfCausalLmLoadOptions,
) -> Result<LoadedSafetensorsTensorF32> {
    let entry = plan
        .entries
        .iter()
        .find(|entry| entry.role == role && entry.layer == layer && entry.expert.is_none())
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!(
                "HF causal LM missing f32 tensor role {:?} layer {:?}",
                role, layer
            ),
        })?;
    read_safetensors_tensor_f32_with_hash(
        dir.join(&entry.shard_file),
        entry,
        options.compute_data_hash,
    )
}

fn load_tensor_with_expert(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    role: WeightBlockRole,
    layer: Option<u32>,
    expert: Option<u32>,
    options: HfCausalLmLoadOptions,
) -> Result<LoadedSafetensorsTensorU16> {
    let entry = plan
        .entries
        .iter()
        .find(|entry| entry.role == role && entry.layer == layer && entry.expert == expert)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!(
                "HF causal LM missing tensor role {:?} layer {:?} expert {:?}",
                role, layer, expert
            ),
        })?;
    read_safetensors_tensor_u16_with_hash(
        dir.join(&entry.shard_file),
        entry,
        options.compute_data_hash,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_sparse_moe_expert_gate_up(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    architecture: HfArchitectureKind,
    layer: u32,
    num_experts: usize,
    moe_intermediate: usize,
    hidden: usize,
    options: HfCausalLmLoadOptions,
    accounting: &mut LoadAccounting,
) -> Result<Vec<u16>> {
    let per_projection =
        moe_intermediate
            .checked_mul(hidden)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: moe_intermediate,
                reason: "HF MoE expert gate/up projection size overflow".to_string(),
            })?;
    if architecture == HfArchitectureKind::Qwen35Moe {
        let tensor = load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::ExpertGateUpProjection,
            layer,
            options,
            accounting,
        )?;
        require_vec_len(
            "Qwen3.5-MoE packed expert gate/up projection",
            tensor.len(),
            num_experts * per_projection * 2,
        )?;
        return Ok(tensor);
    }
    let total = num_experts
        .checked_mul(per_projection)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: per_projection,
            reason: "HF MoE expert gate/up buffer size overflow".to_string(),
        })?;
    let mut out = Vec::with_capacity(total);
    for expert in 0..num_experts {
        let expert = u32::try_from(expert).map_err(|_| NervaError::InvalidArgument {
            reason: "HF MoE expert index does not fit u32".to_string(),
        })?;
        let gate = load_tensor_with_expert(
            dir,
            plan,
            WeightBlockRole::ExpertGateProjection,
            Some(layer),
            Some(expert),
            options,
        )?;
        let up = load_tensor_with_expert(
            dir,
            plan,
            WeightBlockRole::ExpertUpProjection,
            Some(layer),
            Some(expert),
            options,
        )?;
        require_loaded_tensor_len(&gate, per_projection)?;
        require_loaded_tensor_len(&up, per_projection)?;
        accounting.record(gate.bytes_read, gate.data_hash);
        accounting.record(up.bytes_read, up.data_hash);
        out.extend_from_slice(&gate.values);
        out.extend_from_slice(&up.values);
    }
    Ok(out)
}

fn load_sparse_moe_expert_down(
    dir: &Path,
    plan: &SafetensorsShardPlan,
    architecture: HfArchitectureKind,
    layer: u32,
    num_experts: usize,
    moe_intermediate: usize,
    hidden: usize,
    options: HfCausalLmLoadOptions,
    accounting: &mut LoadAccounting,
) -> Result<Vec<u16>> {
    let per_projection =
        hidden
            .checked_mul(moe_intermediate)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: hidden,
                reason: "HF MoE expert down projection size overflow".to_string(),
            })?;
    if architecture == HfArchitectureKind::Qwen35Moe {
        let tensor = load_layer_tensor(
            dir,
            plan,
            WeightBlockRole::ExpertDownProjection,
            layer,
            options,
            accounting,
        )?;
        require_vec_len(
            "Qwen3.5-MoE packed expert down projection",
            tensor.len(),
            num_experts * per_projection,
        )?;
        return Ok(tensor);
    }
    let total =
        num_experts
            .checked_mul(per_projection)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: per_projection,
                reason: "HF MoE expert down buffer size overflow".to_string(),
            })?;
    let mut out = Vec::with_capacity(total);
    for expert in 0..num_experts {
        let expert = u32::try_from(expert).map_err(|_| NervaError::InvalidArgument {
            reason: "HF MoE expert index does not fit u32".to_string(),
        })?;
        let tensor = load_tensor_with_expert(
            dir,
            plan,
            WeightBlockRole::ExpertDownProjection,
            Some(layer),
            Some(expert),
            options,
        )?;
        require_loaded_tensor_len(&tensor, per_projection)?;
        accounting.record(tensor.bytes_read, tensor.data_hash);
        out.extend_from_slice(&tensor.values);
    }
    Ok(out)
}

fn require_vec_len(label: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!("{label} values length {actual} does not match expected {expected}"),
        })
    }
}

fn require_loaded_tensor_len(tensor: &LoadedSafetensorsTensorU16, expected: usize) -> Result<()> {
    if tensor.values.len() == expected {
        Ok(())
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!(
                "HF causal LM tensor {} values length {} does not match expected {}",
                tensor.name,
                tensor.values.len(),
                expected
            ),
        })
    }
}

fn moe_config_from_metadata(metadata: &HfModelMetadata) -> Result<PrecisionMoeConfig> {
    Ok(PrecisionMoeConfig {
        moe_intermediate: required_metadata_usize(
            metadata.moe_intermediate_size,
            "moe_intermediate_size",
        )?,
        shared_expert_intermediate: metadata.shared_expert_intermediate_size.unwrap_or(0),
        num_experts: required_metadata_usize(metadata.num_experts, "num_experts")?,
        experts_per_token: required_metadata_usize(
            metadata.num_experts_per_tok,
            "num_experts_per_tok",
        )?,
        norm_topk_prob: metadata.norm_topk_prob,
    })
}

fn required_metadata_usize(value: Option<usize>, key: &'static str) -> Result<usize> {
    value.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("HF causal LM metadata is missing {key}"),
    })
}
