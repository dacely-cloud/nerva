use nerva_core::types::error::{NervaError, Result};

use crate::hf::architecture::HfArchitectureKind;
use crate::weights::layout::entry::WeightBlockRole;

pub(crate) fn ensure_supported_hf_tensor_names(architecture: HfArchitectureKind) -> Result<()> {
    match architecture {
        HfArchitectureKind::Llama | HfArchitectureKind::Mistral | HfArchitectureKind::Qwen2 => {
            Ok(())
        }
        HfArchitectureKind::Gemma | HfArchitectureKind::Unknown => {
            Err(NervaError::InvalidArgument {
                reason: format!(
                    "HF tensor names for architecture {} are not implemented",
                    architecture.as_str()
                ),
            })
        }
    }
}

pub(crate) fn hf_tensor_name(
    architecture: HfArchitectureKind,
    role: WeightBlockRole,
    layer: Option<u32>,
) -> Result<String> {
    ensure_supported_hf_tensor_names(architecture)?;
    match role {
        WeightBlockRole::TokenEmbedding => {
            require_static_tensor(role, layer).map(|()| "model.embed_tokens.weight".to_string())
        }
        WeightBlockRole::LmHead => {
            require_static_tensor(role, layer).map(|()| "lm_head.weight".to_string())
        }
        WeightBlockRole::FinalNorm => {
            require_static_tensor(role, layer).map(|()| "model.norm.weight".to_string())
        }
        WeightBlockRole::AttentionNorm => layer_name(role, layer, "input_layernorm.weight"),
        WeightBlockRole::MlpNorm => layer_name(role, layer, "post_attention_layernorm.weight"),
        WeightBlockRole::QueryProjection => layer_name(role, layer, "self_attn.q_proj.weight"),
        WeightBlockRole::QueryBias => layer_name(role, layer, "self_attn.q_proj.bias"),
        WeightBlockRole::KeyProjection => layer_name(role, layer, "self_attn.k_proj.weight"),
        WeightBlockRole::KeyBias => layer_name(role, layer, "self_attn.k_proj.bias"),
        WeightBlockRole::ValueProjection => layer_name(role, layer, "self_attn.v_proj.weight"),
        WeightBlockRole::ValueBias => layer_name(role, layer, "self_attn.v_proj.bias"),
        WeightBlockRole::OutputProjection => layer_name(role, layer, "self_attn.o_proj.weight"),
        WeightBlockRole::OutputBias => layer_name(role, layer, "self_attn.o_proj.bias"),
        WeightBlockRole::GateProjection => layer_name(role, layer, "mlp.gate_proj.weight"),
        WeightBlockRole::UpProjection => layer_name(role, layer, "mlp.up_proj.weight"),
        WeightBlockRole::DownProjection => layer_name(role, layer, "mlp.down_proj.weight"),
    }
}

fn require_static_tensor(role: WeightBlockRole, layer: Option<u32>) -> Result<()> {
    if layer.is_none() {
        Ok(())
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!("weight block {} must not have a layer", role.as_str()),
        })
    }
}

fn layer_name(role: WeightBlockRole, layer: Option<u32>, suffix: &'static str) -> Result<String> {
    let layer = layer.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("weight block {} must have a layer", role.as_str()),
    })?;
    Ok(format!("model.layers.{layer}.{suffix}"))
}

pub(crate) fn weight_block_rank(role: WeightBlockRole) -> u8 {
    match role {
        WeightBlockRole::AttentionNorm
        | WeightBlockRole::QueryBias
        | WeightBlockRole::KeyBias
        | WeightBlockRole::ValueBias
        | WeightBlockRole::OutputBias
        | WeightBlockRole::MlpNorm
        | WeightBlockRole::FinalNorm => 1,
        WeightBlockRole::TokenEmbedding
        | WeightBlockRole::QueryProjection
        | WeightBlockRole::KeyProjection
        | WeightBlockRole::ValueProjection
        | WeightBlockRole::OutputProjection
        | WeightBlockRole::GateProjection
        | WeightBlockRole::UpProjection
        | WeightBlockRole::DownProjection
        | WeightBlockRole::LmHead => 2,
    }
}
