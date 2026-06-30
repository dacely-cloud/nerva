use nerva_core::types::error::{NervaError, Result};

use crate::hf::architecture::HfArchitectureKind;
use crate::weights::layout::entry::WeightBlockRole;

pub(crate) fn ensure_supported_hf_tensor_names(architecture: HfArchitectureKind) -> Result<()> {
    match architecture {
        HfArchitectureKind::Llama
        | HfArchitectureKind::Mistral
        | HfArchitectureKind::MixtralMoe
        | HfArchitectureKind::Qwen2
        | HfArchitectureKind::Qwen2Moe
        | HfArchitectureKind::Qwen3
        | HfArchitectureKind::Qwen3Moe
        | HfArchitectureKind::Qwen35
        | HfArchitectureKind::Qwen35Moe
        | HfArchitectureKind::DeepSeekV3
        | HfArchitectureKind::DeepSeekV32 => Ok(()),
        HfArchitectureKind::DeepSeekV4
        | HfArchitectureKind::Gemma
        | HfArchitectureKind::Unknown => Err(NervaError::InvalidArgument {
            reason: format!(
                "HF tensor names for architecture {} are not implemented",
                architecture.as_str()
            ),
        }),
    }
}

pub(crate) fn hf_tensor_name(
    architecture: HfArchitectureKind,
    role: WeightBlockRole,
    layer: Option<u32>,
) -> Result<String> {
    ensure_supported_hf_tensor_names(architecture)?;
    match role {
        WeightBlockRole::TokenEmbedding => require_static_tensor(role, layer)
            .map(|()| static_tensor_name(architecture, "embed_tokens.weight")),
        WeightBlockRole::LmHead => {
            require_static_tensor(role, layer).map(|()| "lm_head.weight".to_string())
        }
        WeightBlockRole::FinalNorm => require_static_tensor(role, layer)
            .map(|()| static_tensor_name(architecture, "norm.weight")),
        WeightBlockRole::AttentionNorm => {
            layer_name(architecture, role, layer, "input_layernorm.weight")
        }
        WeightBlockRole::MlpNorm => {
            layer_name(architecture, role, layer, "post_attention_layernorm.weight")
        }
        WeightBlockRole::QueryProjection => {
            layer_name(architecture, role, layer, "self_attn.q_proj.weight")
        }
        WeightBlockRole::QueryNorm => {
            layer_name(architecture, role, layer, "self_attn.q_norm.weight")
        }
        WeightBlockRole::QueryBias => {
            layer_name(architecture, role, layer, "self_attn.q_proj.bias")
        }
        WeightBlockRole::KeyProjection => {
            layer_name(architecture, role, layer, "self_attn.k_proj.weight")
        }
        WeightBlockRole::KeyNorm => {
            layer_name(architecture, role, layer, "self_attn.k_norm.weight")
        }
        WeightBlockRole::KeyBias => layer_name(architecture, role, layer, "self_attn.k_proj.bias"),
        WeightBlockRole::ValueProjection => {
            layer_name(architecture, role, layer, "self_attn.v_proj.weight")
        }
        WeightBlockRole::ValueBias => {
            layer_name(architecture, role, layer, "self_attn.v_proj.bias")
        }
        WeightBlockRole::OutputProjection => {
            layer_name(architecture, role, layer, "self_attn.o_proj.weight")
        }
        WeightBlockRole::OutputBias => {
            layer_name(architecture, role, layer, "self_attn.o_proj.bias")
        }
        WeightBlockRole::DeepSeekQALoraProjection => {
            deepseek_v3_layer_name(architecture, role, layer, "self_attn.q_a_proj.weight")
        }
        WeightBlockRole::DeepSeekQALoraScaleInv => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "self_attn.q_a_proj.weight_scale_inv",
        ),
        WeightBlockRole::DeepSeekQALoraNorm => {
            deepseek_v3_layer_name(architecture, role, layer, "self_attn.q_a_layernorm.weight")
        }
        WeightBlockRole::DeepSeekQBProjection => {
            deepseek_v3_layer_name(architecture, role, layer, "self_attn.q_b_proj.weight")
        }
        WeightBlockRole::DeepSeekQBScaleInv => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "self_attn.q_b_proj.weight_scale_inv",
        ),
        WeightBlockRole::DeepSeekKvAProjection => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "self_attn.kv_a_proj_with_mqa.weight",
        ),
        WeightBlockRole::DeepSeekKvAScaleInv => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "self_attn.kv_a_proj_with_mqa.weight_scale_inv",
        ),
        WeightBlockRole::DeepSeekKvANorm => {
            deepseek_v3_layer_name(architecture, role, layer, "self_attn.kv_a_layernorm.weight")
        }
        WeightBlockRole::DeepSeekKvBProjection => {
            deepseek_v3_layer_name(architecture, role, layer, "self_attn.kv_b_proj.weight")
        }
        WeightBlockRole::DeepSeekKvBScaleInv => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "self_attn.kv_b_proj.weight_scale_inv",
        ),
        WeightBlockRole::DeepSeekOutputScaleInv => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "self_attn.o_proj.weight_scale_inv",
        ),
        WeightBlockRole::DeepSeekIndexerQueryProjection => {
            deepseek_v3_layer_name(architecture, role, layer, "self_attn.indexer.wq_b.weight")
        }
        WeightBlockRole::DeepSeekIndexerQueryScaleInv => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "self_attn.indexer.wq_b.weight_scale_inv",
        ),
        WeightBlockRole::DeepSeekIndexerKeyProjection => {
            deepseek_v3_layer_name(architecture, role, layer, "self_attn.indexer.wk.weight")
        }
        WeightBlockRole::DeepSeekIndexerKeyScaleInv => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "self_attn.indexer.wk.weight_scale_inv",
        ),
        WeightBlockRole::DeepSeekIndexerKeyNorm => {
            deepseek_v3_layer_name(architecture, role, layer, "self_attn.indexer.k_norm.weight")
        }
        WeightBlockRole::DeepSeekIndexerKeyNormBias => {
            deepseek_v3_layer_name(architecture, role, layer, "self_attn.indexer.k_norm.bias")
        }
        WeightBlockRole::DeepSeekIndexerWeightsProjection => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "self_attn.indexer.weights_proj.weight",
        ),
        WeightBlockRole::LinearConvProjection => {
            layer_name(architecture, role, layer, "linear_attn.conv1d.weight")
        }
        WeightBlockRole::LinearQkvProjection => {
            layer_name(architecture, role, layer, "linear_attn.in_proj_qkv.weight")
        }
        WeightBlockRole::LinearZProjection => {
            layer_name(architecture, role, layer, "linear_attn.in_proj_z.weight")
        }
        WeightBlockRole::LinearBProjection => {
            layer_name(architecture, role, layer, "linear_attn.in_proj_b.weight")
        }
        WeightBlockRole::LinearAProjection => {
            layer_name(architecture, role, layer, "linear_attn.in_proj_a.weight")
        }
        WeightBlockRole::LinearDtBias => {
            layer_name(architecture, role, layer, "linear_attn.dt_bias")
        }
        WeightBlockRole::LinearALog => layer_name(architecture, role, layer, "linear_attn.A_log"),
        WeightBlockRole::LinearNorm => {
            layer_name(architecture, role, layer, "linear_attn.norm.weight")
        }
        WeightBlockRole::LinearOutputProjection => {
            layer_name(architecture, role, layer, "linear_attn.out_proj.weight")
        }
        WeightBlockRole::GateProjection => {
            layer_name(architecture, role, layer, "mlp.gate_proj.weight")
        }
        WeightBlockRole::UpProjection => {
            layer_name(architecture, role, layer, "mlp.up_proj.weight")
        }
        WeightBlockRole::DownProjection => {
            layer_name(architecture, role, layer, "mlp.down_proj.weight")
        }
        WeightBlockRole::GateScaleInv => {
            deepseek_v3_layer_name(architecture, role, layer, "mlp.gate_proj.weight_scale_inv")
        }
        WeightBlockRole::UpScaleInv => {
            deepseek_v3_layer_name(architecture, role, layer, "mlp.up_proj.weight_scale_inv")
        }
        WeightBlockRole::DownScaleInv => {
            deepseek_v3_layer_name(architecture, role, layer, "mlp.down_proj.weight_scale_inv")
        }
        WeightBlockRole::RouterProjection => {
            if architecture == HfArchitectureKind::MixtralMoe {
                layer_name(architecture, role, layer, "block_sparse_moe.gate.weight")
            } else {
                layer_name(architecture, role, layer, "mlp.gate.weight")
            }
        }
        WeightBlockRole::RouterCorrectionBias => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "mlp.gate.e_score_correction_bias",
        ),
        WeightBlockRole::SharedExpertGateProjection
            if matches!(
                architecture,
                HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32
            ) =>
        {
            layer_name(
                architecture,
                role,
                layer,
                "mlp.shared_experts.gate_proj.weight",
            )
        }
        WeightBlockRole::SharedExpertGateProjection => layer_name(
            architecture,
            role,
            layer,
            "mlp.shared_expert.gate_proj.weight",
        ),
        WeightBlockRole::SharedExpertUpProjection
            if matches!(
                architecture,
                HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32
            ) =>
        {
            layer_name(
                architecture,
                role,
                layer,
                "mlp.shared_experts.up_proj.weight",
            )
        }
        WeightBlockRole::SharedExpertUpProjection => layer_name(
            architecture,
            role,
            layer,
            "mlp.shared_expert.up_proj.weight",
        ),
        WeightBlockRole::SharedExpertDownProjection
            if matches!(
                architecture,
                HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32
            ) =>
        {
            layer_name(
                architecture,
                role,
                layer,
                "mlp.shared_experts.down_proj.weight",
            )
        }
        WeightBlockRole::SharedExpertDownProjection => layer_name(
            architecture,
            role,
            layer,
            "mlp.shared_expert.down_proj.weight",
        ),
        WeightBlockRole::SharedExpertGateScaleInv => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "mlp.shared_experts.gate_proj.weight_scale_inv",
        ),
        WeightBlockRole::SharedExpertUpScaleInv => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "mlp.shared_experts.up_proj.weight_scale_inv",
        ),
        WeightBlockRole::SharedExpertDownScaleInv => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "mlp.shared_experts.down_proj.weight_scale_inv",
        ),
        WeightBlockRole::SharedExpertRouterProjection => {
            layer_name(architecture, role, layer, "mlp.shared_expert_gate.weight")
        }
        WeightBlockRole::ExpertGateUpProjection
            if architecture == HfArchitectureKind::Qwen35Moe =>
        {
            layer_name(architecture, role, layer, "mlp.experts.gate_up_proj")
        }
        WeightBlockRole::ExpertDownProjection if architecture == HfArchitectureKind::Qwen35Moe => {
            layer_name(architecture, role, layer, "mlp.experts.down_proj")
        }
        WeightBlockRole::ExpertGateProjection
        | WeightBlockRole::ExpertUpProjection
        | WeightBlockRole::ExpertGateUpProjection
        | WeightBlockRole::ExpertDownProjection
        | WeightBlockRole::ExpertGateScaleInv
        | WeightBlockRole::ExpertUpScaleInv
        | WeightBlockRole::ExpertDownScaleInv => Err(NervaError::InvalidArgument {
            reason: format!("weight block {} requires an expert index", role.as_str()),
        }),
    }
}

pub(crate) fn hf_expert_tensor_name(
    architecture: HfArchitectureKind,
    role: WeightBlockRole,
    layer: Option<u32>,
    expert: u32,
) -> Result<String> {
    ensure_supported_hf_tensor_names(architecture)?;
    let suffix = match role {
        WeightBlockRole::ExpertGateProjection => match architecture {
            HfArchitectureKind::MixtralMoe => {
                format!("block_sparse_moe.experts.{expert}.w1.weight")
            }
            _ => format!("mlp.experts.{expert}.gate_proj.weight"),
        },
        WeightBlockRole::ExpertUpProjection => match architecture {
            HfArchitectureKind::MixtralMoe => {
                format!("block_sparse_moe.experts.{expert}.w3.weight")
            }
            _ => format!("mlp.experts.{expert}.up_proj.weight"),
        },
        WeightBlockRole::ExpertDownProjection => match architecture {
            HfArchitectureKind::MixtralMoe => {
                format!("block_sparse_moe.experts.{expert}.w2.weight")
            }
            _ => format!("mlp.experts.{expert}.down_proj.weight"),
        },
        WeightBlockRole::ExpertGateScaleInv => {
            format!("mlp.experts.{expert}.gate_proj.weight_scale_inv")
        }
        WeightBlockRole::ExpertUpScaleInv => {
            format!("mlp.experts.{expert}.up_proj.weight_scale_inv")
        }
        WeightBlockRole::ExpertDownScaleInv => {
            format!("mlp.experts.{expert}.down_proj.weight_scale_inv")
        }
        _ => {
            return Err(NervaError::InvalidArgument {
                reason: format!("weight block {} is not an expert tensor", role.as_str()),
            });
        }
    };
    layer_name_owned(architecture, role, layer, suffix)
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

fn static_tensor_name(architecture: HfArchitectureKind, suffix: &'static str) -> String {
    if uses_language_model_prefix(architecture) {
        format!("model.language_model.{suffix}")
    } else {
        format!("model.{suffix}")
    }
}

fn layer_name(
    architecture: HfArchitectureKind,
    role: WeightBlockRole,
    layer: Option<u32>,
    suffix: &'static str,
) -> Result<String> {
    layer_name_owned(architecture, role, layer, suffix.to_string())
}

fn deepseek_v3_layer_name(
    architecture: HfArchitectureKind,
    role: WeightBlockRole,
    layer: Option<u32>,
    suffix: &'static str,
) -> Result<String> {
    if matches!(
        architecture,
        HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32
    ) {
        layer_name(architecture, role, layer, suffix)
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!(
                "weight block {} is only defined for DeepSeek V3-family manifests",
                role.as_str()
            ),
        })
    }
}

fn layer_name_owned(
    architecture: HfArchitectureKind,
    role: WeightBlockRole,
    layer: Option<u32>,
    suffix: String,
) -> Result<String> {
    let layer = layer.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("weight block {} must have a layer", role.as_str()),
    })?;
    let prefix = if uses_language_model_prefix(architecture) {
        "model.language_model.layers"
    } else {
        "model.layers"
    };
    Ok(format!("{prefix}.{layer}.{suffix}"))
}

fn uses_language_model_prefix(architecture: HfArchitectureKind) -> bool {
    matches!(
        architecture,
        HfArchitectureKind::Qwen35 | HfArchitectureKind::Qwen35Moe
    )
}

pub(crate) fn weight_block_rank(role: WeightBlockRole) -> u8 {
    match role {
        WeightBlockRole::AttentionNorm
        | WeightBlockRole::QueryNorm
        | WeightBlockRole::QueryBias
        | WeightBlockRole::KeyNorm
        | WeightBlockRole::KeyBias
        | WeightBlockRole::ValueBias
        | WeightBlockRole::OutputBias
        | WeightBlockRole::DeepSeekQALoraNorm
        | WeightBlockRole::DeepSeekKvANorm
        | WeightBlockRole::DeepSeekIndexerKeyNorm
        | WeightBlockRole::DeepSeekIndexerKeyNormBias
        | WeightBlockRole::LinearDtBias
        | WeightBlockRole::LinearALog
        | WeightBlockRole::LinearNorm
        | WeightBlockRole::MlpNorm
        | WeightBlockRole::RouterCorrectionBias
        | WeightBlockRole::FinalNorm => 1,
        WeightBlockRole::TokenEmbedding
        | WeightBlockRole::QueryProjection
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
        | WeightBlockRole::LinearOutputProjection
        | WeightBlockRole::GateProjection
        | WeightBlockRole::UpProjection
        | WeightBlockRole::DownProjection
        | WeightBlockRole::GateScaleInv
        | WeightBlockRole::UpScaleInv
        | WeightBlockRole::DownScaleInv
        | WeightBlockRole::RouterProjection
        | WeightBlockRole::ExpertGateProjection
        | WeightBlockRole::ExpertUpProjection
        | WeightBlockRole::SharedExpertGateProjection
        | WeightBlockRole::SharedExpertUpProjection
        | WeightBlockRole::SharedExpertDownProjection
        | WeightBlockRole::SharedExpertGateScaleInv
        | WeightBlockRole::SharedExpertUpScaleInv
        | WeightBlockRole::SharedExpertDownScaleInv
        | WeightBlockRole::SharedExpertRouterProjection
        | WeightBlockRole::LmHead => 2,
        WeightBlockRole::ExpertGateUpProjection
        | WeightBlockRole::ExpertDownProjection
        | WeightBlockRole::ExpertGateScaleInv
        | WeightBlockRole::ExpertUpScaleInv
        | WeightBlockRole::ExpertDownScaleInv => 3,
    }
}
