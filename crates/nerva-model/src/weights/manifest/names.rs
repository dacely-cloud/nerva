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
        | HfArchitectureKind::DeepSeekV32
        | HfArchitectureKind::DeepSeekV4 => Ok(()),
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
        WeightBlockRole::TokenEmbedding => require_static_tensor(role, layer)
            .map(|()| static_tensor_name(architecture, "embed_tokens.weight")),
        WeightBlockRole::LmHead => {
            require_static_tensor(role, layer).map(|()| static_lm_head_name(architecture))
        }
        WeightBlockRole::FinalNorm => require_static_tensor(role, layer)
            .map(|()| static_tensor_name(architecture, "norm.weight")),
        WeightBlockRole::AttentionNorm if architecture == HfArchitectureKind::DeepSeekV4 => {
            deepseek_v4_layer_name(role, layer, "attn_norm.weight")
        }
        WeightBlockRole::AttentionNorm => {
            layer_name(architecture, role, layer, "input_layernorm.weight")
        }
        WeightBlockRole::MlpNorm if architecture == HfArchitectureKind::DeepSeekV4 => {
            deepseek_v4_layer_name(role, layer, "ffn_norm.weight")
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
        WeightBlockRole::DeepSeekV4HcHeadBase => {
            require_static_tensor(role, layer).map(|()| "hc_head_base".to_string())
        }
        WeightBlockRole::DeepSeekV4HcHeadFn => {
            require_static_tensor(role, layer).map(|()| "hc_head_fn".to_string())
        }
        WeightBlockRole::DeepSeekV4HcHeadScale => {
            require_static_tensor(role, layer).map(|()| "hc_head_scale".to_string())
        }
        WeightBlockRole::DeepSeekV4HcAttnBase => {
            deepseek_v4_layer_name(role, layer, "hc_attn_base")
        }
        WeightBlockRole::DeepSeekV4HcAttnFn => deepseek_v4_layer_name(role, layer, "hc_attn_fn"),
        WeightBlockRole::DeepSeekV4HcAttnScale => {
            deepseek_v4_layer_name(role, layer, "hc_attn_scale")
        }
        WeightBlockRole::DeepSeekV4HcFfnBase => deepseek_v4_layer_name(role, layer, "hc_ffn_base"),
        WeightBlockRole::DeepSeekV4HcFfnFn => deepseek_v4_layer_name(role, layer, "hc_ffn_fn"),
        WeightBlockRole::DeepSeekV4HcFfnScale => {
            deepseek_v4_layer_name(role, layer, "hc_ffn_scale")
        }
        WeightBlockRole::DeepSeekV4AttentionSink => {
            deepseek_v4_layer_name(role, layer, "attn.attn_sink")
        }
        WeightBlockRole::DeepSeekV4WqAProjection => {
            deepseek_v4_layer_name(role, layer, "attn.wq_a.weight")
        }
        WeightBlockRole::DeepSeekV4WqAScale => {
            deepseek_v4_layer_name(role, layer, "attn.wq_a.scale")
        }
        WeightBlockRole::DeepSeekV4WqBProjection => {
            deepseek_v4_layer_name(role, layer, "attn.wq_b.weight")
        }
        WeightBlockRole::DeepSeekV4WqBScale => {
            deepseek_v4_layer_name(role, layer, "attn.wq_b.scale")
        }
        WeightBlockRole::DeepSeekV4QNorm => {
            deepseek_v4_layer_name(role, layer, "attn.q_norm.weight")
        }
        WeightBlockRole::DeepSeekV4WkvProjection => {
            deepseek_v4_layer_name(role, layer, "attn.wkv.weight")
        }
        WeightBlockRole::DeepSeekV4WkvScale => {
            deepseek_v4_layer_name(role, layer, "attn.wkv.scale")
        }
        WeightBlockRole::DeepSeekV4KvNorm => {
            deepseek_v4_layer_name(role, layer, "attn.kv_norm.weight")
        }
        WeightBlockRole::DeepSeekV4WoAProjection => {
            deepseek_v4_layer_name(role, layer, "attn.wo_a.weight")
        }
        WeightBlockRole::DeepSeekV4WoAScale => {
            deepseek_v4_layer_name(role, layer, "attn.wo_a.scale")
        }
        WeightBlockRole::DeepSeekV4WoBProjection => {
            deepseek_v4_layer_name(role, layer, "attn.wo_b.weight")
        }
        WeightBlockRole::DeepSeekV4WoBScale => {
            deepseek_v4_layer_name(role, layer, "attn.wo_b.scale")
        }
        WeightBlockRole::DeepSeekV4CompressorApe => {
            deepseek_v4_layer_name(role, layer, "attn.compressor.ape")
        }
        WeightBlockRole::DeepSeekV4CompressorWkvProjection => {
            deepseek_v4_layer_name(role, layer, "attn.compressor.wkv.weight")
        }
        WeightBlockRole::DeepSeekV4CompressorWkvScale => {
            deepseek_v4_layer_name(role, layer, "attn.compressor.wkv.scale")
        }
        WeightBlockRole::DeepSeekV4CompressorWgateProjection => {
            deepseek_v4_layer_name(role, layer, "attn.compressor.wgate.weight")
        }
        WeightBlockRole::DeepSeekV4CompressorWgateScale => {
            deepseek_v4_layer_name(role, layer, "attn.compressor.wgate.scale")
        }
        WeightBlockRole::DeepSeekV4CompressorNorm => {
            deepseek_v4_layer_name(role, layer, "attn.compressor.norm.weight")
        }
        WeightBlockRole::DeepSeekV4IndexerWqBProjection => {
            deepseek_v4_layer_name(role, layer, "attn.indexer.wq_b.weight")
        }
        WeightBlockRole::DeepSeekV4IndexerWqBScale => {
            deepseek_v4_layer_name(role, layer, "attn.indexer.wq_b.scale")
        }
        WeightBlockRole::DeepSeekV4IndexerCompressorApe => {
            deepseek_v4_layer_name(role, layer, "attn.indexer.compressor.ape")
        }
        WeightBlockRole::DeepSeekV4IndexerCompressorWkvProjection => {
            deepseek_v4_layer_name(role, layer, "attn.indexer.compressor.wkv.weight")
        }
        WeightBlockRole::DeepSeekV4IndexerCompressorWkvScale => {
            deepseek_v4_layer_name(role, layer, "attn.indexer.compressor.wkv.scale")
        }
        WeightBlockRole::DeepSeekV4IndexerCompressorWgateProjection => {
            deepseek_v4_layer_name(role, layer, "attn.indexer.compressor.wgate.weight")
        }
        WeightBlockRole::DeepSeekV4IndexerCompressorWgateScale => {
            deepseek_v4_layer_name(role, layer, "attn.indexer.compressor.wgate.scale")
        }
        WeightBlockRole::DeepSeekV4IndexerCompressorNorm => {
            deepseek_v4_layer_name(role, layer, "attn.indexer.compressor.norm.weight")
        }
        WeightBlockRole::DeepSeekV4IndexerWeightsProjection => {
            deepseek_v4_layer_name(role, layer, "attn.indexer.weights_proj.weight")
        }
        WeightBlockRole::DeepSeekV4IndexerWeightsScale => {
            deepseek_v4_layer_name(role, layer, "attn.indexer.weights_proj.scale")
        }
        WeightBlockRole::DeepSeekV4HashRouteTable => {
            deepseek_v4_layer_name(role, layer, "ffn.gate.tid2eid")
        }
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
        WeightBlockRole::RouterProjection if architecture == HfArchitectureKind::DeepSeekV4 => {
            deepseek_v4_layer_name(role, layer, "ffn.gate.weight")
        }
        WeightBlockRole::RouterProjection => {
            if architecture == HfArchitectureKind::MixtralMoe {
                layer_name(architecture, role, layer, "block_sparse_moe.gate.weight")
            } else {
                layer_name(architecture, role, layer, "mlp.gate.weight")
            }
        }
        WeightBlockRole::RouterCorrectionBias if architecture == HfArchitectureKind::DeepSeekV4 => {
            deepseek_v4_layer_name(role, layer, "ffn.gate.bias")
        }
        WeightBlockRole::RouterCorrectionBias => deepseek_v3_layer_name(
            architecture,
            role,
            layer,
            "mlp.gate.e_score_correction_bias",
        ),
        WeightBlockRole::SharedExpertGateProjection
            if architecture == HfArchitectureKind::DeepSeekV4 =>
        {
            deepseek_v4_layer_name(role, layer, "ffn.shared_experts.w1.weight")
        }
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
            if architecture == HfArchitectureKind::DeepSeekV4 =>
        {
            deepseek_v4_layer_name(role, layer, "ffn.shared_experts.w3.weight")
        }
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
            if architecture == HfArchitectureKind::DeepSeekV4 =>
        {
            deepseek_v4_layer_name(role, layer, "ffn.shared_experts.w2.weight")
        }
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
        | WeightBlockRole::ExpertDownScaleInv
        | WeightBlockRole::DeepSeekV4ExpertGateScale
        | WeightBlockRole::DeepSeekV4ExpertUpScale
        | WeightBlockRole::DeepSeekV4ExpertDownScale => Err(NervaError::InvalidArgument {
            reason: format!("weight block {} requires an expert index", role.as_str()),
        }),
        WeightBlockRole::DeepSeekV4SharedExpertGateScale => {
            deepseek_v4_layer_name(role, layer, "ffn.shared_experts.w1.scale")
        }
        WeightBlockRole::DeepSeekV4SharedExpertUpScale => {
            deepseek_v4_layer_name(role, layer, "ffn.shared_experts.w3.scale")
        }
        WeightBlockRole::DeepSeekV4SharedExpertDownScale => {
            deepseek_v4_layer_name(role, layer, "ffn.shared_experts.w2.scale")
        }
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
            HfArchitectureKind::DeepSeekV4 => {
                format!("ffn.experts.{expert}.w1.weight")
            }
            HfArchitectureKind::MixtralMoe => {
                format!("block_sparse_moe.experts.{expert}.w1.weight")
            }
            _ => format!("mlp.experts.{expert}.gate_proj.weight"),
        },
        WeightBlockRole::ExpertUpProjection => match architecture {
            HfArchitectureKind::DeepSeekV4 => {
                format!("ffn.experts.{expert}.w3.weight")
            }
            HfArchitectureKind::MixtralMoe => {
                format!("block_sparse_moe.experts.{expert}.w3.weight")
            }
            _ => format!("mlp.experts.{expert}.up_proj.weight"),
        },
        WeightBlockRole::ExpertDownProjection => match architecture {
            HfArchitectureKind::DeepSeekV4 => {
                format!("ffn.experts.{expert}.w2.weight")
            }
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
        WeightBlockRole::DeepSeekV4ExpertGateScale => format!("ffn.experts.{expert}.w1.scale"),
        WeightBlockRole::DeepSeekV4ExpertUpScale => format!("ffn.experts.{expert}.w3.scale"),
        WeightBlockRole::DeepSeekV4ExpertDownScale => format!("ffn.experts.{expert}.w2.scale"),
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
    if architecture == HfArchitectureKind::DeepSeekV4 {
        return match suffix {
            "embed_tokens.weight" => "embed.weight".to_string(),
            "norm.weight" => "norm.weight".to_string(),
            _ => suffix.to_string(),
        };
    }
    if uses_language_model_prefix(architecture) {
        format!("model.language_model.{suffix}")
    } else {
        format!("model.{suffix}")
    }
}

fn static_lm_head_name(architecture: HfArchitectureKind) -> String {
    if architecture == HfArchitectureKind::DeepSeekV4 {
        "head.weight".to_string()
    } else {
        "lm_head.weight".to_string()
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

fn deepseek_v4_layer_name(
    role: WeightBlockRole,
    layer: Option<u32>,
    suffix: &'static str,
) -> Result<String> {
    let layer = layer.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("weight block {} must have a layer", role.as_str()),
    })?;
    Ok(format!("layers.{layer}.{suffix}"))
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
    let prefix = if architecture == HfArchitectureKind::DeepSeekV4 {
        "layers"
    } else if uses_language_model_prefix(architecture) {
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
        | WeightBlockRole::DeepSeekV4HcHeadBase
        | WeightBlockRole::DeepSeekV4HcHeadScale
        | WeightBlockRole::DeepSeekV4HcAttnBase
        | WeightBlockRole::DeepSeekV4HcAttnScale
        | WeightBlockRole::DeepSeekV4HcFfnBase
        | WeightBlockRole::DeepSeekV4HcFfnScale
        | WeightBlockRole::DeepSeekV4AttentionSink
        | WeightBlockRole::DeepSeekV4QNorm
        | WeightBlockRole::DeepSeekV4KvNorm
        | WeightBlockRole::DeepSeekV4CompressorNorm
        | WeightBlockRole::DeepSeekV4IndexerCompressorNorm
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
        | WeightBlockRole::DeepSeekV4HcHeadFn
        | WeightBlockRole::DeepSeekV4HcAttnFn
        | WeightBlockRole::DeepSeekV4HcFfnFn
        | WeightBlockRole::DeepSeekV4WqAProjection
        | WeightBlockRole::DeepSeekV4WqAScale
        | WeightBlockRole::DeepSeekV4WqBProjection
        | WeightBlockRole::DeepSeekV4WqBScale
        | WeightBlockRole::DeepSeekV4WkvProjection
        | WeightBlockRole::DeepSeekV4WkvScale
        | WeightBlockRole::DeepSeekV4WoAProjection
        | WeightBlockRole::DeepSeekV4WoAScale
        | WeightBlockRole::DeepSeekV4WoBProjection
        | WeightBlockRole::DeepSeekV4WoBScale
        | WeightBlockRole::DeepSeekV4CompressorApe
        | WeightBlockRole::DeepSeekV4CompressorWkvProjection
        | WeightBlockRole::DeepSeekV4CompressorWkvScale
        | WeightBlockRole::DeepSeekV4CompressorWgateProjection
        | WeightBlockRole::DeepSeekV4CompressorWgateScale
        | WeightBlockRole::DeepSeekV4IndexerWqBProjection
        | WeightBlockRole::DeepSeekV4IndexerWqBScale
        | WeightBlockRole::DeepSeekV4IndexerCompressorApe
        | WeightBlockRole::DeepSeekV4IndexerCompressorWkvProjection
        | WeightBlockRole::DeepSeekV4IndexerCompressorWkvScale
        | WeightBlockRole::DeepSeekV4IndexerCompressorWgateProjection
        | WeightBlockRole::DeepSeekV4IndexerCompressorWgateScale
        | WeightBlockRole::DeepSeekV4IndexerWeightsProjection
        | WeightBlockRole::DeepSeekV4IndexerWeightsScale
        | WeightBlockRole::DeepSeekV4HashRouteTable
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
        | WeightBlockRole::DeepSeekV4SharedExpertGateScale
        | WeightBlockRole::DeepSeekV4SharedExpertUpScale
        | WeightBlockRole::DeepSeekV4SharedExpertDownScale
        | WeightBlockRole::SharedExpertRouterProjection
        | WeightBlockRole::LmHead => 2,
        WeightBlockRole::ExpertGateUpProjection
        | WeightBlockRole::ExpertDownProjection
        | WeightBlockRole::ExpertGateScaleInv
        | WeightBlockRole::ExpertUpScaleInv
        | WeightBlockRole::ExpertDownScaleInv
        | WeightBlockRole::DeepSeekV4ExpertGateScale
        | WeightBlockRole::DeepSeekV4ExpertUpScale
        | WeightBlockRole::DeepSeekV4ExpertDownScale => 3,
    }
}
