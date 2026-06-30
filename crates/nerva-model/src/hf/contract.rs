use nerva_core::types::error::{NervaError, Result};

use crate::hf::architecture::HfArchitectureKind;
use crate::hf::deepseek_runtime::validate_deepseek_exact_runtime_contract;
use crate::hf::linear_attention::{ConvStateLayout, Qwen35GatedDeltaNetSpec};
use crate::hf::metadata::{HfAttentionLayerKind, HfMlpLayerKind, HfModelMetadata};

const EXACT_RUNTIME_MOE_EXPERTS_MAX: usize = 256;
const EXACT_RUNTIME_MOE_TOP_K_MAX: usize = 16;

pub fn validate_exact_runtime_contract(metadata: &HfModelMetadata) -> Result<()> {
    validate_weight_layout_contract(metadata)?;
    if metadata.architecture.is_deepseek() {
        return validate_deepseek_exact_runtime_contract(metadata);
    }
    validate_exact_runtime_attention(metadata)
}

pub fn validate_weight_layout_contract(metadata: &HfModelMetadata) -> Result<()> {
    validate_supported_layout_architecture(metadata)?;
    validate_attention_layers(metadata)?;
    validate_qk_norm(metadata)?;
    validate_mlp_activation(metadata)?;
    validate_mlp_bias(metadata)?;
    validate_moe_metadata(metadata)
}

fn validate_supported_layout_architecture(metadata: &HfModelMetadata) -> Result<()> {
    match metadata.architecture {
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
                    "HF architecture {} is not supported by the exact runtime contract",
                    metadata.architecture.as_str()
                ),
            })
        }
    }
}

fn validate_exact_runtime_attention(metadata: &HfModelMetadata) -> Result<()> {
    match metadata.architecture {
        HfArchitectureKind::Qwen35 => {
            let linear_layers = metadata
                .attention_layer_types
                .iter()
                .filter(|kind| **kind == HfAttentionLayerKind::Linear)
                .count();
            if linear_layers == 0 {
                Ok(())
            } else {
                let linear_state = qwen35_linear_attention_state(metadata);
                Err(NervaError::InvalidArgument {
                    reason: format!(
                        "HF architecture {} is recognized, but the exact runtime does not yet implement Qwen3.5 linear_attention layers: {linear_layers} layers require GatedDeltaNet conv/recurrent state kernels ({linear_state})",
                        metadata.architecture.as_str(),
                    ),
                })
            }
        }
        HfArchitectureKind::Qwen35Moe => validate_qwen35_moe_linear_attention_runtime(metadata),
        _ => Ok(()),
    }
}

fn validate_qwen35_moe_linear_attention_runtime(metadata: &HfModelMetadata) -> Result<()> {
    for (index, attention) in metadata.attention_layer_types.iter().enumerate() {
        if *attention == HfAttentionLayerKind::Linear
            && metadata.mlp_layer_types.get(index) != Some(&HfMlpLayerKind::SparseMoe)
        {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "HF architecture {} has linear_attention layer {index} without a sparse MoE MLP; the exact runtime only implements Qwen3.5 GatedDeltaNet-MoE layers",
                    metadata.architecture.as_str(),
                ),
            });
        }
    }
    Ok(())
}

fn qwen35_linear_attention_state(metadata: &HfModelMetadata) -> String {
    match Qwen35GatedDeltaNetSpec::from_metadata(metadata) {
        Ok(Some(spec)) => match spec.state_shape(1, 0, ConvStateLayout::StateLenDim) {
            Ok(shape) => {
                format!(
                    "conv_kernel={}, key_heads={}, key_dim={}, value_heads={}, value_dim={}, conv_state={:?}, recurrent_state={:?}",
                    spec.conv_kernel,
                    spec.key_heads,
                    spec.key_head_dim,
                    spec.value_heads,
                    spec.value_head_dim,
                    shape.conv_state,
                    shape.recurrent_state,
                )
            }
            Err(_) => "linear attention dimensions unavailable".to_string(),
        },
        _ => "linear attention dimensions unavailable".to_string(),
    }
}
fn validate_attention_layers(metadata: &HfModelMetadata) -> Result<()> {
    if metadata.has_linear_attention_layers() {
        if matches!(
            metadata.architecture,
            HfArchitectureKind::Qwen35 | HfArchitectureKind::Qwen35Moe
        ) {
            validate_qwen35_linear_attention_metadata(metadata)?;
            return Ok(());
        }
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "HF linear_attention layers require a dedicated attention runtime and cannot use the standard QKV attention CUDA path"
            ),
        });
    }
    Ok(())
}

fn validate_qwen35_linear_attention_metadata(metadata: &HfModelMetadata) -> Result<()> {
    for (name, value) in [
        ("linear_conv_kernel_dim", metadata.linear_conv_kernel_dim),
        ("linear_key_head_dim", metadata.linear_key_head_dim),
        ("linear_value_head_dim", metadata.linear_value_head_dim),
        ("linear_num_key_heads", metadata.linear_num_key_heads),
        ("linear_num_value_heads", metadata.linear_num_value_heads),
    ] {
        if value.unwrap_or(0) == 0 {
            return Err(NervaError::InvalidArgument {
                reason: format!("Qwen3.5 linear_attention is missing {name}"),
            });
        }
    }
    Ok(())
}

fn validate_qk_norm(metadata: &HfModelMetadata) -> Result<()> {
    if metadata.qk_norm
        && !matches!(
            metadata.architecture,
            HfArchitectureKind::Qwen2
                | HfArchitectureKind::Qwen3
                | HfArchitectureKind::Qwen3Moe
                | HfArchitectureKind::Qwen35
                | HfArchitectureKind::Qwen35Moe
        )
    {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "HF architecture {} does not define supported q_norm/k_norm tensor names",
                metadata.architecture.as_str()
            ),
        });
    }
    Ok(())
}

fn validate_moe_metadata(metadata: &HfModelMetadata) -> Result<()> {
    let moe_layers = metadata
        .mlp_layer_types
        .iter()
        .filter(|kind| **kind == HfMlpLayerKind::SparseMoe)
        .count();
    if moe_layers == 0 {
        return Ok(());
    }
    let Some(num_experts) = metadata.num_experts else {
        return Err(NervaError::InvalidArgument {
            reason: "HF MoE metadata is missing num_experts".to_string(),
        });
    };
    let Some(num_experts_per_tok) = metadata.num_experts_per_tok else {
        return Err(NervaError::InvalidArgument {
            reason: "HF MoE metadata is missing num_experts_per_tok".to_string(),
        });
    };
    let Some(moe_intermediate_size) = metadata.moe_intermediate_size else {
        return Err(NervaError::InvalidArgument {
            reason: "HF MoE metadata is missing moe_intermediate_size".to_string(),
        });
    };
    let shared_expert_intermediate_size = metadata.shared_expert_intermediate_size.unwrap_or(0);
    if num_experts == 0 || num_experts_per_tok == 0 || moe_intermediate_size == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "HF MoE expert counts and dimensions must be non-zero".to_string(),
        });
    }
    if num_experts_per_tok > num_experts {
        return Err(NervaError::InvalidArgument {
            reason: "HF MoE num_experts_per_tok cannot exceed num_experts".to_string(),
        });
    }
    if !metadata.architecture.is_deepseek() && num_experts > EXACT_RUNTIME_MOE_EXPERTS_MAX {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "HF MoE num_experts {num_experts} exceeds exact runtime limit {EXACT_RUNTIME_MOE_EXPERTS_MAX}"
            ),
        });
    }
    if num_experts_per_tok > EXACT_RUNTIME_MOE_TOP_K_MAX {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "HF MoE num_experts_per_tok {num_experts_per_tok} exceeds exact runtime top-k limit {EXACT_RUNTIME_MOE_TOP_K_MAX}"
            ),
        });
    }
    if shared_expert_intermediate_size > metadata.intermediate_size {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "HF MoE shared_expert_intermediate_size {shared_expert_intermediate_size} exceeds exact runtime scratch intermediate {}",
                metadata.intermediate_size
            ),
        });
    }
    Ok(())
}

fn validate_mlp_activation(metadata: &HfModelMetadata) -> Result<()> {
    match metadata.hidden_act.as_deref().unwrap_or("silu") {
        "silu" => Ok(()),
        activation => Err(NervaError::InvalidArgument {
            reason: format!(
                "HF hidden activation {activation} is not supported by the exact runtime contract"
            ),
        }),
    }
}

fn validate_mlp_bias(metadata: &HfModelMetadata) -> Result<()> {
    if metadata.mlp_bias {
        return Err(NervaError::InvalidArgument {
            reason: "HF MLP bias is not supported by the exact runtime contract".to_string(),
        });
    }
    Ok(())
}
