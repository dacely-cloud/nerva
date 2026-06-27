use nerva_core::types::error::{NervaError, Result};

use crate::hf::architecture::HfArchitectureKind;
use crate::hf::metadata::HfModelMetadata;

pub(crate) fn validate_exact_runtime_contract(metadata: &HfModelMetadata) -> Result<()> {
    validate_supported_architecture(metadata.architecture)?;
    validate_qk_norm(metadata)?;
    validate_mlp_activation(metadata)?;
    validate_mlp_bias(metadata)
}

fn validate_supported_architecture(architecture: HfArchitectureKind) -> Result<()> {
    match architecture {
        HfArchitectureKind::Llama
        | HfArchitectureKind::Mistral
        | HfArchitectureKind::Qwen2
        | HfArchitectureKind::Qwen3 => Ok(()),
        HfArchitectureKind::Gemma | HfArchitectureKind::Unknown => {
            Err(NervaError::InvalidArgument {
                reason: format!(
                    "HF architecture {} is not supported by the exact runtime contract",
                    architecture.as_str()
                ),
            })
        }
    }
}

fn validate_qk_norm(metadata: &HfModelMetadata) -> Result<()> {
    if metadata.qk_norm
        && !matches!(
            metadata.architecture,
            HfArchitectureKind::Qwen2 | HfArchitectureKind::Qwen3
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
