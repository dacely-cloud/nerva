use nerva_core::types::{DType, NervaError, Result};

use crate::common::json::{
    optional_bool, optional_f32, optional_first_string, optional_string, optional_usize,
    required_usize,
};
use crate::hf::architecture::{HfArchitectureKind, architecture_kind_from_str};
use crate::hf::metadata::HfModelMetadata;
use crate::hf::validate::validate_hf_metadata;

pub fn parse_hf_config_metadata(config_json: &str) -> Result<HfModelMetadata> {
    let architecture = architecture_from_config(config_json)?;
    let hidden_size = required_usize(config_json, "hidden_size")?;
    let num_hidden_layers = required_usize(config_json, "num_hidden_layers")?;
    let num_attention_heads = required_usize(config_json, "num_attention_heads")?;
    let num_key_value_heads =
        optional_usize(config_json, "num_key_value_heads")?.unwrap_or(num_attention_heads);
    let intermediate_size = required_usize(config_json, "intermediate_size")?;
    let vocab_size = required_usize(config_json, "vocab_size")?;
    let max_position_embeddings = optional_usize(config_json, "max_position_embeddings")?;
    let rope_theta = optional_f32(config_json, "rope_theta")?;
    let rms_norm_eps = match optional_f32(config_json, "rms_norm_eps")? {
        Some(value) => Some(value),
        None => optional_f32(config_json, "layer_norm_eps")?,
    };
    let tie_word_embeddings = optional_bool(config_json, "tie_word_embeddings")?.unwrap_or(false);
    let torch_dtype = optional_string(config_json, "torch_dtype")?
        .as_deref()
        .map(dtype_from_hf_string)
        .transpose()?;

    validate_hf_metadata(
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        intermediate_size,
        vocab_size,
    )?;

    Ok(HfModelMetadata {
        architecture,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        intermediate_size,
        vocab_size,
        max_position_embeddings,
        rope_theta,
        rms_norm_eps,
        tie_word_embeddings,
        torch_dtype,
    })
}

pub(crate) fn architecture_from_config(config_json: &str) -> Result<HfArchitectureKind> {
    if let Some(architecture) = optional_first_string(config_json, "architectures")? {
        return Ok(architecture_kind_from_str(&architecture));
    }
    if let Some(model_type) = optional_string(config_json, "model_type")? {
        return Ok(architecture_kind_from_str(&model_type));
    }
    Ok(HfArchitectureKind::Unknown)
}

pub(crate) fn dtype_from_hf_string(value: &str) -> Result<DType> {
    match value.to_ascii_lowercase().as_str() {
        "float16" | "fp16" | "f16" => Ok(DType::F16),
        "bfloat16" | "bf16" => Ok(DType::BF16),
        "float32" | "fp32" | "f32" => Ok(DType::F32),
        other => Err(NervaError::InvalidArgument {
            reason: format!("unsupported HF torch_dtype {other}"),
        }),
    }
}
