use nerva_core::{DType, NervaError, Result};

use crate::common::{
    TransformerBlockShape, dtype_to_str, json_opt_dtype, json_opt_f32, json_opt_usize,
    optional_bool, optional_f32, optional_first_string, optional_string, optional_usize,
    required_usize,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfArchitectureKind {
    Llama,
    Mistral,
    Gemma,
    Qwen2,
    Unknown,
}

impl HfArchitectureKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Gemma => "gemma",
            Self::Qwen2 => "qwen2",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfModelMetadata {
    pub architecture: HfArchitectureKind,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: Option<usize>,
    pub rope_theta: Option<f32>,
    pub rms_norm_eps: Option<f32>,
    pub tie_word_embeddings: bool,
    pub torch_dtype: Option<DType>,
}

impl HfModelMetadata {
    pub fn block_shape(&self) -> TransformerBlockShape {
        TransformerBlockShape::new(
            self.hidden_size,
            self.num_attention_heads,
            self.intermediate_size,
        )
    }

    pub const fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub const fn kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"architecture\":\"{}\",\"hidden_size\":{},\"num_hidden_layers\":{},\"num_attention_heads\":{},\"num_key_value_heads\":{},\"head_dim\":{},\"kv_groups\":{},\"intermediate_size\":{},\"vocab_size\":{},\"max_position_embeddings\":{},\"rope_theta\":{},\"rms_norm_eps\":{},\"tie_word_embeddings\":{},\"torch_dtype\":{}}}",
            self.architecture.as_str(),
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim(),
            self.kv_groups(),
            self.intermediate_size,
            self.vocab_size,
            json_opt_usize(self.max_position_embeddings),
            json_opt_f32(self.rope_theta),
            json_opt_f32(self.rms_norm_eps),
            self.tie_word_embeddings,
            json_opt_dtype(self.torch_dtype),
        )
    }
}

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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfMetadataProbeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfMetadataProbeSummary {
    pub status: HfMetadataProbeStatus,
    pub metadata: HfModelMetadata,
    pub metadata_hash: u64,
}

impl HfMetadataProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            HfMetadataProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"metadata\":{},\"metadata_hash\":{}}}",
            status,
            self.metadata.to_json(),
            self.metadata_hash,
        )
    }
}

pub fn hf_metadata_probe() -> Result<HfMetadataProbeSummary> {
    let config = r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 0.000001,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16"
    }"#;
    let metadata = parse_hf_config_metadata(config)?;
    Ok(HfMetadataProbeSummary {
        metadata_hash: hash_metadata(&metadata),
        status: HfMetadataProbeStatus::Ok,
        metadata,
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

pub(crate) fn architecture_kind_from_str(value: &str) -> HfArchitectureKind {
    let lower = value.to_ascii_lowercase();
    if lower.contains("llama") {
        HfArchitectureKind::Llama
    } else if lower.contains("mistral") {
        HfArchitectureKind::Mistral
    } else if lower.contains("gemma") {
        HfArchitectureKind::Gemma
    } else if lower.contains("qwen2") {
        HfArchitectureKind::Qwen2
    } else {
        HfArchitectureKind::Unknown
    }
}

pub(crate) fn validate_hf_metadata(
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
) -> Result<()> {
    if hidden_size == 0
        || num_hidden_layers == 0
        || num_attention_heads == 0
        || num_key_value_heads == 0
        || intermediate_size == 0
        || vocab_size == 0
    {
        return Err(NervaError::InvalidArgument {
            reason: "HF model metadata dimensions must be non-zero".to_string(),
        });
    }
    if !hidden_size.is_multiple_of(num_attention_heads) {
        return Err(NervaError::InvalidArgument {
            reason: "HF hidden size must be divisible by attention head count".to_string(),
        });
    }
    if num_key_value_heads > num_attention_heads {
        return Err(NervaError::InvalidArgument {
            reason: "HF KV head count cannot exceed attention head count".to_string(),
        });
    }
    if !num_attention_heads.is_multiple_of(num_key_value_heads) {
        return Err(NervaError::InvalidArgument {
            reason: "HF attention head count must be divisible by KV head count".to_string(),
        });
    }
    Ok(())
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
pub(crate) fn hash_metadata(metadata: &HfModelMetadata) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in [
        metadata.hidden_size as u64,
        metadata.num_hidden_layers as u64,
        metadata.num_attention_heads as u64,
        metadata.num_key_value_heads as u64,
        metadata.intermediate_size as u64,
        metadata.vocab_size as u64,
        metadata.max_position_embeddings.unwrap_or_default() as u64,
        metadata.head_dim() as u64,
        metadata.kv_groups() as u64,
        u64::from(metadata.tie_word_embeddings),
    ] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for byte in metadata.architecture.as_str().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if let Some(dtype) = metadata.torch_dtype {
        for byte in dtype_to_str(dtype).as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}
