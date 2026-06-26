use nerva_core::types::Result;

use crate::hf::hash::hash_metadata;
use crate::hf::metadata::HfModelMetadata;
use crate::hf::parser::parse_hf_config_metadata;

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
