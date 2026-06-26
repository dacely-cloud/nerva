use nerva_core::types::DType;

use crate::common::dtype::json_opt_dtype;
use crate::common::json::{json_opt_f32, json_opt_usize};
use crate::common::shape::TransformerBlockShape;
use crate::hf::architecture::HfArchitectureKind;

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
