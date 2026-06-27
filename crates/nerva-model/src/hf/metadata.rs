use nerva_core::types::dtype::DType;

use crate::common::dtype::json_opt_dtype;
use crate::common::json::format::{json_opt_f32, json_opt_str, json_opt_u32, json_opt_usize};
use crate::common::shape::TransformerBlockShape;
use crate::hf::architecture::HfArchitectureKind;

#[derive(Clone, Debug, PartialEq)]
pub struct HfModelMetadata {
    pub architecture: HfArchitectureKind,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: Option<usize>,
    pub rope_theta: Option<f32>,
    pub rms_norm_eps: Option<f32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub tie_word_embeddings: bool,
    pub hidden_act: Option<String>,
    pub attention_bias: bool,
    pub qk_norm: bool,
    pub mlp_bias: bool,
    pub torch_dtype: Option<DType>,
}

impl HfModelMetadata {
    pub fn block_shape(&self) -> TransformerBlockShape {
        TransformerBlockShape::new_with_kv_heads_and_head_dim(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.intermediate_size,
        )
    }

    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub const fn attention_hidden(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub const fn kv_hidden(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub const fn kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"architecture\":\"{}\",\"hidden_size\":{},\"num_hidden_layers\":{},\"num_attention_heads\":{},\"num_key_value_heads\":{},\"head_dim\":{},\"attention_hidden_size\":{},\"kv_hidden_size\":{},\"kv_groups\":{},\"intermediate_size\":{},\"vocab_size\":{},\"max_position_embeddings\":{},\"rope_theta\":{},\"rms_norm_eps\":{},\"bos_token_id\":{},\"eos_token_id\":{},\"tie_word_embeddings\":{},\"hidden_act\":{},\"attention_bias\":{},\"qk_norm\":{},\"mlp_bias\":{},\"torch_dtype\":{}}}",
            self.architecture.as_str(),
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim(),
            self.attention_hidden(),
            self.kv_hidden(),
            self.kv_groups(),
            self.intermediate_size,
            self.vocab_size,
            json_opt_usize(self.max_position_embeddings),
            json_opt_f32(self.rope_theta),
            json_opt_f32(self.rms_norm_eps),
            json_opt_u32(self.bos_token_id),
            json_opt_u32(self.eos_token_id),
            self.tie_word_embeddings,
            json_opt_str(self.hidden_act.as_deref()),
            self.attention_bias,
            self.qk_norm,
            self.mlp_bias,
            json_opt_dtype(self.torch_dtype),
        )
    }
}
