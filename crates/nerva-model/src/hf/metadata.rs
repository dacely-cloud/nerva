use nerva_core::types::dtype::DType;

use crate::common::dtype::json_opt_dtype;
use crate::common::json::format::{json_opt_f32, json_opt_str, json_opt_u32, json_opt_usize};
use crate::common::shape::TransformerBlockShape;
use crate::hf::architecture::HfArchitectureKind;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfAttentionLayerKind {
    Full,
    Linear,
}

impl HfAttentionLayerKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full_attention",
            Self::Linear => "linear_attention",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfMlpLayerKind {
    Dense,
    SparseMoe,
}

impl HfMlpLayerKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::SparseMoe => "sparse_moe",
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
    pub attention_qkv_bias: bool,
    pub attention_output_bias: bool,
    pub qk_norm: bool,
    pub mlp_bias: bool,
    pub linear_conv_kernel_dim: Option<usize>,
    pub linear_key_head_dim: Option<usize>,
    pub linear_value_head_dim: Option<usize>,
    pub linear_num_key_heads: Option<usize>,
    pub linear_num_value_heads: Option<usize>,
    pub attention_layer_types: Vec<HfAttentionLayerKind>,
    pub mlp_layer_types: Vec<HfMlpLayerKind>,
    pub moe_intermediate_size: Option<usize>,
    pub shared_expert_intermediate_size: Option<usize>,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub decoder_sparse_step: Option<usize>,
    pub norm_topk_prob: bool,
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

    pub fn has_linear_attention_layers(&self) -> bool {
        self.attention_layer_types
            .iter()
            .any(|kind| *kind == HfAttentionLayerKind::Linear)
    }

    pub fn has_moe_layers(&self) -> bool {
        self.mlp_layer_types
            .iter()
            .any(|kind| *kind == HfMlpLayerKind::SparseMoe)
    }

    pub fn to_json(&self) -> String {
        let linear_layers = self
            .attention_layer_types
            .iter()
            .filter(|kind| **kind == HfAttentionLayerKind::Linear)
            .count();
        let full_layers = self
            .attention_layer_types
            .len()
            .saturating_sub(linear_layers);
        let moe_layers = self
            .mlp_layer_types
            .iter()
            .filter(|kind| **kind == HfMlpLayerKind::SparseMoe)
            .count();
        let dense_mlp_layers = self.mlp_layer_types.len().saturating_sub(moe_layers);
        format!(
            "{{\"architecture\":\"{}\",\"hidden_size\":{},\"num_hidden_layers\":{},\"num_attention_heads\":{},\"num_key_value_heads\":{},\"head_dim\":{},\"attention_hidden_size\":{},\"kv_hidden_size\":{},\"kv_groups\":{},\"intermediate_size\":{},\"moe_intermediate_size\":{},\"shared_expert_intermediate_size\":{},\"num_experts\":{},\"num_experts_per_tok\":{},\"decoder_sparse_step\":{},\"norm_topk_prob\":{},\"vocab_size\":{},\"max_position_embeddings\":{},\"rope_theta\":{},\"rms_norm_eps\":{},\"bos_token_id\":{},\"eos_token_id\":{},\"tie_word_embeddings\":{},\"hidden_act\":{},\"attention_bias\":{},\"attention_qkv_bias\":{},\"attention_output_bias\":{},\"qk_norm\":{},\"mlp_bias\":{},\"linear_conv_kernel_dim\":{},\"linear_key_head_dim\":{},\"linear_value_head_dim\":{},\"linear_num_key_heads\":{},\"linear_num_value_heads\":{},\"attention_full_layers\":{},\"attention_linear_layers\":{},\"mlp_dense_layers\":{},\"mlp_moe_layers\":{},\"torch_dtype\":{}}}",
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
            json_opt_usize(self.moe_intermediate_size),
            json_opt_usize(self.shared_expert_intermediate_size),
            json_opt_usize(self.num_experts),
            json_opt_usize(self.num_experts_per_tok),
            json_opt_usize(self.decoder_sparse_step),
            self.norm_topk_prob,
            self.vocab_size,
            json_opt_usize(self.max_position_embeddings),
            json_opt_f32(self.rope_theta),
            json_opt_f32(self.rms_norm_eps),
            json_opt_u32(self.bos_token_id),
            json_opt_u32(self.eos_token_id),
            self.tie_word_embeddings,
            json_opt_str(self.hidden_act.as_deref()),
            self.attention_bias,
            self.attention_qkv_bias,
            self.attention_output_bias,
            self.qk_norm,
            self.mlp_bias,
            json_opt_usize(self.linear_conv_kernel_dim),
            json_opt_usize(self.linear_key_head_dim),
            json_opt_usize(self.linear_value_head_dim),
            json_opt_usize(self.linear_num_key_heads),
            json_opt_usize(self.linear_num_value_heads),
            full_layers,
            linear_layers,
            dense_mlp_layers,
            moe_layers,
            json_opt_dtype(self.torch_dtype),
        )
    }
}
