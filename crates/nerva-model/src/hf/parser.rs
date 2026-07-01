use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

use crate::common::json::fields::{
    optional_bool, optional_f32, optional_first_string, optional_nonnegative_f32,
    optional_object_f32, optional_object_json, optional_object_string, optional_string,
    optional_u32_or_first, optional_usize,
};
use crate::hf::architecture::{architecture_kind_from_str, HfArchitectureKind};
use crate::hf::metadata::{
    HfAttentionLayerKind, HfMlpLayerKind, HfModelMetadata, HfRopeScalingMetadata,
};
use crate::hf::validate::validate_hf_metadata;

pub fn parse_hf_config_metadata(config_json: &str) -> Result<HfModelMetadata> {
    let architecture = architecture_from_config(config_json)?;
    validate_supported_rope_config(config_json, architecture)?;
    let decoder_config_json = decoder_config_json(config_json, architecture)?;
    let hidden_size = required_model_usize(config_json, decoder_config_json, "hidden_size")?;
    let num_hidden_layers =
        required_model_usize(config_json, decoder_config_json, "num_hidden_layers")?;
    let num_attention_heads =
        required_model_usize(config_json, decoder_config_json, "num_attention_heads")?;
    let num_key_value_heads =
        optional_model_usize(config_json, decoder_config_json, "num_key_value_heads")?
            .unwrap_or(num_attention_heads);
    let head_dim = parse_head_dim(
        decoder_config_json,
        architecture,
        hidden_size,
        num_attention_heads,
    )?;
    let explicit_intermediate_size =
        optional_model_usize(config_json, decoder_config_json, "intermediate_size")?;
    let vocab_size = required_model_usize(config_json, decoder_config_json, "vocab_size")?;
    let max_position_embeddings =
        optional_model_usize(config_json, decoder_config_json, "max_position_embeddings")?;
    let sliding_window = optional_model_usize(config_json, decoder_config_json, "sliding_window")?;
    let rope_theta = parse_rope_theta(config_json, decoder_config_json)?;
    let rope_scaling = parse_rope_scaling(config_json, decoder_config_json, architecture)?;
    let rms_norm_eps = parse_rms_norm_eps(config_json, decoder_config_json)?;
    let bos_token_id =
        optional_model_u32_or_first(config_json, decoder_config_json, "bos_token_id")?;
    let eos_token_id =
        optional_model_u32_or_first(config_json, decoder_config_json, "eos_token_id")?;
    let tie_word_embeddings =
        optional_model_bool(config_json, decoder_config_json, "tie_word_embeddings")?
            .unwrap_or(false);
    let hidden_act = parse_hidden_act(config_json, decoder_config_json)?;
    let attention_bias = parse_attention_bias(config_json, decoder_config_json, architecture)?;
    let qk_norm = parse_qk_norm(config_json, decoder_config_json, architecture)?;
    let mlp_bias =
        optional_model_bool(config_json, decoder_config_json, "mlp_bias")?.unwrap_or(false);
    let linear_attention =
        parse_linear_attention_config(config_json, decoder_config_json, architecture)?;
    let attention_layer_types = parse_attention_layer_types(
        config_json,
        decoder_config_json,
        architecture,
        num_hidden_layers,
    )?;
    let deepseek_config = parse_deepseek_config(config_json, decoder_config_json, architecture)?;
    let moe_config = parse_moe_config(
        config_json,
        decoder_config_json,
        architecture,
        explicit_intermediate_size,
        num_hidden_layers,
    )?;
    let intermediate_size = resolve_intermediate_size(explicit_intermediate_size, &moe_config)?;
    let torch_dtype = parse_torch_dtype(config_json, decoder_config_json)?;

    validate_hf_metadata(
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        intermediate_size,
        vocab_size,
    )?;

    Ok(HfModelMetadata {
        architecture,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_position_embeddings,
        sliding_window,
        rope_theta,
        rope_scaling,
        compress_rope_theta: deepseek_config.compress_rope_theta,
        rms_norm_eps,
        bos_token_id,
        eos_token_id,
        tie_word_embeddings,
        hidden_act,
        attention_bias: attention_bias.any(),
        attention_qkv_bias: attention_bias.qkv,
        attention_output_bias: attention_bias.output,
        qk_norm,
        mlp_bias,
        linear_conv_kernel_dim: linear_attention.linear_conv_kernel_dim,
        linear_key_head_dim: linear_attention.linear_key_head_dim,
        linear_value_head_dim: linear_attention.linear_value_head_dim,
        linear_num_key_heads: linear_attention.linear_num_key_heads,
        linear_num_value_heads: linear_attention.linear_num_value_heads,
        attention_layer_types,
        mlp_layer_types: moe_config.mlp_layer_types,
        moe_intermediate_size: moe_config.moe_intermediate_size,
        shared_expert_intermediate_size: moe_config.shared_expert_intermediate_size,
        num_experts: moe_config.num_experts,
        num_experts_per_tok: moe_config.num_experts_per_tok,
        decoder_sparse_step: moe_config.decoder_sparse_step,
        norm_topk_prob: moe_config.norm_topk_prob,
        moe_first_k_dense_replace: moe_config.first_k_dense_replace,
        moe_layer_freq: moe_config.moe_layer_freq,
        num_expert_groups: moe_config.num_expert_groups,
        topk_group: moe_config.topk_group,
        topk_method: moe_config.topk_method,
        scoring_func: moe_config.scoring_func,
        routed_scaling_factor: moe_config.routed_scaling_factor,
        q_lora_rank: deepseek_config.q_lora_rank,
        kv_lora_rank: deepseek_config.kv_lora_rank,
        o_lora_rank: deepseek_config.o_lora_rank,
        o_groups: deepseek_config.o_groups,
        qk_nope_head_dim: deepseek_config.qk_nope_head_dim,
        qk_rope_head_dim: deepseek_config.qk_rope_head_dim,
        v_head_dim: deepseek_config.v_head_dim,
        index_topk: deepseek_config.index_topk,
        index_topk_freq: deepseek_config.index_topk_freq,
        index_skip_topk_offset: deepseek_config.index_skip_topk_offset,
        index_topk_pattern: deepseek_config.index_topk_pattern,
        index_n_heads: deepseek_config.index_n_heads,
        index_head_dim: deepseek_config.index_head_dim,
        compress_ratios: deepseek_config.compress_ratios,
        hc_mult: deepseek_config.hc_mult,
        hc_sinkhorn_iters: deepseek_config.hc_sinkhorn_iters,
        hc_eps: deepseek_config.hc_eps,
        num_nextn_predict_layers: deepseek_config.num_nextn_predict_layers,
        num_hash_layers: deepseek_config.num_hash_layers,
        swiglu_limit: deepseek_config.swiglu_limit,
        expert_dtype: deepseek_config.expert_dtype,
        torch_dtype,
    })
}

struct ParsedMoeConfig {
    mlp_layer_types: Vec<HfMlpLayerKind>,
    moe_intermediate_size: Option<usize>,
    shared_expert_intermediate_size: Option<usize>,
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    decoder_sparse_step: Option<usize>,
    norm_topk_prob: bool,
    first_k_dense_replace: Option<usize>,
    moe_layer_freq: Option<usize>,
    num_expert_groups: Option<usize>,
    topk_group: Option<usize>,
    topk_method: Option<String>,
    scoring_func: Option<String>,
    routed_scaling_factor: Option<f32>,
}

struct ParsedDeepSeekConfig {
    q_lora_rank: Option<usize>,
    kv_lora_rank: Option<usize>,
    o_lora_rank: Option<usize>,
    o_groups: Option<usize>,
    qk_nope_head_dim: Option<usize>,
    qk_rope_head_dim: Option<usize>,
    v_head_dim: Option<usize>,
    index_topk: Option<usize>,
    index_topk_freq: Option<usize>,
    index_skip_topk_offset: Option<usize>,
    index_topk_pattern: Vec<String>,
    index_n_heads: Option<usize>,
    index_head_dim: Option<usize>,
    compress_ratios: Vec<usize>,
    hc_mult: Option<usize>,
    hc_sinkhorn_iters: Option<usize>,
    hc_eps: Option<f32>,
    num_nextn_predict_layers: Option<usize>,
    num_hash_layers: Option<usize>,
    compress_rope_theta: Option<f32>,
    swiglu_limit: Option<f32>,
    expert_dtype: Option<String>,
}

struct ParsedLinearAttentionConfig {
    linear_conv_kernel_dim: Option<usize>,
    linear_key_head_dim: Option<usize>,
    linear_value_head_dim: Option<usize>,
    linear_num_key_heads: Option<usize>,
    linear_num_value_heads: Option<usize>,
}

struct ParsedAttentionBiasConfig {
    qkv: bool,
    output: bool,
}

impl ParsedAttentionBiasConfig {
    const fn any(&self) -> bool {
        self.qkv || self.output
    }
}

fn decoder_config_json<'a>(
    config_json: &'a str,
    architecture: HfArchitectureKind,
) -> Result<&'a str> {
    match architecture {
        HfArchitectureKind::Qwen35 | HfArchitectureKind::Qwen35Moe => {
            Ok(optional_object_json(config_json, "text_config")?.unwrap_or(config_json))
        }
        _ => Ok(config_json),
    }
}

fn required_model_usize(root_json: &str, decoder_json: &str, key: &'static str) -> Result<usize> {
    optional_model_usize(root_json, decoder_json, key)?.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("HF config is missing required field {key}"),
    })
}

fn optional_model_usize(
    root_json: &str,
    decoder_json: &str,
    key: &'static str,
) -> Result<Option<usize>> {
    match optional_usize(decoder_json, key)? {
        Some(value) => Ok(Some(value)),
        None if !std::ptr::eq(root_json, decoder_json) => optional_usize(root_json, key),
        None => Ok(None),
    }
}

fn optional_model_u32_or_first(
    root_json: &str,
    decoder_json: &str,
    key: &'static str,
) -> Result<Option<u32>> {
    match optional_u32_or_first(decoder_json, key)? {
        Some(value) => Ok(Some(value)),
        None if !std::ptr::eq(root_json, decoder_json) => optional_u32_or_first(root_json, key),
        None => Ok(None),
    }
}

fn optional_model_bool(
    root_json: &str,
    decoder_json: &str,
    key: &'static str,
) -> Result<Option<bool>> {
    match optional_bool(decoder_json, key)? {
        Some(value) => Ok(Some(value)),
        None if !std::ptr::eq(root_json, decoder_json) => optional_bool(root_json, key),
        None => Ok(None),
    }
}

fn optional_model_f32(
    root_json: &str,
    decoder_json: &str,
    key: &'static str,
) -> Result<Option<f32>> {
    match optional_f32(decoder_json, key)? {
        Some(value) => Ok(Some(value)),
        None if !std::ptr::eq(root_json, decoder_json) => optional_f32(root_json, key),
        None => Ok(None),
    }
}

fn optional_model_string(
    root_json: &str,
    decoder_json: &str,
    key: &'static str,
) -> Result<Option<String>> {
    match optional_string(decoder_json, key)? {
        Some(value) => Ok(Some(value)),
        None if !std::ptr::eq(root_json, decoder_json) => optional_string(root_json, key),
        None => Ok(None),
    }
}

fn parse_rms_norm_eps(root_json: &str, decoder_json: &str) -> Result<Option<f32>> {
    match optional_f32(decoder_json, "rms_norm_eps")? {
        Some(value) => Ok(Some(value)),
        None => match optional_f32(decoder_json, "layer_norm_eps")? {
            Some(value) => Ok(Some(value)),
            None if !std::ptr::eq(root_json, decoder_json) => {
                match optional_f32(root_json, "rms_norm_eps")? {
                    Some(value) => Ok(Some(value)),
                    None => optional_f32(root_json, "layer_norm_eps"),
                }
            }
            None => Ok(None),
        },
    }
}

fn parse_head_dim(
    config_json: &str,
    architecture: HfArchitectureKind,
    hidden_size: usize,
    num_attention_heads: usize,
) -> Result<usize> {
    if let Some(head_dim) = optional_usize(config_json, "head_dim")? {
        return Ok(head_dim);
    }
    if architecture.is_deepseek() {
        let qk_nope_head_dim = optional_usize(config_json, "qk_nope_head_dim")?;
        let qk_rope_head_dim = optional_usize(config_json, "qk_rope_head_dim")?;
        if let (Some(nope), Some(rope)) = (qk_nope_head_dim, qk_rope_head_dim) {
            return nope
                .checked_add(rope)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes: nope,
                    reason: "DeepSeek qk head dimension overflow".to_string(),
                });
        }
    }
    if num_attention_heads == 0 || !hidden_size.is_multiple_of(num_attention_heads) {
        return Err(NervaError::InvalidArgument {
            reason: "HF hidden size must be divisible by attention head count".to_string(),
        });
    }
    Ok(hidden_size / num_attention_heads)
}

fn parse_rope_theta(root_json: &str, decoder_json: &str) -> Result<Option<f32>> {
    if let Some(theta) = optional_f32(decoder_json, "rope_theta")? {
        return Ok(Some(theta));
    }
    if let Some(theta) = optional_object_f32(decoder_json, "rope_parameters", "rope_theta")? {
        return Ok(Some(theta));
    }
    if let Some(theta) = optional_object_f32(decoder_json, "rope_scaling", "rope_theta")? {
        return Ok(Some(theta));
    }
    if !std::ptr::eq(root_json, decoder_json) {
        if let Some(theta) = optional_f32(root_json, "rope_theta")? {
            return Ok(Some(theta));
        }
        if let Some(theta) = optional_object_f32(root_json, "rope_parameters", "rope_theta")? {
            return Ok(Some(theta));
        }
        return optional_object_f32(root_json, "rope_scaling", "rope_theta");
    }
    Ok(None)
}

fn parse_rope_scaling(
    root_json: &str,
    decoder_json: &str,
    architecture: HfArchitectureKind,
) -> Result<Option<HfRopeScalingMetadata>> {
    let Some((object_json, object_key)) = rope_config_object(root_json, decoder_json)? else {
        return Ok(None);
    };
    let modern = optional_string(object_json, "rope_type")?;
    let legacy = optional_string(object_json, "type")?;
    let Some(raw_type) = modern.or(legacy) else {
        return Ok(None);
    };
    if raw_type == "default" {
        return Ok(None);
    }
    if !architecture.is_deepseek() {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "unsupported HF {object_key} rope_type {raw_type} for exact runtime path"
            ),
        });
    }

    let apply_yarn_scaling = optional_bool(object_json, "apply_yarn_scaling")?.unwrap_or(true);
    let rope_type = match raw_type.as_str() {
        "yarn" | "deepseek_yarn" if apply_yarn_scaling => "deepseek_yarn",
        "yarn" | "deepseek_yarn" | "deepseek_llama_scaling" => "deepseek_llama_scaling",
        unsupported => {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "unsupported DeepSeek {object_key} rope_type {unsupported} for exact runtime path"
                ),
            });
        }
    };

    Ok(Some(HfRopeScalingMetadata {
        rope_type: rope_type.to_string(),
        factor: optional_f32(object_json, "factor")?,
        original_max_position_embeddings: optional_usize(
            object_json,
            "original_max_position_embeddings",
        )?,
        extrapolation_factor: optional_f32(object_json, "extrapolation_factor")?,
        attn_factor: optional_f32(object_json, "attn_factor")?,
        beta_fast: optional_f32(object_json, "beta_fast")?,
        beta_slow: optional_f32(object_json, "beta_slow")?,
        mscale: optional_nonnegative_f32(object_json, "mscale")?,
        mscale_all_dim: optional_nonnegative_f32(object_json, "mscale_all_dim")?,
    }))
}

fn rope_config_object<'a>(
    root_json: &'a str,
    decoder_json: &'a str,
) -> Result<Option<(&'a str, &'static str)>> {
    if let Some(object_json) = optional_object_json(decoder_json, "rope_parameters")? {
        return Ok(Some((object_json, "rope_parameters")));
    }
    if let Some(object_json) = optional_object_json(decoder_json, "rope_scaling")? {
        return Ok(Some((object_json, "rope_scaling")));
    }
    if !std::ptr::eq(root_json, decoder_json) {
        if let Some(object_json) = optional_object_json(root_json, "rope_parameters")? {
            return Ok(Some((object_json, "rope_parameters")));
        }
        if let Some(object_json) = optional_object_json(root_json, "rope_scaling")? {
            return Ok(Some((object_json, "rope_scaling")));
        }
    }
    Ok(None)
}

fn validate_supported_rope_config(
    config_json: &str,
    architecture: HfArchitectureKind,
) -> Result<()> {
    if architecture.is_deepseek() {
        return Ok(());
    }
    validate_default_rope_object(config_json, "rope_parameters")?;
    validate_default_rope_object(config_json, "rope_scaling")
}

fn validate_default_rope_object(config_json: &str, key: &'static str) -> Result<()> {
    let modern = optional_object_string(config_json, key, "rope_type")?;
    let legacy = optional_object_string(config_json, key, "type")?;
    match modern.as_deref().or(legacy.as_deref()) {
        None | Some("default") => Ok(()),
        Some(rope_type) => Err(NervaError::InvalidArgument {
            reason: format!("unsupported HF {key} rope_type {rope_type} for exact runtime path"),
        }),
    }
}

fn parse_hidden_act(root_json: &str, decoder_json: &str) -> Result<Option<String>> {
    if let Some(value) = optional_model_string(root_json, decoder_json, "hidden_act")? {
        return Ok(Some(value));
    }
    if let Some(value) = optional_model_string(root_json, decoder_json, "hidden_activation")? {
        return Ok(Some(value));
    }
    optional_model_string(root_json, decoder_json, "activation_function")
}

fn parse_attention_bias(
    root_json: &str,
    decoder_json: &str,
    architecture: HfArchitectureKind,
) -> Result<ParsedAttentionBiasConfig> {
    let attention_bias = optional_model_bool(root_json, decoder_json, "attention_bias")?;
    let qkv_bias = optional_model_bool(root_json, decoder_json, "qkv_bias")?;
    let default_qkv_bias = matches!(architecture, HfArchitectureKind::Qwen2Moe)
        && attention_bias.is_none()
        && qkv_bias.is_none();
    let attention_bias = attention_bias.unwrap_or(false);
    Ok(ParsedAttentionBiasConfig {
        qkv: attention_bias || qkv_bias.unwrap_or(default_qkv_bias),
        output: attention_bias,
    })
}

fn parse_qk_norm(
    root_json: &str,
    decoder_json: &str,
    architecture: HfArchitectureKind,
) -> Result<bool> {
    match optional_model_bool(root_json, decoder_json, "qk_norm")? {
        Some(value) => Ok(value),
        None => match optional_model_bool(root_json, decoder_json, "use_qk_norm")? {
            Some(value) => Ok(value),
            None => Ok(matches!(
                architecture,
                HfArchitectureKind::Qwen3
                    | HfArchitectureKind::Qwen3Moe
                    | HfArchitectureKind::Qwen35
                    | HfArchitectureKind::Qwen35Moe
            )),
        },
    }
}

fn parse_moe_config(
    root_json: &str,
    decoder_json: &str,
    architecture: HfArchitectureKind,
    intermediate_size: Option<usize>,
    num_hidden_layers: usize,
) -> Result<ParsedMoeConfig> {
    let num_experts = match optional_model_usize(root_json, decoder_json, "num_experts")? {
        Some(value) => Some(value),
        None => match optional_model_usize(root_json, decoder_json, "num_local_experts")? {
            Some(value) => Some(value),
            None => optional_model_usize(root_json, decoder_json, "n_routed_experts")?,
        },
    };
    let num_experts_per_tok = optional_model_usize(root_json, decoder_json, "num_experts_per_tok")?;
    let moe_intermediate_size =
        match optional_model_usize(root_json, decoder_json, "moe_intermediate_size")? {
            Some(value) => Some(value),
            None if architecture == HfArchitectureKind::MixtralMoe => intermediate_size,
            None => None,
        };
    let n_shared_experts = optional_model_usize(root_json, decoder_json, "n_shared_experts")?;
    let shared_expert_intermediate_size =
        match optional_model_usize(root_json, decoder_json, "shared_expert_intermediate_size")? {
            Some(value) => value,
            None => match (n_shared_experts, moe_intermediate_size) {
                (Some(experts), Some(intermediate)) => experts
                    .checked_mul(intermediate)
                    .ok_or_else(|| NervaError::AllocationFailed {
                        bytes: intermediate,
                        reason: "HF MoE shared expert intermediate overflow".to_string(),
                    })?,
                _ => 0,
            },
        };
    let decoder_sparse_step = optional_model_usize(root_json, decoder_json, "decoder_sparse_step")?;
    let first_k_dense_replace =
        optional_model_usize(root_json, decoder_json, "first_k_dense_replace")?;
    let moe_layer_freq = optional_model_usize(root_json, decoder_json, "moe_layer_freq")?;
    let num_expert_groups =
        match optional_model_usize(root_json, decoder_json, "num_expert_groups")? {
            Some(value) => Some(value),
            None => optional_model_usize(root_json, decoder_json, "n_group")?,
        };
    let topk_group = optional_model_usize(root_json, decoder_json, "topk_group")?;
    let topk_method = optional_model_string(root_json, decoder_json, "topk_method")?;
    let scoring_func = optional_model_string(root_json, decoder_json, "scoring_func")?;
    let routed_scaling_factor =
        optional_model_f32(root_json, decoder_json, "routed_scaling_factor")?;
    let norm_topk_prob =
        optional_model_bool(root_json, decoder_json, "norm_topk_prob")?.unwrap_or(false);
    let mlp_only_layers =
        optional_model_usize_array(root_json, decoder_json, "mlp_only_layers")?.unwrap_or_default();

    let is_moe_architecture = matches!(
        architecture,
        HfArchitectureKind::MixtralMoe
            | HfArchitectureKind::Qwen2Moe
            | HfArchitectureKind::Qwen3Moe
            | HfArchitectureKind::Qwen35Moe
    );
    let is_moe_architecture = is_moe_architecture || architecture.is_deepseek();
    let has_moe_fields = num_experts.unwrap_or(0) > 0
        || num_experts_per_tok.is_some()
        || moe_intermediate_size.is_some();
    if !is_moe_architecture && !has_moe_fields {
        return Ok(ParsedMoeConfig {
            mlp_layer_types: vec![HfMlpLayerKind::Dense; num_hidden_layers],
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_experts: None,
            num_experts_per_tok: None,
            decoder_sparse_step: None,
            norm_topk_prob,
            first_k_dense_replace: None,
            moe_layer_freq: None,
            num_expert_groups,
            topk_group,
            topk_method,
            scoring_func,
            routed_scaling_factor,
        });
    }

    let num_experts = required_moe_usize(num_experts, "num_experts")?;
    let num_experts_per_tok = required_moe_usize(num_experts_per_tok, "num_experts_per_tok")?;
    let moe_intermediate_size = required_moe_usize(moe_intermediate_size, "moe_intermediate_size")?;
    let decoder_sparse_step = decoder_sparse_step.unwrap_or(1);
    if decoder_sparse_step == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "HF MoE decoder_sparse_step must be non-zero".to_string(),
        });
    }
    let resolved_moe_layer_freq = moe_layer_freq.unwrap_or(1);
    if resolved_moe_layer_freq == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "HF MoE moe_layer_freq must be non-zero".to_string(),
        });
    }
    if num_experts_per_tok > num_experts {
        return Err(NervaError::InvalidArgument {
            reason: "HF MoE num_experts_per_tok cannot exceed num_experts".to_string(),
        });
    }
    for layer in &mlp_only_layers {
        if *layer >= num_hidden_layers {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "HF MoE mlp_only_layers entry {layer} exceeds num_hidden_layers {num_hidden_layers}"
                ),
            });
        }
    }

    let mlp_layer_types = if architecture.is_deepseek() {
        let first_dense = first_k_dense_replace.unwrap_or(0);
        (0..num_hidden_layers)
            .map(|layer| {
                if layer >= first_dense && layer % resolved_moe_layer_freq == 0 {
                    HfMlpLayerKind::SparseMoe
                } else {
                    HfMlpLayerKind::Dense
                }
            })
            .collect()
    } else {
        (0..num_hidden_layers)
            .map(|layer| {
                if mlp_only_layers.contains(&layer) || (layer + 1) % decoder_sparse_step != 0 {
                    HfMlpLayerKind::Dense
                } else {
                    HfMlpLayerKind::SparseMoe
                }
            })
            .collect()
    };
    Ok(ParsedMoeConfig {
        mlp_layer_types,
        moe_intermediate_size: Some(moe_intermediate_size),
        shared_expert_intermediate_size: (shared_expert_intermediate_size > 0)
            .then_some(shared_expert_intermediate_size),
        num_experts: Some(num_experts),
        num_experts_per_tok: Some(num_experts_per_tok),
        decoder_sparse_step: Some(decoder_sparse_step),
        norm_topk_prob,
        first_k_dense_replace,
        moe_layer_freq: architecture
            .is_deepseek()
            .then_some(resolved_moe_layer_freq)
            .or(moe_layer_freq),
        num_expert_groups,
        topk_group,
        topk_method,
        scoring_func,
        routed_scaling_factor,
    })
}

fn parse_deepseek_config(
    root_json: &str,
    decoder_json: &str,
    architecture: HfArchitectureKind,
) -> Result<ParsedDeepSeekConfig> {
    let q_lora_rank = optional_model_usize(root_json, decoder_json, "q_lora_rank")?;
    let kv_lora_rank = optional_model_usize(root_json, decoder_json, "kv_lora_rank")?;
    let qk_rope_head_dim = optional_model_usize(root_json, decoder_json, "qk_rope_head_dim")?;
    let explicit_qk_nope_head_dim =
        optional_model_usize(root_json, decoder_json, "qk_nope_head_dim")?;
    let explicit_v_head_dim = optional_model_usize(root_json, decoder_json, "v_head_dim")?;
    let explicit_head_dim = optional_model_usize(root_json, decoder_json, "head_dim")?;
    let qk_nope_head_dim = match (
        explicit_qk_nope_head_dim,
        explicit_head_dim,
        qk_rope_head_dim,
    ) {
        (Some(value), _, _) => Some(value),
        (None, Some(head_dim), Some(rope)) if architecture == HfArchitectureKind::DeepSeekV4 => {
            Some(
                head_dim
                    .checked_sub(rope)
                    .ok_or_else(|| NervaError::InvalidArgument {
                        reason: "DeepSeek V4 head_dim must be at least qk_rope_head_dim"
                            .to_string(),
                    })?,
            )
        }
        _ => None,
    };
    let v_head_dim = explicit_v_head_dim.or_else(|| {
        architecture
            .is_deepseek()
            .then_some(explicit_head_dim)
            .flatten()
    });

    Ok(ParsedDeepSeekConfig {
        q_lora_rank,
        kv_lora_rank,
        o_lora_rank: optional_model_usize(root_json, decoder_json, "o_lora_rank")?,
        o_groups: optional_model_usize(root_json, decoder_json, "o_groups")?,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        index_topk: optional_model_usize(root_json, decoder_json, "index_topk")?,
        index_topk_freq: optional_model_usize(root_json, decoder_json, "index_topk_freq")?,
        index_skip_topk_offset: optional_model_usize(
            root_json,
            decoder_json,
            "index_skip_topk_offset",
        )?,
        index_topk_pattern: optional_model_string_array(
            root_json,
            decoder_json,
            "index_topk_pattern",
        )?
        .unwrap_or_default(),
        index_n_heads: optional_model_usize(root_json, decoder_json, "index_n_heads")?,
        index_head_dim: optional_model_usize(root_json, decoder_json, "index_head_dim")?,
        compress_ratios: optional_model_usize_array(root_json, decoder_json, "compress_ratios")?
            .unwrap_or_default(),
        hc_mult: optional_model_usize(root_json, decoder_json, "hc_mult")?,
        hc_sinkhorn_iters: optional_model_usize(root_json, decoder_json, "hc_sinkhorn_iters")?,
        hc_eps: optional_model_f32(root_json, decoder_json, "hc_eps")?,
        num_nextn_predict_layers: optional_model_usize(
            root_json,
            decoder_json,
            "num_nextn_predict_layers",
        )?,
        num_hash_layers: optional_model_usize(root_json, decoder_json, "num_hash_layers")?,
        compress_rope_theta: optional_model_f32(root_json, decoder_json, "compress_rope_theta")?,
        swiglu_limit: optional_model_f32(root_json, decoder_json, "swiglu_limit")?,
        expert_dtype: optional_model_string(root_json, decoder_json, "expert_dtype")?,
    })
}

fn resolve_intermediate_size(
    explicit_intermediate_size: Option<usize>,
    moe_config: &ParsedMoeConfig,
) -> Result<usize> {
    if let Some(intermediate_size) = explicit_intermediate_size {
        return Ok(intermediate_size);
    }
    if moe_config
        .mlp_layer_types
        .iter()
        .any(|kind| *kind == HfMlpLayerKind::Dense)
    {
        return Err(NervaError::InvalidArgument {
            reason: "HF config is missing required field intermediate_size".to_string(),
        });
    }
    let derived = moe_config
        .moe_intermediate_size
        .unwrap_or(0)
        .max(moe_config.shared_expert_intermediate_size.unwrap_or(0));
    if derived == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "HF config is missing required field intermediate_size".to_string(),
        });
    }
    Ok(derived)
}

fn parse_linear_attention_config(
    root_json: &str,
    decoder_json: &str,
    architecture: HfArchitectureKind,
) -> Result<ParsedLinearAttentionConfig> {
    let qwen35_defaults = matches!(
        architecture,
        HfArchitectureKind::Qwen35 | HfArchitectureKind::Qwen35Moe
    );
    Ok(ParsedLinearAttentionConfig {
        linear_conv_kernel_dim: optional_model_usize(
            root_json,
            decoder_json,
            "linear_conv_kernel_dim",
        )?
        .or(qwen35_defaults.then_some(4)),
        linear_key_head_dim: optional_model_usize(root_json, decoder_json, "linear_key_head_dim")?
            .or(qwen35_defaults.then_some(128)),
        linear_value_head_dim: optional_model_usize(
            root_json,
            decoder_json,
            "linear_value_head_dim",
        )?
        .or(qwen35_defaults.then_some(128)),
        linear_num_key_heads: optional_model_usize(
            root_json,
            decoder_json,
            "linear_num_key_heads",
        )?
        .or(qwen35_defaults.then_some(16)),
        linear_num_value_heads: optional_model_usize(
            root_json,
            decoder_json,
            "linear_num_value_heads",
        )?
        .or(qwen35_defaults.then_some(32)),
    })
}

fn required_moe_usize(value: Option<usize>, key: &'static str) -> Result<usize> {
    value.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("HF MoE config is missing required field {key}"),
    })
}

fn optional_model_usize_array(
    root_json: &str,
    decoder_json: &str,
    key: &'static str,
) -> Result<Option<Vec<usize>>> {
    match optional_usize_array(decoder_json, key)? {
        Some(value) => Ok(Some(value)),
        None if !std::ptr::eq(root_json, decoder_json) => optional_usize_array(root_json, key),
        None => Ok(None),
    }
}

fn optional_usize_array(config_json: &str, key: &'static str) -> Result<Option<Vec<usize>>> {
    let value: serde_json::Value =
        serde_json::from_str(config_json).map_err(|err| NervaError::InvalidArgument {
            reason: format!("HF config JSON is malformed: {err}"),
        })?;
    let Some(array) = value.get(key) else {
        return Ok(None);
    };
    if array.is_null() {
        return Ok(None);
    }
    let Some(array) = array.as_array() else {
        return Err(NervaError::InvalidArgument {
            reason: format!("HF config field {key} must be an unsigned integer array"),
        });
    };
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        let Some(value) = item.as_u64() else {
            return Err(NervaError::InvalidArgument {
                reason: format!("HF config field {key} must contain unsigned integers"),
            });
        };
        out.push(
            usize::try_from(value).map_err(|_| NervaError::InvalidArgument {
                reason: format!("HF config field {key} entry does not fit usize"),
            })?,
        );
    }
    Ok(Some(out))
}

fn optional_model_string_array(
    root_json: &str,
    decoder_json: &str,
    key: &'static str,
) -> Result<Option<Vec<String>>> {
    match optional_string_array(decoder_json, key)? {
        Some(value) => Ok(Some(value)),
        None if !std::ptr::eq(root_json, decoder_json) => optional_string_array(root_json, key),
        None => Ok(None),
    }
}

fn optional_string_array(config_json: &str, key: &'static str) -> Result<Option<Vec<String>>> {
    let value: serde_json::Value =
        serde_json::from_str(config_json).map_err(|err| NervaError::InvalidArgument {
            reason: format!("HF config JSON is malformed: {err}"),
        })?;
    let Some(array) = value.get(key) else {
        return Ok(None);
    };
    if array.is_null() {
        return Ok(None);
    }
    let Some(array) = array.as_array() else {
        return Err(NervaError::InvalidArgument {
            reason: format!("HF config field {key} must be a string array"),
        });
    };
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        let Some(value) = item.as_str() else {
            return Err(NervaError::InvalidArgument {
                reason: format!("HF config field {key} must contain strings"),
            });
        };
        out.push(value.to_string());
    }
    Ok(Some(out))
}

fn parse_attention_layer_types(
    root_json: &str,
    decoder_json: &str,
    architecture: HfArchitectureKind,
    num_hidden_layers: usize,
) -> Result<Vec<HfAttentionLayerKind>> {
    if let Some(kinds) = optional_attention_layer_types(decoder_json)? {
        return validate_attention_layer_count(kinds, num_hidden_layers);
    }
    if !std::ptr::eq(root_json, decoder_json) {
        if let Some(kinds) = optional_attention_layer_types(root_json)? {
            return validate_attention_layer_count(kinds, num_hidden_layers);
        }
    }
    if matches!(
        architecture,
        HfArchitectureKind::Qwen35 | HfArchitectureKind::Qwen35Moe
    ) {
        if let Some(interval) =
            optional_model_usize(root_json, decoder_json, "full_attention_interval")?
        {
            if interval == 0 {
                return Err(NervaError::InvalidArgument {
                    reason: "HF Qwen3.5 full_attention_interval must be non-zero".to_string(),
                });
            }
            return Ok((0..num_hidden_layers)
                .map(|layer| {
                    if (layer + 1).is_multiple_of(interval) {
                        HfAttentionLayerKind::Full
                    } else {
                        HfAttentionLayerKind::Linear
                    }
                })
                .collect());
        }
    }
    Ok(vec![HfAttentionLayerKind::Full; num_hidden_layers])
}

fn optional_attention_layer_types(config_json: &str) -> Result<Option<Vec<HfAttentionLayerKind>>> {
    let value: serde_json::Value =
        serde_json::from_str(config_json).map_err(|err| NervaError::InvalidArgument {
            reason: format!("HF config JSON is malformed: {err}"),
        })?;
    let Some(layer_types) = value.get("layer_types") else {
        return Ok(None);
    };
    let Some(layer_types) = layer_types.as_array() else {
        return Err(NervaError::InvalidArgument {
            reason: "HF config field layer_types must be a string array".to_string(),
        });
    };
    let mut kinds = Vec::with_capacity(layer_types.len());
    for layer_type in layer_types {
        let Some(layer_type) = layer_type.as_str() else {
            return Err(NervaError::InvalidArgument {
                reason: "HF config field layer_types must contain strings".to_string(),
            });
        };
        kinds.push(parse_attention_layer_kind(layer_type)?);
    }
    Ok(Some(kinds))
}

fn parse_attention_layer_kind(value: &str) -> Result<HfAttentionLayerKind> {
    match value {
        "full_attention"
        | "self_attention"
        | "attention"
        | "sliding_attention"
        | "compressed_sparse_attention"
        | "heavily_compressed_attention" => Ok(HfAttentionLayerKind::Full),
        "linear_attention" => Ok(HfAttentionLayerKind::Linear),
        other => Err(NervaError::InvalidArgument {
            reason: format!("unsupported HF attention layer type {other}"),
        }),
    }
}

fn validate_attention_layer_count(
    kinds: Vec<HfAttentionLayerKind>,
    num_hidden_layers: usize,
) -> Result<Vec<HfAttentionLayerKind>> {
    if kinds.len() != num_hidden_layers {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "HF layer_types length {} does not match num_hidden_layers {}",
                kinds.len(),
                num_hidden_layers
            ),
        });
    }
    Ok(kinds)
}

fn parse_torch_dtype(root_json: &str, decoder_json: &str) -> Result<Option<DType>> {
    if let Some(value) = optional_model_string(root_json, decoder_json, "torch_dtype")? {
        return dtype_from_hf_string(&value).map(Some);
    }
    optional_model_string(root_json, decoder_json, "dtype")?
        .as_deref()
        .map(dtype_from_hf_string)
        .transpose()
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
        "tensorfloat32" | "tf32" | "bf32" => Ok(DType::TF32),
        "float8" | "float8_e4m3" | "float8_e4m3fn" | "fp8" | "fp8_e4m3" | "f8" | "f8_e4m3" => {
            Ok(DType::F8E4M3)
        }
        "float8_e5m2" | "fp8_e5m2" | "f8_e5m2" => Ok(DType::F8E5M2),
        "float8_e8m0" | "fp8_e8m0" | "f8_e8m0" => Ok(DType::F8E8M0),
        "float4" | "float4_e2m1" | "fp4" | "f4" | "f4_e2m1" | "mxfp4" | "nvfp4" => {
            Ok(DType::F4E2M1)
        }
        "int4" | "i4" => Ok(DType::I4),
        "uint4" | "u4" => Ok(DType::U4),
        "int8" | "i8" => Ok(DType::I8),
        "uint8" | "u8" => Ok(DType::U8),
        other => Err(NervaError::InvalidArgument {
            reason: format!("unsupported HF torch_dtype {other}"),
        }),
    }
}
