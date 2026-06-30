use crate::common::dtype::dtype_to_str;
use crate::hf::metadata::HfModelMetadata;

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
        metadata.bos_token_id.unwrap_or_default() as u64,
        metadata.eos_token_id.unwrap_or_default() as u64,
        metadata.head_dim() as u64,
        metadata.kv_groups() as u64,
        u64::from(metadata.tie_word_embeddings),
        u64::from(metadata.attention_bias),
        u64::from(metadata.attention_qkv_bias),
        u64::from(metadata.attention_output_bias),
        u64::from(metadata.mlp_bias),
        metadata.linear_conv_kernel_dim.unwrap_or_default() as u64,
        metadata.linear_key_head_dim.unwrap_or_default() as u64,
        metadata.linear_value_head_dim.unwrap_or_default() as u64,
        metadata.linear_num_key_heads.unwrap_or_default() as u64,
        metadata.linear_num_value_heads.unwrap_or_default() as u64,
        metadata.moe_intermediate_size.unwrap_or_default() as u64,
        metadata.shared_expert_intermediate_size.unwrap_or_default() as u64,
        metadata.num_experts.unwrap_or_default() as u64,
        metadata.num_experts_per_tok.unwrap_or_default() as u64,
        metadata.decoder_sparse_step.unwrap_or_default() as u64,
        u64::from(metadata.norm_topk_prob),
        metadata.moe_first_k_dense_replace.unwrap_or_default() as u64,
        metadata.moe_layer_freq.unwrap_or_default() as u64,
        metadata.num_expert_groups.unwrap_or_default() as u64,
        metadata.topk_group.unwrap_or_default() as u64,
        metadata.routed_scaling_factor.unwrap_or_default().to_bits() as u64,
        metadata.q_lora_rank.unwrap_or_default() as u64,
        metadata.kv_lora_rank.unwrap_or_default() as u64,
        metadata.qk_nope_head_dim.unwrap_or_default() as u64,
        metadata.qk_rope_head_dim.unwrap_or_default() as u64,
        metadata.v_head_dim.unwrap_or_default() as u64,
        metadata.index_topk.unwrap_or_default() as u64,
        metadata.index_n_heads.unwrap_or_default() as u64,
        metadata.index_head_dim.unwrap_or_default() as u64,
        metadata.hc_mult.unwrap_or_default() as u64,
        metadata.hc_sinkhorn_iters.unwrap_or_default() as u64,
        metadata.hc_eps.unwrap_or_default().to_bits() as u64,
        metadata.num_nextn_predict_layers.unwrap_or_default() as u64,
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
    if let Some(hidden_act) = metadata.hidden_act.as_deref() {
        for byte in hidden_act.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    if let Some(dtype) = metadata.torch_dtype {
        for byte in dtype_to_str(dtype).as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for value in &metadata.compress_ratios {
        for byte in (*value as u64).to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for value in [
        metadata.topk_method.as_deref(),
        metadata.scoring_func.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        for byte in value.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for kind in &metadata.attention_layer_types {
        for byte in kind.as_str().as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for kind in &metadata.mlp_layer_types {
        for byte in kind.as_str().as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}
