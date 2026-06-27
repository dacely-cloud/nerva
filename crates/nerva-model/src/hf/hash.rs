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
        u64::from(metadata.mlp_bias),
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
    hash
}
