use crate::decode::hf_sequence::request::{
    CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16, CudaHfDecodeSequenceRequest,
};

pub(super) fn validate_request(request: &CudaHfDecodeSequenceRequest<'_>) -> Option<String> {
    if request.hidden == 0 || request.heads == 0 || request.kv_heads == 0 || request.head_dim == 0 {
        return Some("CUDA HF decode sequence dimensions must be non-zero".to_string());
    }
    if request.vocab_size == 0 || request.intermediate == 0 || request.steps == 0 {
        return Some("CUDA HF decode sequence steps and vocabulary must be non-zero".to_string());
    }
    if request.layers.is_empty() {
        return Some("CUDA HF decode sequence requires at least one layer".to_string());
    }
    if request.seed_token as usize >= request.vocab_size {
        return Some("CUDA HF decode sequence seed token is outside vocabulary".to_string());
    }
    validate_prompt(request)
        .or_else(|| validate_attention(request))
        .or_else(|| validate_weight_plan(request))
        .or_else(|| validate_lengths(request))
}

fn validate_prompt(request: &CudaHfDecodeSequenceRequest<'_>) -> Option<String> {
    if request.prompt_tokens.is_empty() {
        return Some("CUDA HF decode sequence requires prompt tokens".to_string());
    }
    if request
        .prompt_tokens
        .iter()
        .any(|token| *token as usize >= request.vocab_size)
    {
        return Some("CUDA HF decode sequence prompt token is outside vocabulary".to_string());
    }
    None
}

fn validate_attention(request: &CudaHfDecodeSequenceRequest<'_>) -> Option<String> {
    if request.kv_heads > request.heads || !request.heads.is_multiple_of(request.kv_heads) {
        return Some("CUDA HF decode sequence KV heads must divide attention heads".to_string());
    }
    if request.dtype > CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16 {
        return Some("CUDA HF decode sequence dtype is unsupported".to_string());
    }
    if request.rope_theta.is_some() && !request.head_dim.is_multiple_of(2) {
        return Some("CUDA HF decode sequence RoPE requires an even head dimension".to_string());
    }
    None
}

fn validate_weight_plan(request: &CudaHfDecodeSequenceRequest<'_>) -> Option<String> {
    request.weight_plan.and_then(|plan| {
        plan.validate()
            .or_else(|| plan.validate_descriptors(request.weight_blocks))
    })
}

fn validate_lengths(request: &CudaHfDecodeSequenceRequest<'_>) -> Option<String> {
    if request.embeddings.len() != request.vocab_size * request.hidden {
        return Some("CUDA HF decode sequence embeddings length does not match shape".to_string());
    }
    if request.final_norm_weight.len() != request.hidden {
        return Some("CUDA HF decode sequence final norm length does not match hidden".to_string());
    }
    if request.lm_head.len() != request.vocab_size * request.hidden {
        return Some("CUDA HF decode sequence LM head length does not match shape".to_string());
    }
    let attention_hidden = request.heads * request.head_dim;
    let kv_hidden = request.kv_heads * request.head_dim;
    request
        .layers
        .iter()
        .enumerate()
        .find_map(|(index, layer)| {
            layer
                .validate(
                    request.hidden,
                    attention_hidden,
                    kv_hidden,
                    request.intermediate,
                )
                .map(|error| format!("layer {index}: {error}"))
        })
}
