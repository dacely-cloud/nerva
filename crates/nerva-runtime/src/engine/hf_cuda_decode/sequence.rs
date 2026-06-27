use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use nerva_cuda::decode::hf_sequence::request::{
    CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16, CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
    CudaHfDecodeSequenceRequest,
};
use nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use nerva_cuda::decode::hf_sequence::weight_plan::{
    CudaHfDecodeSequenceWeightBlock, CudaHfDecodeSequenceWeightPlan,
};
use nerva_model::causal_lm::types::HfCausalLmModel;

pub(super) fn run_device_sequence(
    model: &HfCausalLmModel,
    prompt_tokens: &[TokenId],
    steps: usize,
    weight_plan: Option<CudaHfDecodeSequenceWeightPlan>,
    weight_blocks: &[CudaHfDecodeSequenceWeightBlock],
) -> Result<CudaHfDecodeSequenceSummary> {
    let seed = prompt_tokens
        .last()
        .copied()
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "CUDA HF decode sequence requires prompt tokens".to_string(),
        })?;
    let prompt_token_ids = prompt_tokens
        .iter()
        .map(|token| token.0)
        .collect::<Vec<_>>();
    let layers = sequence_layers(model)?;
    let first_layer = model.layer(0).ok_or_else(|| NervaError::InvalidArgument {
        reason: "CUDA HF decode sequence requires at least one loaded layer".to_string(),
    })?;
    let shape = model.shape();
    Ok(CudaHfDecodeSequenceRequest {
        dtype: cuda_dtype(model.dtype())?,
        hidden: shape.hidden,
        heads: shape.heads,
        kv_heads: shape.kv_heads,
        head_dim: shape.head_dim(),
        intermediate: shape.intermediate,
        vocab_size: model.metadata().vocab_size,
        steps,
        seed_token: seed.0,
        prompt_tokens: &prompt_token_ids,
        eos_token: model.metadata().eos_token_id,
        rms_eps: model.rms_eps(),
        rope_theta: first_layer.rope_theta(),
        embeddings: model.token_embeddings(),
        layers: &layers,
        final_norm_weight: model.final_norm_weight(),
        lm_head: model.lm_head(),
        weight_plan,
        weight_blocks,
    }
    .run())
}

fn sequence_layers(model: &HfCausalLmModel) -> Result<Vec<CudaHfDecodeChainLayer<'_>>> {
    let mut layers = Vec::with_capacity(model.layer_count());
    for index in 0..model.layer_count() {
        let layer = model
            .layer(index)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("CUDA HF decode sequence layer index {index} is out of range"),
            })?;
        let view = layer.encoded_view();
        layers.push(CudaHfDecodeChainLayer {
            rms_attn_weight: view.rms_attn_weight,
            rms_mlp_weight: view.rms_mlp_weight,
            w_q: view.w_q,
            w_k: view.w_k,
            w_v: view.w_v,
            w_o: view.w_o,
            q_bias: view.q_bias,
            k_bias: view.k_bias,
            v_bias: view.v_bias,
            o_bias: view.o_bias,
            w_gate: view.w_gate,
            w_up: view.w_up,
            w_down: view.w_down,
        });
    }
    Ok(layers)
}

fn cuda_dtype(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(CUDA_HF_DECODE_SEQUENCE_DTYPE_F16),
        DType::BF16 => Ok(CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16),
        other => Err(NervaError::InvalidArgument {
            reason: format!("CUDA HF decode sequence does not support dtype {other:?}"),
        }),
    }
}
