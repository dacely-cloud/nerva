use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_step::request::{
    CUDA_HF_DECODE_STEP_DTYPE_BF16, CUDA_HF_DECODE_STEP_DTYPE_F16, CudaHfDecodeStepRequest,
};
use nerva_cuda::decode::hf_step::summary::CudaHfDecodeStepSummary;
use nerva_model::causal_lm::types::HfCausalLmModel;

pub(super) fn run_fused_step(
    model: &HfCausalLmModel,
    input_token: TokenId,
    step: usize,
) -> Result<CudaHfDecodeStepSummary> {
    let layer = model.layer(0).ok_or_else(|| NervaError::InvalidArgument {
        reason: "CUDA HF fused decode step requires one loaded layer".to_string(),
    })?;
    let view = layer.encoded_view();
    let shape = view.shape;
    Ok(CudaHfDecodeStepRequest {
        dtype: cuda_dtype(model.dtype())?,
        hidden: shape.hidden,
        heads: shape.heads,
        kv_heads: shape.kv_heads,
        head_dim: shape.head_dim,
        intermediate: shape.intermediate,
        vocab_size: model.metadata().vocab_size,
        position: step as u32,
        token_index: step as u64,
        rms_eps: view.rms_eps,
        rope_theta: view.rope_theta,
        input: model.embedding_row(input_token)?,
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
        final_norm_weight: model.final_norm_weight(),
        lm_head: model.lm_head(),
    }
    .run())
}

fn cuda_dtype(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(CUDA_HF_DECODE_STEP_DTYPE_F16),
        DType::BF16 => Ok(CUDA_HF_DECODE_STEP_DTYPE_BF16),
        other => Err(NervaError::InvalidArgument {
            reason: format!("CUDA HF fused decode step does not support dtype {other:?}"),
        }),
    }
}
