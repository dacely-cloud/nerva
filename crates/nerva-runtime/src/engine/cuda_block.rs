use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_cuda::block::forward::request::{
    CUDA_BLOCK_DTYPE_BF16, CUDA_BLOCK_DTYPE_F16, CudaBlockForwardRequest,
};
use nerva_cuda::block::forward::summary::CudaBlockForwardSummary;
use nerva_model::precision::block::model::PrecisionTransformerBlock;

pub fn run_precision_block_on_cuda(
    block: &PrecisionTransformerBlock,
    input: &[u16],
    position: u32,
) -> Result<CudaBlockForwardSummary> {
    let view = block.encoded_view();
    let shape = view.shape;
    let request = CudaBlockForwardRequest {
        dtype: cuda_dtype(view.dtype)?,
        hidden: shape.hidden,
        heads: shape.heads,
        kv_heads: shape.kv_heads,
        head_dim: shape.head_dim,
        intermediate: shape.intermediate,
        position,
        rms_eps: view.rms_eps,
        rope_theta: view.rope_theta,
        input,
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
    };
    Ok(request.run())
}

fn cuda_dtype(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(CUDA_BLOCK_DTYPE_F16),
        DType::BF16 => Ok(CUDA_BLOCK_DTYPE_BF16),
        other => Err(NervaError::InvalidArgument {
            reason: format!("CUDA precision block forward does not support dtype {other:?}"),
        }),
    }
}
