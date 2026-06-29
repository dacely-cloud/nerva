use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_chain::layer::{
    CUDA_HF_ATTENTION_FULL, CUDA_HF_ATTENTION_LINEAR_GDN, CUDA_HF_MLP_DENSE,
    CUDA_HF_MLP_SPARSE_MOE, CudaHfDecodeChainLayer, CudaHfLinearGdnLayer,
};
use nerva_cuda::decode::hf_sequence::request::{
    CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16, CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
    CudaHfDecodeSamplerConfig, CudaHfDecodeSequenceRequest,
};
use nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use nerva_cuda::decode::hf_sequence::weight_plan::{
    CudaHfDecodeSequenceWeightBlock, CudaHfDecodeSequenceWeightPlan,
};
use nerva_model::causal_lm::types::{HfCausalLmLayer, HfCausalLmModel};

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
    let first_layer = model
        .causal_layer(0)
        .ok_or_else(|| NervaError::InvalidArgument {
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
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    }
    .run())
}

fn sequence_layers(model: &HfCausalLmModel) -> Result<Vec<CudaHfDecodeChainLayer<'_>>> {
    let mut layers = Vec::with_capacity(model.layer_count());
    for index in 0..model.layer_count() {
        let layer = model
            .causal_layer(index)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("CUDA HF decode sequence layer index {index} is out of range"),
            })?;
        layers.push(sequence_layer(layer)?);
    }
    Ok(layers)
}

fn sequence_layer(layer: &HfCausalLmLayer) -> Result<CudaHfDecodeChainLayer<'_>> {
    match layer {
        HfCausalLmLayer::Dense(layer) => {
            let view = layer.encoded_view();
            Ok(CudaHfDecodeChainLayer {
                rms_attn_weight: view.rms_attn_weight,
                rms_mlp_weight: view.rms_mlp_weight,
                w_q: view.w_q,
                w_q_gate: view.w_q_gate,
                w_k: view.w_k,
                q_norm_weight: view.q_norm_weight,
                k_norm_weight: view.k_norm_weight,
                w_v: view.w_v,
                w_o: view.w_o,
                q_bias: view.q_bias,
                k_bias: view.k_bias,
                v_bias: view.v_bias,
                o_bias: view.o_bias,
                w_gate: view.w_gate,
                w_up: view.w_up,
                w_down: view.w_down,
                w_router: None,
                w_expert_gate_up: None,
                w_expert_down: None,
                w_shared_expert_gate: None,
                w_shared_expert_up: None,
                w_shared_expert_down: None,
                w_shared_expert_router: None,
                linear_gdn: None,
                mlp_kind: CUDA_HF_MLP_DENSE,
                moe_intermediate: 0,
                shared_expert_intermediate: 0,
                num_experts: 0,
                experts_per_token: 0,
                norm_topk_prob: false,
                attention_kind: CUDA_HF_ATTENTION_FULL,
            })
        }
        HfCausalLmLayer::SparseMoe(layer) => {
            let view = layer.encoded_view();
            Ok(CudaHfDecodeChainLayer {
                rms_attn_weight: view.rms_attn_weight,
                rms_mlp_weight: view.rms_mlp_weight,
                w_q: view.w_q,
                w_q_gate: view.w_q_gate,
                w_k: view.w_k,
                q_norm_weight: view.q_norm_weight,
                k_norm_weight: view.k_norm_weight,
                w_v: view.w_v,
                w_o: view.w_o,
                q_bias: view.q_bias,
                k_bias: view.k_bias,
                v_bias: view.v_bias,
                o_bias: view.o_bias,
                w_gate: &[],
                w_up: &[],
                w_down: &[],
                w_router: Some(view.router),
                w_expert_gate_up: Some(view.expert_gate_up),
                w_expert_down: Some(view.expert_down),
                w_shared_expert_gate: non_empty(view.shared_expert_gate),
                w_shared_expert_up: non_empty(view.shared_expert_up),
                w_shared_expert_down: non_empty(view.shared_expert_down),
                w_shared_expert_router: non_empty(view.shared_expert_router),
                linear_gdn: None,
                mlp_kind: CUDA_HF_MLP_SPARSE_MOE,
                moe_intermediate: view.moe_intermediate,
                shared_expert_intermediate: view.shared_expert_intermediate,
                num_experts: view.num_experts,
                experts_per_token: view.experts_per_token,
                norm_topk_prob: view.norm_topk_prob,
                attention_kind: CUDA_HF_ATTENTION_FULL,
            })
        }
        HfCausalLmLayer::GatedDeltaNetMoe(layer) => {
            let view = layer.encoded_view();
            Ok(CudaHfDecodeChainLayer {
                rms_attn_weight: view.rms_attn_weight,
                rms_mlp_weight: view.rms_mlp_weight,
                w_q: &[],
                w_q_gate: None,
                w_k: &[],
                q_norm_weight: None,
                k_norm_weight: None,
                w_v: &[],
                w_o: &[],
                q_bias: None,
                k_bias: None,
                v_bias: None,
                o_bias: None,
                w_gate: &[],
                w_up: &[],
                w_down: &[],
                w_router: Some(view.router),
                w_expert_gate_up: Some(view.expert_gate_up),
                w_expert_down: Some(view.expert_down),
                w_shared_expert_gate: non_empty(view.shared_expert_gate),
                w_shared_expert_up: non_empty(view.shared_expert_up),
                w_shared_expert_down: non_empty(view.shared_expert_down),
                w_shared_expert_router: non_empty(view.shared_expert_router),
                linear_gdn: Some(CudaHfLinearGdnLayer {
                    key_heads: view.gdn.key_heads,
                    value_heads: view.gdn.value_heads,
                    key_head_dim: view.gdn.key_head_dim,
                    value_head_dim: view.gdn.value_head_dim,
                    conv_kernel: view.gdn.conv_kernel,
                    w_conv: view.linear_conv,
                    w_qkv: view.linear_qkv,
                    w_z: view.linear_z,
                    w_b: view.linear_b,
                    w_a: view.linear_a,
                    dt_bias: view.linear_dt_bias,
                    a_log: view.linear_a_log,
                    norm_weight: view.linear_norm_bits,
                    w_out: view.linear_out,
                }),
                mlp_kind: CUDA_HF_MLP_SPARSE_MOE,
                moe_intermediate: view.moe.moe_intermediate,
                shared_expert_intermediate: view.moe.shared_expert_intermediate,
                num_experts: view.moe.num_experts,
                experts_per_token: view.moe.experts_per_token,
                norm_topk_prob: view.moe.norm_topk_prob,
                attention_kind: CUDA_HF_ATTENTION_LINEAR_GDN,
            })
        }
    }
}

fn non_empty(slice: &[u16]) -> Option<&[u16]> {
    (!slice.is_empty()).then_some(slice)
}

pub(super) fn cuda_dtype(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(CUDA_HF_DECODE_SEQUENCE_DTYPE_F16),
        DType::BF16 => Ok(CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16),
        other => Err(NervaError::InvalidArgument {
            reason: format!("CUDA HF decode sequence does not support dtype {other:?}"),
        }),
    }
}
