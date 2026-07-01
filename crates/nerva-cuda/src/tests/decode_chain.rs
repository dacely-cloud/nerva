use crate::decode::hf_chain::layer::{
    CUDA_HF_ATTENTION_FULL, CUDA_HF_DEEPSEEK_MODE_V4_SWA, CUDA_HF_MLP_SPARSE_MOE,
    CudaHfDecodeChainLayer, CudaHfDeepSeekLayer,
};
use crate::decode::hf_chain::request::{CUDA_HF_DECODE_CHAIN_DTYPE_F16, CudaHfDecodeChainRequest};
use crate::decode::hf_chain::summary::CudaHfDecodeChainSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn hf_decode_chain_summary_serializes_layer_count() {
    let summary = CudaHfDecodeChainSummary {
        status: SmokeStatus::Ok,
        dtype: CUDA_HF_DECODE_CHAIN_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        vocab_size: 4,
        layer_count: 2,
        token_index: 3,
        token: 1,
        slot_version: 1,
        completion: 1,
        output_hash: 9,
        resident_weight_bytes: 128,
        device_arena_bytes: 240,
        pinned_host_bytes: 180,
        h2d_bytes: 160,
        d2h_bytes: 40,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"layer_count\":2"));
    assert!(json.contains("\"token\":1"));
    assert!(json.contains("\"kernel_launches\":1"));
}

#[test]
fn hf_decode_chain_runs_two_layers_and_final_head_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let one = 0x3c00;
    let zero = 0x0000;
    let neg_one = 0xbc00;
    let input = [one, zero];
    let rms = [one, one];
    let matrix = [zero; 4];
    let lm_head = [zero, neg_one, one, zero, zero, one, neg_one, zero];
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &matrix,
        w_q_gate: None,
        w_k: &matrix,
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &matrix,
        w_o: &matrix,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &matrix,
        w_up: &matrix,
        w_down: &matrix,
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: None,
        mlp_kind: 0,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: false,
        attention_kind: CUDA_HF_ATTENTION_FULL,
    };
    let layers = [layer.clone(), layer];
    let summary = CudaHfDecodeChainRequest {
        dtype: CUDA_HF_DECODE_CHAIN_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        vocab_size: 4,
        position: 0,
        token_index: 3,
        rms_eps: 1e-5,
        rope_theta: None,
        input: &input,
        layers: &layers,
        final_norm_weight: &rms,
        lm_head: &lm_head,
    }
    .run();

    if summary.status != SmokeStatus::Ok {
        return;
    }
    assert_eq!(summary.layer_count, 2);
    assert_eq!(summary.token_index, 3);
    assert_eq!(summary.token, 1);
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.h2d_bytes >= summary.resident_weight_bytes);
    assert!(summary.d2h_bytes > 0);
}

#[test]
fn hf_decode_chain_runs_sparse_moe_full_attention_layer() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let one = 0x3c00;
    let zero = 0x0000;
    let input = [one, zero];
    let rms = [one, one];
    let matrix = [zero; 4];
    let router = [
        one, zero, // expert 0 wins for hidden state [1, 0]
        zero, zero,
    ];
    let expert_gate_up = [
        one, zero, // expert 0 gate row 0
        zero, zero, // expert 0 gate row 1
        one, zero, // expert 0 up row 0
        zero, zero, // expert 0 up row 1
        zero, zero, // expert 1 gate row 0
        zero, zero, // expert 1 gate row 1
        zero, zero, // expert 1 up row 0
        zero, zero, // expert 1 up row 1
    ];
    let expert_down = [
        zero, zero, // expert 0 output dim 0
        one, zero, // expert 0 output dim 1
        zero, zero, // expert 1 output dim 0
        zero, zero, // expert 1 output dim 1
    ];
    let lm_head = [
        zero, zero, // token 0
        zero, one, // token 1 rewards the MoE-written dimension
        zero, zero, // token 2
        zero, zero, // token 3
    ];
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &matrix,
        w_q_gate: None,
        w_k: &matrix,
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &matrix,
        w_o: &matrix,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &[],
        w_up: &[],
        w_down: &[],
        w_router: Some(&router),
        w_expert_gate_up: Some(&expert_gate_up),
        w_expert_down: Some(&expert_down),
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: None,
        mlp_kind: CUDA_HF_MLP_SPARSE_MOE,
        moe_intermediate: 2,
        shared_expert_intermediate: 0,
        num_experts: 2,
        experts_per_token: 1,
        norm_topk_prob: true,
        attention_kind: CUDA_HF_ATTENTION_FULL,
    };
    let summary = CudaHfDecodeChainRequest {
        dtype: CUDA_HF_DECODE_CHAIN_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 4,
        position: 0,
        token_index: 0,
        rms_eps: 1e-5,
        rope_theta: None,
        input: &input,
        layers: &[layer],
        final_norm_weight: &rms,
        lm_head: &lm_head,
    }
    .run();

    if summary.status != SmokeStatus::Ok {
        return;
    }
    assert_eq!(summary.layer_count, 1);
    assert_eq!(summary.token_index, 0);
    assert_eq!(summary.token, 1);
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
}

#[test]
fn hf_decode_chain_applies_deepseek_swiglu_limit_in_dense_mlp() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let zero = 0x0000;
    let one = 0x3c00;
    let two = 0x4000;
    let input = [one, zero];
    let rms = [one, one];
    let attention_matrix = [zero; 4];
    let gate = [two, zero];
    let up = [two, zero];
    let down = [zero, one];
    let lm_head = [
        one, zero, // token 0 rewards the residual dimension
        zero, one, // token 1 rewards the clamped FFN dimension
    ];
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &attention_matrix,
        w_q_gate: None,
        w_k: &attention_matrix,
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &attention_matrix,
        w_o: &attention_matrix,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &gate,
        w_up: &up,
        w_down: &down,
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: Some(deepseek_layer_with_swiglu_limit(1.0)),
        mlp_kind: 0,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: false,
        attention_kind: CUDA_HF_ATTENTION_FULL,
    };
    let summary = CudaHfDecodeChainRequest {
        dtype: CUDA_HF_DECODE_CHAIN_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 1,
        vocab_size: 2,
        position: 0,
        token_index: 0,
        rms_eps: 1e-5,
        rope_theta: None,
        input: &input,
        layers: &[layer],
        final_norm_weight: &rms,
        lm_head: &lm_head,
    }
    .run();

    if summary.status != SmokeStatus::Ok {
        return;
    }
    assert_eq!(summary.token, 0);
}

fn deepseek_layer_with_swiglu_limit(limit: f32) -> CudaHfDeepSeekLayer {
    CudaHfDeepSeekLayer {
        mode: CUDA_HF_DEEPSEEK_MODE_V4_SWA,
        flags: 0,
        hc_mult: 0,
        hc_sinkhorn_iters: 0,
        q_lora_rank: 0,
        kv_lora_rank: 0,
        o_lora_rank: 0,
        o_groups: 0,
        qk_nope_head_dim: 0,
        qk_rope_head_dim: 0,
        v_head_dim: 0,
        compress_ratio: 0,
        index_topk: 0,
        index_n_heads: 0,
        index_head_dim: 0,
        router_num_groups: 0,
        router_topk_groups: 0,
        routed_scaling_factor: 1.0,
        hc_eps: 0.0,
        hc_post_alpha: 0.0,
        rope_scaling_type: 0,
        rope_original_max_position: 0,
        rope_scaling_factor: 1.0,
        rope_extrapolation_factor: 1.0,
        rope_attn_factor: 1.0,
        rope_beta_fast: 32.0,
        rope_beta_slow: 1.0,
        rope_mscale: 1.0,
        rope_mscale_all_dim: 0.0,
        compress_rope_theta: None,
        swiglu_limit: Some(limit),
    }
}
