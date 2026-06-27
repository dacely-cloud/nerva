use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
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
