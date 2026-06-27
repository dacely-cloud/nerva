use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::request::{
    CUDA_HF_DECODE_SEQUENCE_DTYPE_F16, CudaHfDecodeSequenceRequest,
};
use crate::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn hf_decode_sequence_summary_serializes_device_token_fields() {
    let summary = CudaHfDecodeSequenceSummary {
        status: SmokeStatus::Ok,
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        vocab_size: 4,
        layer_count: 2,
        steps: 4,
        seed_token: 0,
        tokens: vec![1, 2, 3, 0],
        observed_token_hash: 7,
        resident_weight_bytes: 128,
        device_arena_bytes: 240,
        pinned_host_bytes: 180,
        h2d_bytes: 160,
        d2h_bytes: 160,
        kernel_launches: 4,
        sync_calls: 1,
        host_causality_edges: 0,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"steps\":4"));
    assert!(json.contains("\"tokens\":[1,2,3,0]"));
    assert!(json.contains("\"sync_calls\":1"));
    assert!(json.contains("\"host_causality_edges\":0"));
}

#[test]
fn hf_decode_sequence_runs_device_first_steps_when_device_is_available() {
    let one = 0x3c00;
    let zero = 0x0000;
    let neg_one = 0xbc00;
    let embeddings = [one, zero, zero, one, neg_one, zero, zero, neg_one];
    let rms = [one, one];
    let matrix = [zero; 4];
    let lm_head = [zero, neg_one, one, zero, zero, one, neg_one, zero];
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &matrix,
        w_k: &matrix,
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
    let layers = [layer];
    let summary = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        vocab_size: 4,
        steps: 4,
        seed_token: 0,
        eos_token: None,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &embeddings,
        layers: &layers,
        final_norm_weight: &rms,
        lm_head: &lm_head,
    }
    .run();

    if summary.status != SmokeStatus::Ok {
        return;
    }
    assert_eq!(summary.tokens, vec![1, 2, 3, 0]);
    assert_eq!(summary.kernel_launches, 4);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.host_causality_edges, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.h2d_bytes >= summary.resident_weight_bytes);
    assert!(summary.d2h_bytes > 0);
}
