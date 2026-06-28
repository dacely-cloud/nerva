use crate::decode::hf_step::request::{CUDA_HF_DECODE_STEP_DTYPE_F16, CudaHfDecodeStepRequest};
use crate::decode::hf_step::summary::CudaHfDecodeStepSummary;
use crate::decode::summary::CudaTinyDecodeSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn tiny_decode_summary_serializes_device_first_fields() {
    let summary = CudaTinyDecodeSummary {
        status: SmokeStatus::Ok,
        steps: 8,
        ring_capacity: 4,
        seed_token: 0,
        vocab_size: 4,
        hidden: 2,
        last_token: Some(0),
        graph_replays: 8,
        graph_nodes: 1,
        observed_tokens: 8,
        observed_token_hash: 761644941098537893,
        token_ring_slots_touched: 4,
        token_ring_reuses: 4,
        token_ring_max_slot_version: 2,
        stale_tokens: 0,
        missing_tokens: 0,
        extra_tokens: 0,
        mismatched_tokens: 0,
        host_causality_edges: 0,
        resident_weight_bytes: 64,
        device_arena_bytes: 272,
        pinned_host_bytes: 272,
        h2d_bytes: 80,
        d2h_bytes: 208,
        graph_launches: 8,
        sync_calls: 1,
        kernel_launches: 8,
        hot_path_allocations: 0,
        token_ledgers: 8,
        graph_replay_events: 8,
        device_activity_events: 8,
        copy_events: 3,
        soft_visibility_syncs: 1,
        hard_syncs: 0,
        host_event_wait_ns: 1200,
        gpu_active_ns: 800,
        gpu_idle_ns: 0,
        wall_latency_ns: 1600,
        error: None,
    };
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"steps\":8"));
    assert!(json.contains("\"last_token\":0"));
    assert!(json.contains("\"observed_token_hash\":761644941098537893"));
    assert!(json.contains("\"token_ring_reuses\":4"));
    assert!(json.contains("\"host_causality_edges\":0"));
    assert!(json.contains("\"resident_weight_bytes\":64"));
    assert!(json.contains("\"H2D_bytes\":80"));
    assert!(json.contains("\"D2H_bytes\":208"));
    assert!(json.contains("\"kernel_launches\":8"));
    assert!(json.contains("\"token_ledgers\":8"));
    assert!(json.contains("\"device_activity_events\":8"));
    assert!(json.contains("\"soft_visibility_syncs\":1"));
    assert!(json.contains("\"host_event_wait_ns\":1200"));
    assert!(json.contains("\"gpu_idle_ns\":0"));
    assert!(json.contains("\"host_wait_gpu_idle_separated\":true"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn hf_decode_step_summary_serializes_fused_device_step_fields() {
    let summary = CudaHfDecodeStepSummary {
        status: SmokeStatus::Ok,
        dtype: CUDA_HF_DECODE_STEP_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        vocab_size: 4,
        token_index: 3,
        token: 1,
        slot_version: 1,
        completion: 1,
        output_hash: 9,
        resident_weight_bytes: 84,
        device_arena_bytes: 160,
        pinned_host_bytes: 132,
        h2d_bytes: 92,
        d2h_bytes: 40,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"vocab_size\":4"));
    assert!(json.contains("\"token_index\":3"));
    assert!(json.contains("\"token\":1"));
    assert!(json.contains("\"H2D_bytes\":92"));
    assert!(json.contains("\"D2H_bytes\":40"));
}

#[test]
fn hf_decode_step_runs_layer_and_final_head_when_device_is_available() {
    let _guard = super::cuda_test_lock();

    let one = 0x3c00;
    let zero = 0x0000;
    let neg_one = 0xbc00;
    let input = [one, zero];
    let rms = [one, one];
    let matrix = [zero; 4];
    let lm_head = [zero, neg_one, one, zero, zero, one, neg_one, zero];
    let summary = CudaHfDecodeStepRequest {
        dtype: CUDA_HF_DECODE_STEP_DTYPE_F16,
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
        final_norm_weight: &rms,
        lm_head: &lm_head,
    }
    .run();

    if summary.status != SmokeStatus::Ok {
        return;
    }
    assert_eq!(summary.token_index, 3);
    assert_eq!(summary.token, 1);
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.h2d_bytes >= summary.resident_weight_bytes);
    assert!(summary.d2h_bytes > 0);
}
