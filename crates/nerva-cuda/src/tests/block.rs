use crate::block::forward::request::{CUDA_BLOCK_DTYPE_F16, CudaBlockForwardRequest};
use crate::block::summary::{CudaLoadedTinyBlockSummary, CudaTinyBlockSummary};
use crate::smoke::status::SmokeStatus;

#[test]
fn tiny_block_summary_serializes_device_block_fields() {
    let summary = CudaTinyBlockSummary {
        status: SmokeStatus::Ok,
        hidden: 2,
        intermediate: 2,
        output: [15_360, 16_384],
        output_hash: 99,
        device_arena_bytes: 4,
        pinned_host_bytes: 4,
        kernel_launches: 1,
        sync_calls: 1,
        d2h_bytes: 4,
        hot_path_allocations: 0,
        error: None,
    };
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"hidden\":2"));
    assert!(json.contains("\"output_bits\":[15360,16384]"));
    assert!(json.contains("\"kernel_launches\":1"));
    assert!(json.contains("\"D2H_bytes\":4"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn loaded_tiny_block_summary_serializes_residency_fields() {
    let summary = CudaLoadedTinyBlockSummary {
        status: SmokeStatus::Ok,
        hidden: 2,
        intermediate: 2,
        output: [16_126, 17_299],
        output_hash: 17766510782028265595,
        resident_weight_bytes: 64,
        device_arena_bytes: 72,
        pinned_host_bytes: 72,
        h2d_bytes: 72,
        d2h_bytes: 4,
        kernel_launches: 1,
        sync_calls: 2,
        hot_path_allocations: 0,
        error: None,
    };
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"resident_weight_bytes\":64"));
    assert!(json.contains("\"H2D_bytes\":72"));
    assert!(json.contains("\"D2H_bytes\":4"));
    assert!(json.contains("\"kernel_launches\":1"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn generic_block_forward_runs_loaded_tiny_weights() {
    let _guard = super::cuda_test_lock();

    let zero = 0x0000;
    let half = 0x3800;
    let one = 0x3c00;
    let two = 0x4000;
    let input = [one, two];
    let rms = [one, one];
    let identity = [one, zero, zero, one];
    let gate = [half, zero, zero, half];
    let request = CudaBlockForwardRequest {
        dtype: CUDA_BLOCK_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        position: 0,
        rms_eps: 1e-5,
        rope_theta: None,
        input: &input,
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &identity,
        w_k: &identity,
        w_v: &identity,
        w_o: &identity,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &gate,
        w_up: &identity,
        w_down: &identity,
    };

    let summary = request.run();

    if summary.status != SmokeStatus::Ok {
        return;
    }
    assert_eq!(summary.status, SmokeStatus::Ok);
    assert_eq!(summary.output, [16126, 17299]);
    assert_eq!(summary.output_hash, 17766510782028265595);
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.h2d_bytes >= summary.resident_weight_bytes);
    assert!(summary.to_json().contains("\"status\":\"ok\""));
}
