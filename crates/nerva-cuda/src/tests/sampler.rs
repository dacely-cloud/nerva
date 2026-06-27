use crate::sampler::hf_head::request::{CUDA_HF_SAMPLER_DTYPE_F16, CudaHfSamplerRequest};
use crate::sampler::hf_head::summary::CudaHfSamplerSummary;
use crate::sampler::summary::CudaGreedySamplerSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn greedy_sampler_summary_serializes_device_token_fields() {
    let summary = CudaGreedySamplerSummary {
        status: SmokeStatus::Ok,
        vocab_size: 4,
        token_index: 0,
        token: 2,
        slot_version: 1,
        completion: 1,
        device_arena_bytes: 64,
        pinned_host_bytes: 64,
        h2d_bytes: 16,
        d2h_bytes: 40,
        kernel_launches: 1,
        sync_calls: 2,
        hot_path_allocations: 0,
        error: None,
    };
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"vocab_size\":4"));
    assert!(json.contains("\"token\":2"));
    assert!(json.contains("\"slot_version\":1"));
    assert!(json.contains("\"H2D_bytes\":16"));
    assert!(json.contains("\"D2H_bytes\":40"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn hf_sampler_summary_serializes_loaded_final_head_fields() {
    let summary = CudaHfSamplerSummary {
        status: SmokeStatus::Ok,
        dtype: CUDA_HF_SAMPLER_DTYPE_F16,
        hidden: 2,
        vocab_size: 4,
        token_index: 7,
        token: 1,
        slot_version: 1,
        completion: 1,
        output_hash: 9,
        resident_weight_bytes: 20,
        device_arena_bytes: 92,
        pinned_host_bytes: 64,
        h2d_bytes: 24,
        d2h_bytes: 40,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"vocab_size\":4"));
    assert!(json.contains("\"token_index\":7"));
    assert!(json.contains("\"token\":1"));
    assert!(json.contains("\"resident_weight_bytes\":20"));
    assert!(json.contains("\"H2D_bytes\":24"));
    assert!(json.contains("\"D2H_bytes\":40"));
}

#[test]
fn hf_sampler_runs_loaded_final_head_when_device_is_available() {
    let one = 0x3c00;
    let zero = 0x0000;
    let neg_one = 0xbc00;
    let hidden_bits = [one, zero];
    let final_norm = [one, one];
    let lm_head = [zero, neg_one, one, zero, zero, one, neg_one, zero];
    let summary = CudaHfSamplerRequest {
        dtype: CUDA_HF_SAMPLER_DTYPE_F16,
        hidden: 2,
        vocab_size: 4,
        token_index: 7,
        rms_eps: 1e-5,
        hidden_bits: &hidden_bits,
        final_norm_weight: &final_norm,
        lm_head: &lm_head,
    }
    .run();

    if summary.status != SmokeStatus::Ok {
        return;
    }
    assert_eq!(summary.token_index, 7);
    assert_eq!(summary.token, 1);
    assert_eq!(summary.slot_version, 1);
    assert_eq!(summary.completion, 1);
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.h2d_bytes >= summary.resident_weight_bytes);
    assert!(summary.d2h_bytes > 0);
}
