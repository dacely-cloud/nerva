use crate::deepseek_mhc::hc_head::{
    CudaDeepSeekMhcHeadInput, CudaDeepSeekMhcHeadSummary, deepseek_mhc_head,
};
use crate::deepseek_mhc::probe::{
    deepseek_mhc_head_fixture, deepseek_mhc_head_smoke, reference_mhc_head,
};
use crate::smoke::status::SmokeStatus;

#[test]
fn deepseek_mhc_head_summary_serializes_output() {
    let summary = CudaDeepSeekMhcHeadSummary {
        status: SmokeStatus::Ok,
        return_code: 0,
        cuda_error: 0,
        mhc_error: 0,
        tokens: 2,
        hc_mult: 2,
        hidden_size: 3,
        rms_eps: 1e-5,
        hc_eps: 0.001,
        hc_scale: 1.25,
        output: vec![0.1, -0.2, 0.3],
        output_hash: 17,
        device_arena_bytes: 64,
        pinned_host_bytes: 0,
        h2d_bytes: 48,
        d2h_bytes: 32,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"tokens\":2"));
    assert!(json.contains("\"hc_mult\":2"));
    assert!(json.contains("\"hidden_size\":3"));
    assert!(json.contains("\"hc_scale\":1.25"));
    assert!(json.contains("\"output\":[0.1,-0.2,0.3]"));
    assert!(json.contains("\"kernel_launches\":1"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn deepseek_mhc_head_rejects_invalid_shapes_before_cuda() {
    let fixture = deepseek_mhc_head_fixture();
    let summary = deepseek_mhc_head(CudaDeepSeekMhcHeadInput {
        hidden_states: &fixture.hidden_states[..fixture.hidden_states.len() - 1],
        ..fixture.input()
    });

    assert_eq!(summary.status, SmokeStatus::Failed);
    assert_eq!(summary.return_code, -1);
    assert_eq!(summary.cuda_error, 0);
    assert_eq!(summary.kernel_launches, 0);
    assert_eq!(summary.sync_calls, 0);
    assert!(summary.output.iter().all(|value| *value == 0.0));
    assert!(summary.error.as_deref().map_or(false, |error| {
        error.contains("invalid DeepSeek mHC head shape")
    }));
}

#[test]
fn deepseek_mhc_head_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_mhc_head_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_mhc_head_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.tokens, 2);
    assert_eq!(second.hc_mult, 2);
    assert_eq!(second.hidden_size, 3);
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
    assert_eq!(second.output_hash, first.output_hash);
    assert_eq!(second.output, first.output);
}

#[test]
fn deepseek_mhc_head_api_matches_vllm_hc_head_reference() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let fixture = deepseek_mhc_head_fixture();
    let summary = deepseek_mhc_head(fixture.input());
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected = reference_mhc_head(fixture.input());
    assert_eq!(summary.tokens, 2);
    assert_eq!(summary.hc_mult, 2);
    assert_eq!(summary.hidden_size, 3);
    assert_eq!(summary.output.len(), expected.len());
    for (actual, expected) in summary.output.iter().zip(expected.iter()) {
        assert_close(*actual, *expected, 1e-5);
    }
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.output_hash != 0);
}

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} tolerance={tolerance}"
    );
}
