use crate::deepseek_mhc::fused_post_pre::{
    CudaDeepSeekMhcFusedPostPreInput, CudaDeepSeekMhcFusedPostPreSummary,
    deepseek_mhc_fused_post_pre,
};
use crate::deepseek_mhc::hc_head::{
    CudaDeepSeekMhcHeadInput, CudaDeepSeekMhcHeadSummary, deepseek_mhc_head,
};
use crate::deepseek_mhc::post::{
    CudaDeepSeekMhcPostInput, CudaDeepSeekMhcPostSummary, deepseek_mhc_post,
};
use crate::deepseek_mhc::pre::{
    CudaDeepSeekMhcPreInput, CudaDeepSeekMhcPreSummary, deepseek_mhc_pre,
};
use crate::deepseek_mhc::probe::{
    deepseek_mhc_fused_post_pre_smoke, deepseek_mhc_head_fixture, deepseek_mhc_head_smoke,
    deepseek_mhc_post_smoke, deepseek_mhc_pre_fixture, deepseek_mhc_pre_smoke,
    reference_mhc_fused_post_pre, reference_mhc_head, reference_mhc_post, reference_mhc_pre,
};
use crate::smoke::status::SmokeStatus;

#[test]
fn deepseek_mhc_pre_summary_serializes_outputs() {
    let summary = CudaDeepSeekMhcPreSummary {
        status: SmokeStatus::Ok,
        return_code: 0,
        cuda_error: 0,
        mhc_error: 0,
        tokens: 2,
        hc_mult: 2,
        hidden_size: 3,
        sinkhorn_repeat: 3,
        rms_eps: 1e-5,
        hc_pre_eps: 0.001,
        hc_sinkhorn_eps: 0.0001,
        hc_post_mult_value: 0.75,
        post_mix: vec![0.1, -0.2],
        comb_mix: vec![0.3, 0.4],
        layer_input: vec![0.5, -0.6],
        post_mix_hash: 11,
        comb_mix_hash: 13,
        layer_input_hash: 17,
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
    assert!(json.contains("\"sinkhorn_repeat\":3"));
    assert!(json.contains("\"post_mix\":[0.1,-0.2]"));
    assert!(json.contains("\"comb_mix\":[0.3,0.4]"));
    assert!(json.contains("\"layer_input\":[0.5,-0.6]"));
    assert!(json.contains("\"kernel_launches\":1"));
}

#[test]
fn deepseek_mhc_post_summary_serializes_output() {
    let summary = CudaDeepSeekMhcPostSummary {
        status: SmokeStatus::Ok,
        return_code: 0,
        cuda_error: 0,
        mhc_error: 0,
        tokens: 2,
        hc_mult: 2,
        hidden_size: 3,
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
    assert!(json.contains("\"output\":[0.1,-0.2,0.3]"));
    assert!(json.contains("\"kernel_launches\":1"));
}

#[test]
fn deepseek_mhc_fused_post_pre_summary_serializes_outputs() {
    let summary = CudaDeepSeekMhcFusedPostPreSummary {
        status: SmokeStatus::Ok,
        return_code: 0,
        cuda_error: 0,
        mhc_error: 0,
        tokens: 2,
        hc_mult: 2,
        hidden_size: 3,
        sinkhorn_repeat: 3,
        rms_eps: 1e-5,
        hc_pre_eps: 0.001,
        hc_sinkhorn_eps: 0.0001,
        hc_post_mult_value: 0.75,
        new_residual: vec![0.1, -0.2],
        new_post_mix: vec![0.3, 0.4],
        new_comb_mix: vec![0.5, -0.6],
        layer_input: vec![0.7, -0.8],
        new_residual_hash: 11,
        new_post_mix_hash: 13,
        new_comb_mix_hash: 17,
        layer_input_hash: 19,
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
    assert!(json.contains("\"sinkhorn_repeat\":3"));
    assert!(json.contains("\"new_residual\":[0.100000,-0.200000]"));
    assert!(json.contains("\"new_post_mix\":[0.300000,0.400000]"));
    assert!(json.contains("\"new_comb_mix\":[0.500000,-0.600000]"));
    assert!(json.contains("\"layer_input\":[0.700000,-0.800000]"));
    assert!(json.contains("\"kernel_launches\":1"));
}

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
fn deepseek_mhc_pre_rejects_invalid_shapes_before_cuda() {
    let fixture = deepseek_mhc_pre_fixture();
    let summary = deepseek_mhc_pre(CudaDeepSeekMhcPreInput {
        hc_scale: &fixture.hc_scale[..2],
        ..fixture.input()
    });

    assert_eq!(summary.status, SmokeStatus::Failed);
    assert_eq!(summary.return_code, -1);
    assert_eq!(summary.cuda_error, 0);
    assert_eq!(summary.kernel_launches, 0);
    assert_eq!(summary.sync_calls, 0);
    assert!(summary.post_mix.iter().all(|value| *value == 0.0));
    assert!(summary.comb_mix.iter().all(|value| *value == 0.0));
    assert!(summary.layer_input.iter().all(|value| *value == 0.0));
    assert!(summary.error.as_deref().map_or(false, |error| {
        error.contains("invalid DeepSeek mHC pre shape")
    }));
}

#[test]
fn deepseek_mhc_post_rejects_invalid_shapes_before_cuda() {
    let fixture = deepseek_mhc_pre_fixture();
    let pre = reference_mhc_pre(fixture.input());
    let summary = deepseek_mhc_post(CudaDeepSeekMhcPostInput {
        tokens: fixture.tokens,
        hc_mult: fixture.hc_mult,
        hidden_size: fixture.hidden_size,
        x: &pre.layer_input[..pre.layer_input.len() - 1],
        residual: &fixture.residual,
        post_layer_mix: &pre.post_mix,
        comb_res_mix: &pre.comb_mix,
    });

    assert_eq!(summary.status, SmokeStatus::Failed);
    assert_eq!(summary.return_code, -1);
    assert_eq!(summary.cuda_error, 0);
    assert_eq!(summary.kernel_launches, 0);
    assert_eq!(summary.sync_calls, 0);
    assert!(summary.output.iter().all(|value| *value == 0.0));
    assert!(summary.error.as_deref().map_or(false, |error| {
        error.contains("invalid DeepSeek mHC post shape")
    }));
}

#[test]
fn deepseek_mhc_fused_post_pre_rejects_invalid_shapes_before_cuda() {
    let fixture = deepseek_mhc_pre_fixture();
    let pre = reference_mhc_pre(fixture.input());
    let summary = deepseek_mhc_fused_post_pre(CudaDeepSeekMhcFusedPostPreInput {
        tokens: fixture.tokens,
        hc_mult: fixture.hc_mult,
        hidden_size: fixture.hidden_size,
        sinkhorn_repeat: fixture.sinkhorn_repeat,
        rms_eps: fixture.rms_eps,
        hc_pre_eps: fixture.hc_pre_eps,
        hc_sinkhorn_eps: fixture.hc_sinkhorn_eps,
        hc_post_mult_value: fixture.hc_post_mult_value,
        x: &pre.layer_input,
        residual: &fixture.residual,
        post_layer_mix: &pre.post_mix,
        comb_res_mix: &pre.comb_mix,
        fn_weights: &fixture.fn_weights[..fixture.fn_weights.len() - 1],
        hc_scale: &fixture.hc_scale,
        hc_base: &fixture.hc_base,
    });

    assert_eq!(summary.status, SmokeStatus::Failed);
    assert_eq!(summary.return_code, -1);
    assert_eq!(summary.cuda_error, 0);
    assert_eq!(summary.kernel_launches, 0);
    assert_eq!(summary.sync_calls, 0);
    assert!(summary.new_residual.iter().all(|value| *value == 0.0));
    assert!(summary.new_post_mix.iter().all(|value| *value == 0.0));
    assert!(summary.new_comb_mix.iter().all(|value| *value == 0.0));
    assert!(summary.layer_input.iter().all(|value| *value == 0.0));
    assert!(summary.error.as_deref().map_or(false, |error| {
        error.contains("invalid DeepSeek mHC fused post-pre shape")
    }));
}

#[test]
fn deepseek_mhc_pre_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_mhc_pre_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_mhc_pre_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.tokens, 2);
    assert_eq!(second.hc_mult, 2);
    assert_eq!(second.hidden_size, 3);
    assert_eq!(second.sinkhorn_repeat, 3);
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
    assert_eq!(second.post_mix_hash, first.post_mix_hash);
    assert_eq!(second.comb_mix_hash, first.comb_mix_hash);
    assert_eq!(second.layer_input_hash, first.layer_input_hash);
    assert_eq!(second.post_mix, first.post_mix);
    assert_eq!(second.comb_mix, first.comb_mix);
    assert_eq!(second.layer_input, first.layer_input);
}

#[test]
fn deepseek_mhc_post_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_mhc_post_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_mhc_post_smoke();
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
fn deepseek_mhc_fused_post_pre_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_mhc_fused_post_pre_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_mhc_fused_post_pre_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.tokens, 2);
    assert_eq!(second.hc_mult, 2);
    assert_eq!(second.hidden_size, 3);
    assert_eq!(second.sinkhorn_repeat, 3);
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
    assert_eq!(second.new_residual_hash, first.new_residual_hash);
    assert_eq!(second.new_post_mix_hash, first.new_post_mix_hash);
    assert_eq!(second.new_comb_mix_hash, first.new_comb_mix_hash);
    assert_eq!(second.layer_input_hash, first.layer_input_hash);
    assert_eq!(second.new_residual, first.new_residual);
    assert_eq!(second.new_post_mix, first.new_post_mix);
    assert_eq!(second.new_comb_mix, first.new_comb_mix);
    assert_eq!(second.layer_input, first.layer_input);
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
fn deepseek_mhc_pre_api_matches_vllm_mhc_pre_reference() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let fixture = deepseek_mhc_pre_fixture();
    let summary = deepseek_mhc_pre(fixture.input());
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected = reference_mhc_pre(fixture.input());
    assert_eq!(summary.tokens, 2);
    assert_eq!(summary.hc_mult, 2);
    assert_eq!(summary.hidden_size, 3);
    assert_eq!(summary.sinkhorn_repeat, 3);
    for (actual, expected) in summary.post_mix.iter().zip(expected.post_mix.iter()) {
        assert_close(*actual, *expected, 1e-5);
    }
    for (actual, expected) in summary.comb_mix.iter().zip(expected.comb_mix.iter()) {
        assert_close(*actual, *expected, 1e-5);
    }
    for (actual, expected) in summary.layer_input.iter().zip(expected.layer_input.iter()) {
        assert_close(*actual, *expected, 1e-5);
    }
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.post_mix_hash != 0);
    assert!(summary.comb_mix_hash != 0);
    assert!(summary.layer_input_hash != 0);
}

#[test]
fn deepseek_mhc_post_api_matches_vllm_mhc_post_reference() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let fixture = deepseek_mhc_pre_fixture();
    let pre = reference_mhc_pre(fixture.input());
    let summary = deepseek_mhc_post(CudaDeepSeekMhcPostInput {
        tokens: fixture.tokens,
        hc_mult: fixture.hc_mult,
        hidden_size: fixture.hidden_size,
        x: &pre.layer_input,
        residual: &fixture.residual,
        post_layer_mix: &pre.post_mix,
        comb_res_mix: &pre.comb_mix,
    });
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected = reference_mhc_post(CudaDeepSeekMhcPostInput {
        tokens: fixture.tokens,
        hc_mult: fixture.hc_mult,
        hidden_size: fixture.hidden_size,
        x: &pre.layer_input,
        residual: &fixture.residual,
        post_layer_mix: &pre.post_mix,
        comb_res_mix: &pre.comb_mix,
    });
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

#[test]
fn deepseek_mhc_fused_post_pre_api_matches_vllm_post_then_pre_reference() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let fixture = deepseek_mhc_pre_fixture();
    let pre = reference_mhc_pre(fixture.input());
    let input = CudaDeepSeekMhcFusedPostPreInput {
        tokens: fixture.tokens,
        hc_mult: fixture.hc_mult,
        hidden_size: fixture.hidden_size,
        sinkhorn_repeat: fixture.sinkhorn_repeat,
        rms_eps: fixture.rms_eps,
        hc_pre_eps: fixture.hc_pre_eps,
        hc_sinkhorn_eps: fixture.hc_sinkhorn_eps,
        hc_post_mult_value: fixture.hc_post_mult_value,
        x: &pre.layer_input,
        residual: &fixture.residual,
        post_layer_mix: &pre.post_mix,
        comb_res_mix: &pre.comb_mix,
        fn_weights: &fixture.fn_weights,
        hc_scale: &fixture.hc_scale,
        hc_base: &fixture.hc_base,
    };
    let summary = deepseek_mhc_fused_post_pre(input.clone());
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected = reference_mhc_fused_post_pre(input);
    assert_eq!(summary.tokens, 2);
    assert_eq!(summary.hc_mult, 2);
    assert_eq!(summary.hidden_size, 3);
    assert_eq!(summary.sinkhorn_repeat, 3);
    for (actual, expected) in summary
        .new_residual
        .iter()
        .zip(expected.new_residual.iter())
    {
        assert_close(*actual, *expected, 1e-5);
    }
    for (actual, expected) in summary
        .new_post_mix
        .iter()
        .zip(expected.pre.post_mix.iter())
    {
        assert_close(*actual, *expected, 1e-5);
    }
    for (actual, expected) in summary
        .new_comb_mix
        .iter()
        .zip(expected.pre.comb_mix.iter())
    {
        assert_close(*actual, *expected, 1e-5);
    }
    for (actual, expected) in summary
        .layer_input
        .iter()
        .zip(expected.pre.layer_input.iter())
    {
        assert_close(*actual, *expected, 1e-5);
    }
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.new_residual_hash != 0);
    assert!(summary.new_post_mix_hash != 0);
    assert!(summary.new_comb_mix_hash != 0);
    assert!(summary.layer_input_hash != 0);
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
