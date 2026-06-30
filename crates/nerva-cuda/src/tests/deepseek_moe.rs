use crate::deepseek_moe::forward::{CudaDeepSeekMoeForwardInput, deepseek_moe_forward};
use crate::deepseek_moe::probe::deepseek_moe_smoke;
use crate::deepseek_moe::summary::CudaDeepSeekMoeSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn deepseek_moe_summary_serializes_selected_expert_output() {
    let summary = CudaDeepSeekMoeSummary {
        status: SmokeStatus::Ok,
        hidden_size: 3,
        intermediate_size: 2,
        num_experts: 2,
        top_k: 2,
        swiglu_limit: 1.0,
        expert_ids: [1, 0],
        expert_weights: [0.75, 0.25],
        output: [0.1, -0.2, 0.3],
        output_hash: 44,
        mismatches: 0,
        max_abs_diff: 0.0,
        device_arena_bytes: 12,
        pinned_host_bytes: 12,
        d2h_bytes: 12,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"hidden_size\":3"));
    assert!(json.contains("\"expert_ids\":[1,0]"));
    assert!(json.contains("\"swiglu_limit\":1"));
    assert!(json.contains("\"mismatches\":0"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn deepseek_moe_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_moe_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_moe_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.hidden_size, 3);
    assert_eq!(second.intermediate_size, 2);
    assert_eq!(second.num_experts, 2);
    assert_eq!(second.top_k, 2);
    assert_eq!(second.expert_ids, [1, 0]);
    assert_eq!(second.mismatches, 0);
    assert!(second.max_abs_diff <= 1e-6);
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
    assert_eq!(second.output_hash, first.output_hash);
}

#[test]
fn deepseek_moe_forward_api_matches_reference() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let input = [1.2, -0.7, 0.3];
    let expert_ids = [1, 0];
    let expert_weights = [0.75, 0.25];
    let w_gate = [
        1.0, -0.5, 0.25, -0.25, 0.75, 1.25, 0.5, 0.2, -0.1, -1.0, 0.4, 0.3,
    ];
    let w_up = [
        -0.2, 0.4, 1.1, 0.8, -0.6, 0.2, 1.5, -0.3, 0.1, 0.7, 0.6, -0.4,
    ];
    let w_down = [
        0.3, -0.2, 0.4, 0.1, -0.5, 0.2, -0.7, 0.6, -0.1, 0.25, 0.35, -0.45,
    ];

    let summary = deepseek_moe_forward(CudaDeepSeekMoeForwardInput {
        hidden_size: 3,
        intermediate_size: 2,
        num_experts: 2,
        top_k: 2,
        clamp_swiglu: true,
        swiglu_limit: 1.0,
        input: &input,
        expert_ids: &expert_ids,
        expert_weights: &expert_weights,
        w_gate: &w_gate,
        w_up: &w_up,
        w_down: &w_down,
    });
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected = reference_moe_forward(
        3,
        2,
        2,
        true,
        1.0,
        &input,
        &expert_ids,
        &expert_weights,
        &w_gate,
        &w_up,
        &w_down,
    );
    assert_eq!(summary.hidden_size, 3);
    assert_eq!(summary.intermediate_size, 2);
    assert_eq!(summary.num_experts, 2);
    assert_eq!(summary.top_k, 2);
    assert!(summary.clamp_swiglu);
    for (actual, expected) in summary.output.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "actual={actual} expected={expected}"
        );
    }
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.output_hash != 0);
}

#[allow(clippy::too_many_arguments)]
fn reference_moe_forward(
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    clamp_swiglu: bool,
    swiglu_limit: f32,
    input: &[f32],
    expert_ids: &[u32],
    expert_weights: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; hidden_size];
    for (rank, expert) in expert_ids.iter().copied().enumerate() {
        let expert = expert as usize;
        assert!(expert < num_experts);
        let expert_base = expert * intermediate_size * hidden_size;
        let down_base = expert * hidden_size * intermediate_size;
        let mut activation = vec![0.0f32; intermediate_size];
        for row in 0..intermediate_size {
            let start = expert_base + row * hidden_size;
            let gate = dot(&w_gate[start..start + hidden_size], input);
            let up = dot(&w_up[start..start + hidden_size], input);
            activation[row] = swiglu(gate, up, clamp_swiglu, swiglu_limit);
        }
        for hidden in 0..hidden_size {
            let start = down_base + hidden * intermediate_size;
            output[hidden] +=
                expert_weights[rank] * dot(&w_down[start..start + intermediate_size], &activation);
        }
    }
    output
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn swiglu(gate: f32, up: f32, clamp_swiglu: bool, swiglu_limit: f32) -> f32 {
    let gate = if clamp_swiglu {
        gate.min(swiglu_limit)
    } else {
        gate
    };
    let up = if clamp_swiglu {
        up.clamp(-swiglu_limit, swiglu_limit)
    } else {
        up
    };
    gate / (1.0 + (-gate).exp()) * up
}
