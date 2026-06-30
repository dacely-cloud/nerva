use crate::deepseek_router::probe::deepseek_router_smoke;
use crate::deepseek_router::route::{
    deepseek_router_route_v3_grouped_sigmoid, deepseek_router_route_v4_hash,
    deepseek_router_route_v4_sqrtsoftplus,
};
use crate::deepseek_router::summary::CudaDeepSeekRouterSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn deepseek_router_summary_serializes_routes_and_mismatches() {
    let summary = CudaDeepSeekRouterSummary {
        status: SmokeStatus::Ok,
        v3_num_experts: 8,
        v3_num_groups: 2,
        v3_top_k_groups: 1,
        v3_top_k: 2,
        v4_num_experts: 4,
        v4_top_k: 2,
        v4_hash_top_k: 3,
        v3_expert_ids: [3, 2],
        v4_expert_ids: [1, 2],
        v4_hash_expert_ids: [2, 1, 3],
        v3_weights: [0.67, 1.83],
        v4_weights: [0.63, 0.87],
        v4_hash_weights: [0.27, 0.18, 0.55],
        v3_output_hash: 11,
        v4_output_hash: 22,
        v4_hash_output_hash: 33,
        v3_mismatches: 0,
        v4_mismatches: 0,
        v4_hash_mismatches: 0,
        v3_max_abs_diff: 0.0,
        v4_max_abs_diff: 0.0,
        v4_hash_max_abs_diff: 0.0,
        device_arena_bytes: 96,
        pinned_host_bytes: 96,
        d2h_bytes: 96,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"v3_num_groups\":2"));
    assert!(json.contains("\"v3_expert_ids\":[3,2]"));
    assert!(json.contains("\"v4_expert_ids\":[1,2]"));
    assert!(json.contains("\"v4_hash_expert_ids\":[2,1,3]"));
    assert!(json.contains("\"v3_mismatches\":0"));
    assert!(json.contains("\"v4_hash_mismatches\":0"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn deepseek_router_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_router_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_router_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.v3_expert_ids, [3, 2]);
    assert_eq!(second.v4_expert_ids, [1, 2]);
    assert_eq!(second.v4_hash_expert_ids, [2, 1, 3]);
    assert_eq!(second.v3_mismatches, 0);
    assert_eq!(second.v4_mismatches, 0);
    assert_eq!(second.v4_hash_mismatches, 0);
    assert!(second.v3_max_abs_diff <= 1e-6);
    assert!(second.v4_max_abs_diff <= 1e-6);
    assert!(second.v4_hash_max_abs_diff <= 1e-6);
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
    assert_eq!(second.v3_output_hash, first.v3_output_hash);
    assert_eq!(second.v4_output_hash, first.v4_output_hash);
    assert_eq!(second.v4_hash_output_hash, first.v4_hash_output_hash);
}

#[test]
fn deepseek_router_route_api_matches_v3_v4_and_hash_reference() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let v3 = deepseek_router_route_v3_grouped_sigmoid(
        &[-2.0, 0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -3.0],
        Some(&[0.0, 0.0, 0.0, 4.0, 0.0, 0.0, -4.0, 0.0]),
        2,
        1,
        2,
        true,
        2.5,
    );
    if v3.status != SmokeStatus::Ok {
        return;
    }
    assert_eq!(v3.expert_ids, [3, 2]);
    assert_close(
        v3.weights[0],
        sigmoid(-1.0) * 2.5 / (sigmoid(-1.0) + sigmoid(1.0)),
    );
    assert_close(
        v3.weights[1],
        sigmoid(1.0) * 2.5 / (sigmoid(-1.0) + sigmoid(1.0)),
    );
    assert_eq!(v3.kernel_launches, 1);
    assert_eq!(v3.sync_calls, 1);
    assert_eq!(v3.hot_path_allocations, 0);
    assert!(v3.output_hash != 0);

    let v4 = deepseek_router_route_v4_sqrtsoftplus(
        &[-2.0, 0.0, 1.0, 3.0],
        Some(&[0.0, 3.0, 0.0, -3.0]),
        2,
        true,
        1.5,
    );
    assert_eq!(v4.status, SmokeStatus::Ok, "v4 route: {v4:?}");
    assert_eq!(v4.expert_ids, [1, 2]);
    let v4_w1 = sqrt_softplus(0.0);
    let v4_w2 = sqrt_softplus(1.0);
    assert_close(v4.weights[0], v4_w1 * 1.5 / (v4_w1 + v4_w2));
    assert_close(v4.weights[1], v4_w2 * 1.5 / (v4_w1 + v4_w2));
    assert_eq!(v4.kernel_launches, 1);
    assert_eq!(v4.sync_calls, 1);
    assert!(v4.output_hash != 0);

    let hash_table = [
        0u32, 1, 3, // token 0
        2, 1, 3, // token 1
        3, 0, 2, // token 2
    ];
    let v4_hash =
        deepseek_router_route_v4_hash(&[4.0, -1.0, 0.0, 2.0], &hash_table, 1, 3, true, 1.0);
    assert_eq!(
        v4_hash.status,
        SmokeStatus::Ok,
        "v4 hash route: {v4_hash:?}"
    );
    assert_eq!(v4_hash.expert_ids, [2, 1, 3]);
    let expected = [sqrt_softplus(0.0), sqrt_softplus(-1.0), sqrt_softplus(2.0)];
    let sum = expected.iter().sum::<f32>();
    for (actual, expected) in v4_hash.weights.iter().zip(expected) {
        assert_close(*actual, expected / sum);
    }
    assert_eq!(v4_hash.kernel_launches, 1);
    assert_eq!(v4_hash.sync_calls, 1);
    assert!(v4_hash.output_hash != 0);
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn sqrt_softplus(value: f32) -> f32 {
    value.exp().ln_1p().sqrt()
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 1e-6,
        "actual={actual} expected={expected}"
    );
}
