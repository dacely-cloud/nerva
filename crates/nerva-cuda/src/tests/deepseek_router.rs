use crate::deepseek_router::probe::deepseek_router_smoke;
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
