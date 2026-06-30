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
