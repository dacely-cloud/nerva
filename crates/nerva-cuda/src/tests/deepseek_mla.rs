use crate::deepseek_mla::probe::deepseek_mla_smoke;
use crate::deepseek_mla::summary::CudaDeepSeekMlaSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn deepseek_mla_summary_serializes_shape_and_output() {
    let summary = CudaDeepSeekMlaSummary {
        status: SmokeStatus::Ok,
        heads: 2,
        tokens: 3,
        kv_lora_rank: 3,
        qk_nope_head_dim: 2,
        qk_rope_head_dim: 1,
        v_head_dim: 2,
        softmax_scale: 0.7,
        output: [0.1, -0.2, 0.3, -0.4],
        output_hash: 11,
        mismatches: 0,
        max_abs_diff: 0.0,
        device_arena_bytes: 16,
        pinned_host_bytes: 16,
        d2h_bytes: 16,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"heads\":2"));
    assert!(json.contains("\"tokens\":3"));
    assert!(json.contains("\"kv_lora_rank\":3"));
    assert!(json.contains("\"qk_nope_head_dim\":2"));
    assert!(json.contains("\"qk_rope_head_dim\":1"));
    assert!(json.contains("\"v_head_dim\":2"));
    assert!(json.contains("\"output\":[0.1,-0.2,0.3,-0.4]"));
    assert!(json.contains("\"mismatches\":0"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn deepseek_mla_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_mla_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_mla_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.heads, 2);
    assert_eq!(second.tokens, 3);
    assert_eq!(second.kv_lora_rank, 3);
    assert_eq!(second.qk_nope_head_dim, 2);
    assert_eq!(second.qk_rope_head_dim, 1);
    assert_eq!(second.v_head_dim, 2);
    assert_eq!(second.mismatches, 0);
    assert!(second.max_abs_diff <= 1e-6);
    assert!(second.output.iter().all(|value| value.is_finite()));
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
    assert_eq!(second.output_hash, first.output_hash);
}
