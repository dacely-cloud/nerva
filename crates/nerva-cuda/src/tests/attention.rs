use crate::attention::summary::CudaTieredAttentionSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn tiered_attention_summary_serializes_heterogeneous_fields() {
    let summary = CudaTieredAttentionSummary {
        status: SmokeStatus::Ok,
        hidden: 2,
        heads: 1,
        blocks: 2,
        tokens: 4,
        output: [0.8, 0.2],
        output_hash: 123,
        cpu_block_events: 1,
        device_block_events: 1,
        resident_kv_bytes: 32,
        device_arena_bytes: 64,
        pinned_host_bytes: 64,
        h2d_bytes: 40,
        d2h_bytes: 16,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"blocks\":2"));
    assert!(json.contains("\"tokens\":4"));
    assert!(json.contains("\"output\":[0.8,0.2]"));
    assert!(json.contains("\"cpu_block_events\":1"));
    assert!(json.contains("\"device_block_events\":1"));
    assert!(json.contains("\"resident_kv_bytes\":32"));
    assert!(json.contains("\"H2D_bytes\":40"));
    assert!(json.contains("\"D2H_bytes\":16"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}
