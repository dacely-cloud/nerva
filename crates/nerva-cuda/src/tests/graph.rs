use crate::graph::summary::CudaSyntheticGraphSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn synthetic_graph_summary_serializes_token_audit_fields() {
    let summary = CudaSyntheticGraphSummary {
        status: SmokeStatus::Ok,
        steps: 1024,
        ring_capacity: 64,
        seed_token: 1,
        last_token: Some(1025),
        graph_replays: 1024,
        graph_nodes: 2,
        observed_tokens: 1024,
        observed_token_hash: 42,
        token_ring_slots_touched: 64,
        token_ring_reuses: 960,
        token_ring_max_slot_version: 16,
        stale_tokens: 0,
        missing_tokens: 0,
        extra_tokens: 0,
        mismatched_tokens: 0,
        host_causality_edges: 0,
        device_arena_bytes: 4096,
        pinned_host_bytes: 40,
        graph_launches: 1024,
        sync_calls: 1024,
        d2h_bytes: 40960,
        hot_path_allocations: 0,
        error: None,
    };
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"graph_replays\":1024"));
    assert!(json.contains("\"graph_nodes\":2"));
    assert!(json.contains("\"token_ring_reuses\":960"));
    assert!(json.contains("\"host_causality_edges\":0"));
    assert!(json.contains("\"D2H_bytes\":40960"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}
