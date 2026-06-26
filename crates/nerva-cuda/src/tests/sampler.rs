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
