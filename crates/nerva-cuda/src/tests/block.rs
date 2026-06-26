use crate::block::summary::{CudaLoadedTinyBlockSummary, CudaTinyBlockSummary};
use crate::smoke::status::SmokeStatus;

#[test]
fn tiny_block_summary_serializes_device_block_fields() {
    let summary = CudaTinyBlockSummary {
        status: SmokeStatus::Ok,
        hidden: 2,
        intermediate: 2,
        output: [15_360, 16_384],
        output_hash: 99,
        device_arena_bytes: 4,
        pinned_host_bytes: 4,
        kernel_launches: 1,
        sync_calls: 1,
        d2h_bytes: 4,
        hot_path_allocations: 0,
        error: None,
    };
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"hidden\":2"));
    assert!(json.contains("\"output_bits\":[15360,16384]"));
    assert!(json.contains("\"kernel_launches\":1"));
    assert!(json.contains("\"D2H_bytes\":4"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn loaded_tiny_block_summary_serializes_residency_fields() {
    let summary = CudaLoadedTinyBlockSummary {
        status: SmokeStatus::Ok,
        hidden: 2,
        intermediate: 2,
        output: [16_126, 17_299],
        output_hash: 17766510782028265595,
        resident_weight_bytes: 64,
        device_arena_bytes: 72,
        pinned_host_bytes: 72,
        h2d_bytes: 72,
        d2h_bytes: 4,
        kernel_launches: 1,
        sync_calls: 2,
        hot_path_allocations: 0,
        error: None,
    };
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"resident_weight_bytes\":64"));
    assert!(json.contains("\"H2D_bytes\":72"));
    assert!(json.contains("\"D2H_bytes\":4"));
    assert!(json.contains("\"kernel_launches\":1"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}
