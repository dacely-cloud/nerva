use std::os::raw::c_char;

use crate::block::{CudaLoadedTinyBlockSummary, CudaTinyBlockSummary};
use crate::decode::CudaTinyDecodeSummary;
use crate::graph::CudaSyntheticGraphSummary;
use crate::sampler::CudaGreedySamplerSummary;
use crate::smoke::{CudaSmokeSummary, SmokeStatus, c_char_array_to_string, escape_json};

#[test]
fn json_escapes_control_chars() {
    assert_eq!(escape_json("a\"b\\c\n"), "a\\\"b\\\\c\\n");
}

#[test]
fn unavailable_summary_is_valid_shape() {
    let summary = CudaSmokeSummary::unavailable("no cuda", Some(13_010));
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"unavailable\""));
    assert!(json.contains("\"runtime_version\":13010"));
    assert!(json.contains("\"compute_capability_major\":null"));
    assert!(json.contains("\"compute_capability_minor\":null"));
    assert!(json.contains("\"device_total_memory_bytes\":null"));
    assert!(json.contains("\"pci_bus_id\":null"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn c_char_array_conversion_handles_empty_and_terminated_values() {
    let empty = [0 as c_char; 8];
    assert_eq!(c_char_array_to_string(&empty), None);

    let mut value = [0 as c_char; 8];
    value[0] = b'R' as c_char;
    value[1] = b'T' as c_char;
    value[2] = b'X' as c_char;
    assert_eq!(c_char_array_to_string(&value).as_deref(), Some("RTX"));
}

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

#[test]
fn tiny_decode_summary_serializes_device_first_fields() {
    let summary = CudaTinyDecodeSummary {
        status: SmokeStatus::Ok,
        steps: 8,
        ring_capacity: 4,
        seed_token: 0,
        vocab_size: 4,
        hidden: 2,
        last_token: Some(0),
        graph_replays: 8,
        graph_nodes: 2,
        observed_tokens: 8,
        observed_token_hash: 761644941098537893,
        token_ring_slots_touched: 4,
        token_ring_reuses: 4,
        token_ring_max_slot_version: 2,
        stale_tokens: 0,
        missing_tokens: 0,
        extra_tokens: 0,
        mismatched_tokens: 0,
        host_causality_edges: 0,
        resident_weight_bytes: 64,
        device_arena_bytes: 256,
        pinned_host_bytes: 104,
        h2d_bytes: 80,
        d2h_bytes: 320,
        graph_launches: 8,
        sync_calls: 8,
        kernel_launches: 8,
        hot_path_allocations: 0,
        error: None,
    };
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"steps\":8"));
    assert!(json.contains("\"last_token\":0"));
    assert!(json.contains("\"observed_token_hash\":761644941098537893"));
    assert!(json.contains("\"token_ring_reuses\":4"));
    assert!(json.contains("\"host_causality_edges\":0"));
    assert!(json.contains("\"resident_weight_bytes\":64"));
    assert!(json.contains("\"H2D_bytes\":80"));
    assert!(json.contains("\"D2H_bytes\":320"));
    assert!(json.contains("\"kernel_launches\":8"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}
