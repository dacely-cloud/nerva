use nerva_core::types::error::NervaError;
use nerva_core::types::id::token::TokenId;

use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::engine::synthetic::config::SyntheticDecodeConfig;
use crate::engine::synthetic::summary::SyntheticDecodeStatus;

#[test]
fn synthetic_decode_summary_runs_1024_device_first_steps() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_synthetic_decode(SyntheticDecodeConfig::new(1024, 64, TokenId(1)))
        .unwrap();

    assert_eq!(summary.status, SyntheticDecodeStatus::Ok);
    assert_eq!(summary.steps, 1024);
    assert_eq!(summary.last_token, Some(TokenId(1025)));
    assert_eq!(summary.graph_replays, 1024);
    assert_eq!(summary.graph_replay_events, 1024);
    assert_eq!(summary.kernel_events, 2048);
    assert_eq!(summary.device_events, 1024);
    assert_eq!(summary.copy_events, 1024);
    assert_eq!(summary.host_wait_events, 1024);
    assert_eq!(summary.soft_visibility_syncs, 1024);
    assert_eq!(summary.device_timeline_active_ns, 3072);
    assert_eq!(summary.device_timeline_idle_ns, 0);
    assert_eq!(summary.graph_replay_latency_ns, 1024);
    assert_eq!(summary.device_latency_ns, 3072);
    assert_eq!(summary.copy_latency_ns, 1024);
    assert_eq!(summary.host_wait_latency_ns, 1024);
    assert_eq!(summary.soft_visibility_sync_latency_ns, 1024);
    assert_eq!(summary.estimated_events, 4096);
    assert_eq!(summary.estimated_latency_ns, summary.total_latency_ns);
    assert_eq!(summary.total_latency_ns, 6144);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(summary.observed_tokens, 1024);
    assert_ne!(summary.observed_token_hash, 0);
    assert_eq!(summary.token_ring_slots_touched, 64);
    assert_eq!(summary.token_ring_reuses, 960);
    assert_eq!(summary.token_ring_max_slot_version, 16);
    assert_eq!(summary.stale_tokens, 0);
    assert_eq!(summary.missing_tokens, 0);
    assert_eq!(summary.extra_tokens, 0);
    assert_eq!(summary.mismatched_tokens, 0);
    assert_eq!(summary.host_causality_edges, 0);
    assert!(summary.to_json().contains("\"steps\":1024"));
    assert!(summary.to_json().contains("\"observed_token_hash\""));
    assert!(summary.to_json().contains("\"token_ring_reuses\":960"));
    assert!(summary.to_json().contains("\"host_causality_edges\":0"));
}

#[test]
fn synthetic_decode_rejects_zero_steps() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let err = runtime
        .run_synthetic_decode(SyntheticDecodeConfig::new(0, 64, TokenId(1)))
        .unwrap_err();
    assert!(matches!(err, NervaError::InvalidArgument { .. }));
}
