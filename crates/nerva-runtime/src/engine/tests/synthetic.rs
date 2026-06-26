use nerva_core::types::error::NervaError;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;

use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::engine::synthetic::config::SyntheticDecodeConfig;
use crate::engine::synthetic::summary::SyntheticDecodeStatus;
use crate::graph::layout::GraphKey;
use crate::token::ring::{DeviceTokenRef, DeviceTokenRing, TokenInputSource};

#[test]
fn synthetic_launch_collect_records_token_and_ledger() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut engine = runtime.synthetic_engine(4).unwrap();

    let step = engine
        .launch(RequestId(1), SequenceId(1), 0, TokenId(41))
        .unwrap();
    let output = step.collect().unwrap();

    assert_eq!(output.token, TokenId(42));
    assert_eq!(output.input_source, TokenInputSource::Seed);
    assert_eq!(output.device_token_ref.token_index, 0);
    assert_eq!(output.ledger.hot_path_allocations, 0);
    assert_eq!(output.ledger.events.len(), 4);
    assert_eq!(output.ledger.event_count(LedgerEventKind::GraphReplay), 1);
    assert_eq!(
        output.ledger.event_count(LedgerEventKind::DeviceActivity),
        1
    );
    assert_eq!(output.ledger.event_count(LedgerEventKind::Copy), 1);
    assert_eq!(output.ledger.event_count(LedgerEventKind::Sync), 1);
    assert_eq!(
        output.ledger.sync_count_for(SyncClass::SoftVisibilitySync),
        1
    );
    assert_eq!(output.ledger.device_active_ns(DeviceOrdinal(0)).unwrap(), 3);
    assert_eq!(output.ledger.device_idle_ns(DeviceOrdinal(0)).unwrap(), 0);
    assert!(output.ledger.require_classified_syncs().is_ok());
    assert_eq!(
        engine
            .token_ring()
            .consume_device_input(RequestId(1), SequenceId(1), 0)
            .unwrap(),
        TokenId(42)
    );
    assert_eq!(
        engine
            .graph_pool()
            .replay_count(GraphKey {
                bucket: 1,
                max_blocks: 1
            })
            .unwrap(),
        1
    );
}

#[test]
fn synthetic_next_step_must_use_device_token() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut engine = runtime.synthetic_engine(4).unwrap();

    let output = engine
        .launch(RequestId(2), SequenceId(9), 0, TokenId(10))
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(output.token, TokenId(11));

    let err = engine
        .launch(RequestId(2), SequenceId(9), 1, TokenId(99))
        .unwrap_err();
    assert!(matches!(err, NervaError::ResidencyViolation { .. }));

    let output = engine
        .launch_device_next(RequestId(2), SequenceId(9), 1, TokenId(10))
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(output.token, TokenId(12));
    assert!(matches!(
        output.input_source,
        TokenInputSource::DeviceRing(DeviceTokenRef { token_index: 0, .. })
    ));
}

#[test]
fn synthetic_host_policy_path_is_explicit_policy_sync() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut engine = runtime.synthetic_engine(4).unwrap();

    let first = engine
        .launch_device_next(RequestId(3), SequenceId(1), 0, TokenId(20))
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(first.token, TokenId(21));

    let output = engine
        .launch_host_policy_next(RequestId(3), SequenceId(1), 1, first.token)
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(output.token, TokenId(22));
    assert_eq!(output.input_source, TokenInputSource::HostObservation);
    assert_eq!(
        output.ledger.sync_count_for(SyncClass::SoftVisibilitySync),
        1
    );
    assert_eq!(output.ledger.sync_count_for(SyncClass::PolicySync), 1);
    assert!(output.ledger.require_classified_syncs().is_ok());
}

#[test]
fn token_policy_probe_reports_policy_barrier_without_fast_path_host_dependency() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_token_policy_probe().unwrap();

    assert!(summary.passed());
    assert_eq!(summary.steps, 6);
    assert_eq!(summary.device_fast_steps, 3);
    assert_eq!(summary.host_policy_steps, 1);
    assert_eq!(summary.hybrid_validation_steps, 2);
    assert_eq!(summary.seed_edges, 1);
    assert_eq!(summary.device_ring_edges, 4);
    assert_eq!(summary.host_causality_edges, 1);
    assert_eq!(summary.policy_syncs, 1);
    assert_eq!(summary.soft_visibility_syncs, 6);
    assert_eq!(summary.host_visibility_hard_dependencies, 1);
    assert_eq!(summary.device_fast_host_dependencies, 0);
    assert_eq!(summary.graph_replays, 6);
    assert_eq!(summary.observed_tokens, 6);
    assert_eq!(summary.mismatched_tokens, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"policy_syncs\":1"));
    assert!(
        summary
            .to_json()
            .contains("\"device_fast_host_dependencies\":0")
    );
}

#[test]
fn device_token_ring_rejects_stale_reads() {
    let mut ring = DeviceTokenRing::new(2).unwrap();
    ring.publish(RequestId(1), SequenceId(1), 0, TokenId(7))
        .unwrap();
    assert!(
        ring.consume_device_input(RequestId(1), SequenceId(2), 0)
            .is_err()
    );
    assert_eq!(
        ring.consume_device_input(RequestId(1), SequenceId(1), 0)
            .unwrap(),
        TokenId(7)
    );
}

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
