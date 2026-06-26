use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;
use nerva_ledger::types::sync::SyncClass;

use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::token::ring::TokenInputSource;

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
