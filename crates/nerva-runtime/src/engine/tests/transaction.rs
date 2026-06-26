use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::id::device::DeviceOrdinal;

use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::execution::plan::planner::plan_execution_transaction;
use crate::execution::probe::reference_transaction_fixture;
use crate::execution::summary::ExecutionTransactionStatus;

#[test]
fn execution_transaction_probe_records_dependencies_and_sync_classes() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_execution_transaction_probe().unwrap();

    assert_eq!(summary.status, ExecutionTransactionStatus::Ok);
    assert_eq!(summary.operations, 4);
    assert_eq!(summary.graph_capturable_operations, 3);
    assert_eq!(summary.gpu_operations, 3);
    assert_eq!(summary.cpu_operations, 1);
    assert_eq!(summary.execution_decisions, summary.operations);
    assert_eq!(summary.block_version_dependencies, summary.block_uses);
    assert_eq!(summary.hard_syncs, 3);
    assert_eq!(summary.soft_visibility_syncs, 1);
    assert_eq!(summary.phase_handoff_syncs, 2);
    assert_eq!(summary.debug_syncs, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(summary.stale_dependencies, 0);
    assert_eq!(summary.unclassified_syncs, 0);
    assert!(summary.device_active_ns > 0);
    assert!(summary.host_event_wait_ns > 0);
    assert!(summary.passed());
}

#[test]
fn execution_transaction_rejects_stale_block_versions() {
    let (registry, mut spec, blocks) = reference_transaction_fixture(DeviceOrdinal(0)).unwrap();
    spec.operations[0].block_uses[0].required_version = 2;

    assert!(plan_execution_transaction(spec, &registry).is_err());
    assert!(registry.block(blocks.device_token).is_some());
}

#[test]
fn execution_transaction_rejects_non_ready_blocks() {
    let (mut registry, spec, blocks) = reference_transaction_fixture(DeviceOrdinal(0)).unwrap();
    registry
        .transition(blocks.kv_page, ResidencyState::Prefetching)
        .unwrap();

    assert!(plan_execution_transaction(spec, &registry).is_err());
}
