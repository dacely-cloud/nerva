use nerva_core::types::memory::tier::MemoryTier;

use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::memory_loop::plan::plan_memory_loop;
use crate::memory_loop::probe::reference_memory_loop_fixture;
use crate::memory_loop::summary::MemoryLoopStatus;
use crate::memory_loop::types::MemoryLoopTaskKind;

#[test]
fn memory_loop_probe_prefetches_stages_evicts_and_prepares_transport() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_memory_loop_probe().unwrap();

    assert_eq!(summary.status, MemoryLoopStatus::Ok);
    assert_eq!(summary.tasks, 5);
    assert_eq!(summary.queue_capacity, 8);
    assert_eq!(summary.max_inflight, 2);
    assert_eq!(summary.queue_overflows, 0);
    assert_eq!(summary.disk_read_tasks, 1);
    assert_eq!(summary.prefetch_tasks, 1);
    assert_eq!(summary.staging_tasks, 1);
    assert_eq!(summary.eviction_tasks, 1);
    assert_eq!(summary.transport_prepare_tasks, 1);
    assert_eq!(summary.ready_blocks, 4);
    assert_eq!(summary.residency_decisions, summary.tasks);
    assert_eq!(summary.pageable_copies, 0);
    assert_eq!(summary.per_token_registrations, 0);
    assert_eq!(summary.page_faults, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.prefetch_events >= 3);
    assert!(summary.eviction_events > 0);
    assert!(summary.copy_events > 0);
    assert!(summary.transport_events > 0);
    assert!(summary.phase_handoff_syncs > 0);
    assert!(summary.total_predicted_visible_ns > summary.actual_visible_ns);
    assert!(summary.passed());
}

#[test]
fn memory_loop_rejects_queue_overflow_before_execution() {
    let (registry, mut config) = reference_memory_loop_fixture().unwrap();
    config.queue_capacity = 2;

    assert!(plan_memory_loop(config, &registry).is_err());
}

#[test]
fn memory_loop_rejects_out_of_order_tier_transitions() {
    let (registry, mut config) = reference_memory_loop_fixture().unwrap();
    let stage = config
        .tasks
        .iter_mut()
        .find(|task| task.kind == MemoryLoopTaskKind::Stage)
        .unwrap();
    stage.from_tier = MemoryTier::Dram;

    assert!(plan_memory_loop(config, &registry).is_err());
}
