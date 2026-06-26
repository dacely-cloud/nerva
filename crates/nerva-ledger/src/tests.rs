use nerva_core::types::id::{DeviceOrdinal, ResidentBlockId};
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;

use crate::types::decision::{
    BlockVersionDependency, CandidateCost, ExecutionDecision, ResidencyDecision,
};
use crate::types::event::{DeviceTimelineSpan, LedgerEvent, LedgerEventKind};
use crate::types::fallback::{FallbackClass, FallbackDecision};
use crate::types::metric::MetricSource;
use crate::types::sync::SyncClass;
use crate::types::token::TokenLedger;

#[test]
fn allocation_events_increment_hot_path_count() {
    let mut ledger = TokenLedger::new(0);
    ledger.record_hot_path_allocation_attempt("test", 64, MemoryTier::Vram);
    assert_eq!(ledger.hot_path_allocations, 1);
    assert_eq!(ledger.total_latency_ns(), 0);
    assert_eq!(ledger.event_count(LedgerEventKind::Allocation), 1);
    assert!(ledger.require_zero_hot_path_allocations().is_err());
}

#[test]
fn ledger_keeps_host_wait_and_device_activity_separate() {
    let mut ledger = TokenLedger::new(5);
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::GraphReplay,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: None,
        to_tier: Some(MemoryTier::Vram),
        bytes: 0,
        latency_ns: 2,
        label: "graph",
    });
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::DeviceActivity,
        sync_class: None,
        metric_source: MetricSource::GpuEvent,
        block_id: None,
        from_tier: None,
        to_tier: Some(MemoryTier::Vram),
        bytes: 0,
        latency_ns: 7,
        label: "device",
    });
    ledger.record_sync(
        SyncClass::SoftVisibilitySync,
        None,
        Some(MemoryTier::Vram),
        Some(MemoryTier::PinnedDram),
        0,
        3,
        MetricSource::EstimatedModel,
        "soft_visibility_host_wait",
    );

    assert_eq!(ledger.event_count(LedgerEventKind::GraphReplay), 1);
    assert_eq!(ledger.event_count(LedgerEventKind::DeviceActivity), 1);
    assert_eq!(ledger.event_count(LedgerEventKind::Sync), 1);
    assert_eq!(ledger.sync_count_for(SyncClass::SoftVisibilitySync), 1);
    assert_eq!(ledger.latency_ns_for(LedgerEventKind::DeviceActivity), 7);
    assert_eq!(ledger.latency_ns_for(LedgerEventKind::Sync), 3);
    assert_eq!(ledger.sync_latency_ns_for(SyncClass::SoftVisibilitySync), 3);
    assert_eq!(ledger.event_count_for_source(MetricSource::GpuEvent), 1);
    assert_eq!(
        ledger.latency_ns_for_source(MetricSource::EstimatedModel),
        5
    );
    ledger
        .record_device_span(DeviceTimelineSpan::new(
            DeviceOrdinal(0),
            0,
            7,
            MetricSource::GpuEvent,
            "device_active",
        ))
        .unwrap();
    assert_eq!(ledger.device_active_ns(DeviceOrdinal(0)).unwrap(), 7);
    assert_eq!(ledger.device_idle_ns(DeviceOrdinal(0)).unwrap(), 0);
    assert_eq!(ledger.total_latency_ns(), 12);
    assert!(ledger.require_classified_syncs().is_ok());
}

#[test]
fn token_ledger_serializes_raw_events_and_decisions() {
    let mut ledger = TokenLedger::new(12);
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::GraphReplay,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(ResidentBlockId(3)),
        from_tier: None,
        to_tier: Some(MemoryTier::Vram),
        bytes: 0,
        latency_ns: 4,
        label: "graph\"replay",
    });
    ledger.record_sync(
        SyncClass::SoftVisibilitySync,
        Some(ResidentBlockId(3)),
        Some(MemoryTier::Vram),
        Some(MemoryTier::PinnedDram),
        4,
        2,
        MetricSource::RuntimeTimestamp,
        "host\nwait",
    );
    ledger
        .record_device_span(DeviceTimelineSpan::new(
            DeviceOrdinal(0),
            0,
            4,
            MetricSource::GpuEvent,
            "device",
        ))
        .unwrap();
    ledger.record_fallback_decision(FallbackDecision {
        label: "fallback",
        class: FallbackClass::ExactNamed,
        requested: "cuda",
        selected: "cpu",
        reason: "test",
        visible_ns: Some(8),
        metric_source: MetricSource::EstimatedModel,
    });
    ledger.record_block_version_dependency(BlockVersionDependency {
        block_id: ResidentBlockId(3),
        required_version: 2,
        observed_version: 2,
        label: "version",
    });
    ledger.record_residency_decision(ResidencyDecision {
        block_id: ResidentBlockId(3),
        old_tier: MemoryTier::Dram,
        new_tier: MemoryTier::Vram,
        executor_selected: ExecutionOwner::Gpu(DeviceOrdinal(0)),
        candidate_costs: vec![CandidateCost::estimated("copy", 11)],
        reason: "prefetch",
        predicted_overlap_ns: 5,
        actual_visible_ns: Some(6),
        metric_source: MetricSource::EstimatedModel,
    });
    ledger.record_execution_decision(ExecutionDecision {
        operation: "matvec",
        executor_selected: ExecutionOwner::Cpu,
        candidate_costs: vec![CandidateCost::measured("cpu", 7)],
        reason: "near_data",
        predicted_visible_ns: 7,
        actual_visible_ns: Some(7),
        metric_source: MetricSource::RuntimeTimestamp,
    });

    let json = ledger.to_json();
    assert!(json.contains("\"token_index\":12"));
    assert!(json.contains("\"kind\":\"graph_replay\""));
    assert!(json.contains("graph\\\"replay"));
    assert!(json.contains("host\\nwait"));
    assert!(json.contains("\"sync_class\":\"soft_visibility_sync\""));
    assert!(json.contains("\"device_timeline\""));
    assert!(json.contains("\"fallback_decisions\""));
    assert!(json.contains("\"block_version_dependencies\""));
    assert!(json.contains("\"residency_decisions\""));
    assert!(json.contains("\"execution_decisions\""));
    assert!(json.contains("\"executor_selected\":\"gpu:0\""));
    assert!(json.contains("\"executor_selected\":\"cpu\""));
}

#[test]
fn device_idle_is_derived_from_device_timeline_gaps() {
    let mut ledger = TokenLedger::new(0);
    ledger
        .record_device_span(DeviceTimelineSpan::new(
            DeviceOrdinal(0),
            0,
            10,
            MetricSource::GpuEvent,
            "kernel_a",
        ))
        .unwrap();
    ledger
        .record_device_span(DeviceTimelineSpan::new(
            DeviceOrdinal(0),
            15,
            25,
            MetricSource::GpuEvent,
            "kernel_b",
        ))
        .unwrap();
    ledger
        .record_device_span(DeviceTimelineSpan::new(
            DeviceOrdinal(0),
            20,
            30,
            MetricSource::GpuEvent,
            "overlap_kernel",
        ))
        .unwrap();
    ledger.record_sync(
        SyncClass::SoftVisibilitySync,
        None,
        Some(MemoryTier::Vram),
        Some(MemoryTier::PinnedDram),
        0,
        100,
        MetricSource::RuntimeTimestamp,
        "host_wait_not_gpu_idle",
    );

    assert_eq!(ledger.latency_ns_for(LedgerEventKind::Sync), 100);
    assert_eq!(ledger.device_active_ns(DeviceOrdinal(0)).unwrap(), 25);
    assert_eq!(ledger.device_idle_ns(DeviceOrdinal(0)).unwrap(), 5);
}

#[test]
fn device_timeline_rejects_invalid_spans() {
    let mut ledger = TokenLedger::new(0);
    let result = ledger.record_device_span(DeviceTimelineSpan::new(
        DeviceOrdinal(0),
        10,
        9,
        MetricSource::GpuEvent,
        "bad_span",
    ));

    assert!(result.is_err());
}

#[test]
fn fallback_decisions_are_recorded_separately_from_events() {
    let mut ledger = TokenLedger::new(0);
    ledger.record_fallback_decision(FallbackDecision {
        label: "cpu_reference_fallback",
        class: FallbackClass::ExactNamed,
        requested: "cuda_dense_matvec_f16",
        selected: "cpu_reference_dense_matvec_f32",
        reason: "declared exact fallback",
        visible_ns: Some(12),
        metric_source: MetricSource::EstimatedModel,
    });
    ledger.record_fallback_decision(FallbackDecision {
        label: "host_staged_transport",
        class: FallbackClass::CapabilityDegraded,
        requested: "gpu_direct_rdma",
        selected: "pinned_host_bounce",
        reason: "direct path unverified",
        visible_ns: Some(7),
        metric_source: MetricSource::EstimatedModel,
    });

    assert_eq!(ledger.events.len(), 0);
    assert_eq!(ledger.fallback_count(), 2);
    assert_eq!(ledger.fallback_count_for(FallbackClass::ExactNamed), 1);
    assert_eq!(
        ledger.fallback_count_for(FallbackClass::CapabilityDegraded),
        1
    );
}

#[test]
fn block_version_dependencies_validate_observed_versions() {
    let mut ledger = TokenLedger::new(0);
    ledger.record_block_version_dependency(BlockVersionDependency {
        block_id: ResidentBlockId(7),
        required_version: 2,
        observed_version: 2,
        label: "weight_step",
    });
    ledger.record_block_version_dependency(BlockVersionDependency {
        block_id: ResidentBlockId(8),
        required_version: 2,
        observed_version: 3,
        label: "newer_replica",
    });

    assert_eq!(ledger.block_version_dependencies.len(), 2);
    assert!(ledger.require_satisfied_block_versions().is_ok());
}

#[test]
fn block_version_dependencies_reject_stale_observations() {
    let mut ledger = TokenLedger::new(0);
    ledger.record_block_version_dependency(BlockVersionDependency {
        block_id: ResidentBlockId(7),
        required_version: 4,
        observed_version: 3,
        label: "stale_weight_step",
    });

    assert!(ledger.require_satisfied_block_versions().is_err());
}

#[test]
fn classified_sync_validation_rejects_missing_or_misplaced_classes() {
    let mut missing = TokenLedger::new(0);
    missing.record(LedgerEvent {
        kind: LedgerEventKind::Sync,
        sync_class: None,
        metric_source: MetricSource::RuntimeTimestamp,
        block_id: None,
        from_tier: None,
        to_tier: None,
        bytes: 0,
        latency_ns: 1,
        label: "unclassified_wait",
    });
    assert!(missing.require_classified_syncs().is_err());

    let mut misplaced = TokenLedger::new(1);
    misplaced.record(LedgerEvent {
        kind: LedgerEventKind::Copy,
        sync_class: Some(SyncClass::HardSync),
        metric_source: MetricSource::RuntimeTimestamp,
        block_id: None,
        from_tier: Some(MemoryTier::Dram),
        to_tier: Some(MemoryTier::Vram),
        bytes: 4,
        latency_ns: 1,
        label: "copy_with_sync_class",
    });
    assert!(misplaced.require_classified_syncs().is_err());
}

#[test]
fn residency_decisions_are_recorded_separately_from_timing_events() {
    let mut ledger = TokenLedger::new(3);
    ledger.record_residency_decision(ResidencyDecision {
        block_id: ResidentBlockId(9),
        old_tier: MemoryTier::Dram,
        new_tier: MemoryTier::Vram,
        executor_selected: ExecutionOwner::Cpu,
        candidate_costs: vec![
            CandidateCost::estimated("cpu-dram", 100),
            CandidateCost::estimated("gpu-prefetch", 80),
        ],
        reason: "prefetch hides transfer",
        predicted_overlap_ns: 40,
        actual_visible_ns: None,
        metric_source: MetricSource::EstimatedModel,
    });

    assert_eq!(ledger.events.len(), 0);
    assert_eq!(ledger.residency_decisions.len(), 1);
    assert_eq!(
        ledger.residency_decisions[0].candidate_costs[1].label,
        "gpu-prefetch"
    );
    assert!(ledger.require_zero_hot_path_allocations().is_ok());
}

#[test]
fn execution_decisions_record_operation_placement() {
    let mut ledger = TokenLedger::new(8);
    ledger.record_execution_decision(ExecutionDecision {
        operation: "matvec",
        executor_selected: ExecutionOwner::Cpu,
        candidate_costs: vec![
            CandidateCost::estimated("cpu-dram", 16),
            CandidateCost::estimated("gpu-staged", 68),
        ],
        reason: "compute near warm DRAM weights",
        predicted_visible_ns: 16,
        actual_visible_ns: Some(16),
        metric_source: MetricSource::EstimatedModel,
    });

    assert_eq!(ledger.events.len(), 0);
    assert_eq!(ledger.execution_decisions.len(), 1);
    assert_eq!(ledger.execution_decisions[0].operation, "matvec");
    assert_eq!(
        ledger.execution_decisions[0].candidate_costs[0].label,
        "cpu-dram"
    );
    assert!(ledger.require_zero_hot_path_allocations().is_ok());
}
