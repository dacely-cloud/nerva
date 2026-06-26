use nerva_core::types::id::DeviceOrdinal;
use nerva_core::types::memory::MemoryTier;

use crate::types::event::{DeviceTimelineSpan, LedgerEvent, LedgerEventKind};
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
