use nerva_core::types::id::DeviceOrdinal;
use nerva_core::types::memory::MemoryTier;

use crate::types::event::{DeviceTimelineSpan, LedgerEvent, LedgerEventKind};
use crate::types::metric::MetricSource;
use crate::types::sync::SyncClass;
use crate::types::token::critical::TokenCriticalPathReport;
use crate::types::token::ledger::TokenLedger;

#[test]
fn critical_path_report_separates_host_wait_from_gpu_idle() {
    let mut ledger = TokenLedger::new(4);
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
        latency_ns: 8,
        label: "device",
    });
    ledger
        .record_device_span(DeviceTimelineSpan::new(
            DeviceOrdinal(0),
            0,
            8,
            MetricSource::GpuEvent,
            "kernel",
        ))
        .unwrap();
    ledger.record_sync(
        SyncClass::SoftVisibilitySync,
        None,
        Some(MemoryTier::Vram),
        Some(MemoryTier::PinnedDram),
        0,
        5,
        MetricSource::RuntimeTimestamp,
        "host_wait",
    );

    let report = TokenCriticalPathReport::from_ledger(&ledger, DeviceOrdinal(0)).unwrap();

    assert_eq!(report.token_index, 4);
    assert_eq!(report.host_event_wait_ns, 5);
    assert_eq!(report.device_timeline_active_ns, 8);
    assert_eq!(report.gpu_idle_ns, 0);
    assert_eq!(report.host_wait_events, 1);
    assert_eq!(report.device_timeline_spans, 1);
    assert_eq!(report.runtime_timestamp_event_count, 1);
    assert_eq!(report.gpu_event_count, 1);
    assert_eq!(report.estimated_event_count, 1);
    assert!(report.host_wait_gpu_idle_sources_separate);
    assert!(!report.host_wait_equals_gpu_idle_value);
    assert!(report.proves_host_wait_not_gpu_idle());

    let json = report.to_json();
    assert!(json.contains("\"host_event_wait_ns\":5"));
    assert!(json.contains("\"gpu_idle_ns\":0"));
    assert!(json.contains("\"proves_host_wait_not_gpu_idle\":true"));
}

#[test]
fn critical_path_report_preserves_provenance_even_when_values_match() {
    let mut ledger = TokenLedger::new(1);
    ledger
        .record_device_span(DeviceTimelineSpan::new(
            DeviceOrdinal(0),
            0,
            5,
            MetricSource::GpuEvent,
            "kernel_a",
        ))
        .unwrap();
    ledger
        .record_device_span(DeviceTimelineSpan::new(
            DeviceOrdinal(0),
            8,
            13,
            MetricSource::GpuEvent,
            "kernel_b",
        ))
        .unwrap();
    ledger.record_sync(
        SyncClass::SoftVisibilitySync,
        None,
        Some(MemoryTier::Vram),
        Some(MemoryTier::PinnedDram),
        0,
        3,
        MetricSource::RuntimeTimestamp,
        "host_wait",
    );

    let report = TokenCriticalPathReport::from_ledger(&ledger, DeviceOrdinal(0)).unwrap();

    assert_eq!(report.host_event_wait_ns, 3);
    assert_eq!(report.gpu_idle_ns, 3);
    assert!(report.host_wait_gpu_idle_sources_separate);
    assert!(report.host_wait_equals_gpu_idle_value);
    assert!(!report.proves_host_wait_not_gpu_idle());
    assert!(!report.estimated_presented_as_measured);
}
