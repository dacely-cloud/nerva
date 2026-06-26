use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::device::DeviceOrdinal;

use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;

use crate::types::decision::{
    BlockVersionDependency, CandidateCost, ExecutionDecision, ResidencyDecision,
};
use crate::types::event::{DeviceTimelineSpan, LedgerEvent, LedgerEventKind};
use crate::types::fallback::{FallbackClass, FallbackDecision};
use crate::types::metric::MetricSource;
use crate::types::sync::SyncClass;
use crate::types::token::ledger::TokenLedger;

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
