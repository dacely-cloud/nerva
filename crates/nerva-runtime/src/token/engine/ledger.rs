use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::token::TokenId;

use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::{DeviceTimelineSpan, LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

pub(crate) fn synthetic_step_ledger(
    device: DeviceOrdinal,
    token_index: u64,
    host_policy_barrier: bool,
) -> Result<TokenLedger> {
    let mut ledger = TokenLedger::new(token_index);
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::GraphReplay,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: None,
        to_tier: Some(MemoryTier::Vram),
        bytes: 0,
        latency_ns: 1,
        label: "synthetic_graph_replay",
    });
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::DeviceActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: None,
        to_tier: Some(MemoryTier::Vram),
        bytes: 0,
        latency_ns: 3,
        label: "synthetic_decode_kernel",
    });
    ledger.record_device_span(DeviceTimelineSpan::new(
        device,
        0,
        3,
        MetricSource::EstimatedModel,
        "synthetic_decode_device_active",
    ))?;
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Copy,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(MemoryTier::Vram),
        to_tier: Some(MemoryTier::PinnedDram),
        bytes: core::mem::size_of::<TokenId>(),
        latency_ns: 1,
        label: "async_host_token_observation",
    });
    ledger.record_sync(
        SyncClass::SoftVisibilitySync,
        None,
        Some(MemoryTier::Vram),
        Some(MemoryTier::PinnedDram),
        0,
        1,
        MetricSource::EstimatedModel,
        "soft_visibility_host_wait",
    );
    if host_policy_barrier {
        ledger.record_sync(
            SyncClass::PolicySync,
            None,
            Some(MemoryTier::PinnedDram),
            Some(MemoryTier::Vram),
            0,
            1,
            MetricSource::EstimatedModel,
            "host_policy_barrier",
        );
    }

    Ok(ledger)
}
