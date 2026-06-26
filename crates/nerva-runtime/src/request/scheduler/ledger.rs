use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::token::TokenId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::{DeviceTimelineSpan, LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::critical::TokenCriticalPathReport;
use nerva_ledger::types::token::ledger::TokenLedger;

pub(crate) fn scheduler_token_ledger(
    device: DeviceOrdinal,
    request_id: RequestId,
    token_index: u64,
    token: TokenId,
) -> Result<(TokenLedger, TokenCriticalPathReport)> {
    let mut ledger = TokenLedger::new(token_index);
    record_scheduler_select(&mut ledger, request_id);
    record_device_step(&mut ledger, device, token_index, token)?;
    record_host_observation(&mut ledger);
    ledger.require_classified_syncs()?;
    ledger.require_zero_hot_path_allocations()?;
    let report = TokenCriticalPathReport::from_ledger(&ledger, device)?;
    Ok((ledger, report))
}

fn record_scheduler_select(ledger: &mut TokenLedger, request_id: RequestId) {
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::CpuActivity,
        sync_class: None,
        metric_source: MetricSource::RuntimeTimestamp,
        block_id: None,
        from_tier: None,
        to_tier: None,
        bytes: core::mem::size_of_val(&request_id),
        latency_ns: 1,
        label: "request_scheduler_select",
    });
}

fn record_device_step(
    ledger: &mut TokenLedger,
    device: DeviceOrdinal,
    token_index: u64,
    token: TokenId,
) -> Result<()> {
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::GraphReplay,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: None,
        to_tier: Some(MemoryTier::Vram),
        bytes: core::mem::size_of_val(&token),
        latency_ns: 1,
        label: "request_scheduler_decode_replay",
    });
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::DeviceActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: None,
        to_tier: Some(MemoryTier::Vram),
        bytes: core::mem::size_of::<TokenId>(),
        latency_ns: 3,
        label: "request_scheduler_device_token",
    });
    ledger.record_device_span(DeviceTimelineSpan::new(
        device,
        token_index.saturating_mul(4),
        token_index.saturating_mul(4).saturating_add(3),
        MetricSource::EstimatedModel,
        "request_scheduler_device_token",
    ))
}

fn record_host_observation(ledger: &mut TokenLedger) {
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Copy,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(MemoryTier::Vram),
        to_tier: Some(MemoryTier::PinnedDram),
        bytes: core::mem::size_of::<TokenId>(),
        latency_ns: 1,
        label: "request_scheduler_host_token_copy",
    });
    ledger.record_sync(
        SyncClass::SoftVisibilitySync,
        None,
        Some(MemoryTier::Vram),
        Some(MemoryTier::PinnedDram),
        0,
        1,
        MetricSource::RuntimeTimestamp,
        "request_scheduler_host_visibility",
    );
}
