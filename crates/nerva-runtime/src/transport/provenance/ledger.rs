use nerva_core::types::memory::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::transport::provenance::entry::TransportMetricProvenanceEntry;

pub(crate) const MEASURED_LABEL: &str = "transport_measured_kernel_udp_p95";
pub(crate) const ESTIMATED_LABEL: &str = "transport_estimated_pinned_host_visible";

pub(crate) fn record_transport_provenance_events(
    ledger: &mut TokenLedger,
    entries: &[TransportMetricProvenanceEntry],
) {
    for entry in entries {
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Transport,
            sync_class: None,
            metric_source: entry.measured_source,
            block_id: None,
            from_tier: Some(MemoryTier::PinnedDram),
            to_tier: Some(MemoryTier::PinnedDram),
            bytes: entry.payload_bytes,
            latency_ns: entry.measured_p95_ns,
            label: MEASURED_LABEL,
        });
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Transport,
            sync_class: None,
            metric_source: entry.estimated_source,
            block_id: None,
            from_tier: Some(MemoryTier::Vram),
            to_tier: Some(MemoryTier::Vram),
            bytes: entry.payload_bytes,
            latency_ns: entry.estimated_visible_ns,
            label: ESTIMATED_LABEL,
        });
    }
}
