use crate::transport::estimate::estimate_transport_visible_ns;
use crate::transport::path::request::TransportPathRequest;
use crate::transport::path::types::{TransportPathClass, TransportPathKind};
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::fallback::{FallbackClass, FallbackDecision};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportPathDecision {
    pub path: TransportPathKind,
    pub class: TransportPathClass,
    pub request: TransportPathRequest,
    pub reason: &'static str,
    pub estimated_visible_ns: u64,
    pub explicit_copy_bytes: usize,
    pub nic_tx_bytes: usize,
    pub nic_rx_bytes: usize,
    pub pageable_copy: bool,
    pub per_token_registration: bool,
}

impl TransportPathDecision {
    pub fn record_to_ledger(self, ledger: &mut TokenLedger) {
        if self.class == TransportPathClass::HostStaged {
            ledger.record_fallback_decision(FallbackDecision {
                label: "transport_host_staged_fallback",
                class: FallbackClass::CapabilityDegraded,
                requested: "gpu_direct_rdma",
                selected: self.path.as_str(),
                reason: self.reason,
                visible_ns: Some(self.estimated_visible_ns),
                metric_source: MetricSource::EstimatedModel,
            });
        }
        if self.explicit_copy_bytes > 0 {
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::Copy,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: None,
                from_tier: Some(self.request.source_tier),
                to_tier: Some(MemoryTier::PinnedDram),
                bytes: self.explicit_copy_bytes,
                latency_ns: self.estimated_visible_ns / 2,
                label: "transport_explicit_copy",
            });
        }
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Transport,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: None,
            from_tier: Some(self.request.source_tier),
            to_tier: Some(self.request.destination_tier),
            bytes: self.request.bytes,
            latency_ns: self.estimated_visible_ns,
            label: self.path.as_str(),
        });
        ledger.record_sync(
            SyncClass::PhaseHandoff,
            None,
            Some(self.request.source_tier),
            Some(self.request.destination_tier),
            0,
            1,
            MetricSource::EstimatedModel,
            "transport_ordering_barrier",
        );
    }
}

pub(crate) fn make_transport_decision(
    request: TransportPathRequest,
    path: TransportPathKind,
    class: TransportPathClass,
    reason: &'static str,
    explicit_copy_bytes: usize,
) -> TransportPathDecision {
    TransportPathDecision {
        path,
        class,
        request,
        reason,
        estimated_visible_ns: estimate_transport_visible_ns(path, request.bytes, request.mode),
        explicit_copy_bytes,
        nic_tx_bytes: request.bytes,
        nic_rx_bytes: request.bytes,
        pageable_copy: false,
        per_token_registration: false,
    }
}
