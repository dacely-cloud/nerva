use crate::transport::path::decision::TransportPathDecision;
use crate::transport::path::types::{TransferMode, TransportPathKind};
use crate::transport::probe::status::TransportPathProbeStatus;
use crate::transport::probe::summary::TransportPathProbeSummary;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TransportProbeAccumulator {
    pub(crate) ledger: TokenLedger,
    requests: u64,
    decode_requests: u64,
    prefill_requests: u64,
    gpu_direct_paths: u64,
    pinned_host_paths: u64,
    cpu_produced_paths: u64,
    mapped_pinned_paths: u64,
    nic_tx_bytes: usize,
    nic_rx_bytes: usize,
    explicit_copy_bytes: usize,
    pageable_copies: u64,
    per_token_registrations: u64,
}

impl TransportProbeAccumulator {
    pub(crate) fn new() -> Self {
        Self {
            ledger: TokenLedger::new(0),
            requests: 0,
            decode_requests: 0,
            prefill_requests: 0,
            gpu_direct_paths: 0,
            pinned_host_paths: 0,
            cpu_produced_paths: 0,
            mapped_pinned_paths: 0,
            nic_tx_bytes: 0,
            nic_rx_bytes: 0,
            explicit_copy_bytes: 0,
            pageable_copies: 0,
            per_token_registrations: 0,
        }
    }

    pub(crate) fn record(&mut self, decision: TransportPathDecision) {
        self.requests = self.requests.saturating_add(1);
        match decision.request.mode {
            TransferMode::Decode => self.decode_requests = self.decode_requests.saturating_add(1),
            TransferMode::Prefill => {
                self.prefill_requests = self.prefill_requests.saturating_add(1)
            }
        }
        match decision.path {
            TransportPathKind::TrueGpuDirectRdma => {
                self.gpu_direct_paths = self.gpu_direct_paths.saturating_add(1)
            }
            TransportPathKind::OptimizedPinnedHostBounce => {
                self.pinned_host_paths = self.pinned_host_paths.saturating_add(1)
            }
            TransportPathKind::CpuProducedBoundary => {
                self.cpu_produced_paths = self.cpu_produced_paths.saturating_add(1)
            }
            TransportPathKind::MappedPinnedHostWrite => {
                self.mapped_pinned_paths = self.mapped_pinned_paths.saturating_add(1)
            }
        }
        self.nic_tx_bytes = self.nic_tx_bytes.saturating_add(decision.nic_tx_bytes);
        self.nic_rx_bytes = self.nic_rx_bytes.saturating_add(decision.nic_rx_bytes);
        self.explicit_copy_bytes = self
            .explicit_copy_bytes
            .saturating_add(decision.explicit_copy_bytes);
        if decision.pageable_copy {
            self.pageable_copies = self.pageable_copies.saturating_add(1);
        }
        if decision.per_token_registration {
            self.per_token_registrations = self.per_token_registrations.saturating_add(1);
        }
        decision.record_to_ledger(&mut self.ledger);
    }

    pub(crate) fn finish(self) -> TransportPathProbeSummary {
        TransportPathProbeSummary {
            status: TransportPathProbeStatus::Ok,
            requests: self.requests,
            decode_requests: self.decode_requests,
            prefill_requests: self.prefill_requests,
            gpu_direct_paths: self.gpu_direct_paths,
            pinned_host_paths: self.pinned_host_paths,
            cpu_produced_paths: self.cpu_produced_paths,
            mapped_pinned_paths: self.mapped_pinned_paths,
            transport_events: self.ledger.event_count(LedgerEventKind::Transport),
            copy_events: self.ledger.event_count(LedgerEventKind::Copy),
            sync_events: self.ledger.event_count(LedgerEventKind::Sync),
            phase_handoff_syncs: self.ledger.sync_count_for(SyncClass::PhaseHandoff),
            fallback_decisions: self.ledger.fallback_count(),
            nic_tx_bytes: self.nic_tx_bytes,
            nic_rx_bytes: self.nic_rx_bytes,
            explicit_copy_bytes: self.explicit_copy_bytes,
            pageable_copies: self.pageable_copies,
            per_token_registrations: self.per_token_registrations,
            estimated_events: self
                .ledger
                .event_count_for_source(MetricSource::EstimatedModel),
            estimated_latency_ns: self
                .ledger
                .latency_ns_for_source(MetricSource::EstimatedModel),
            total_latency_ns: self.ledger.total_latency_ns(),
            hot_path_allocations: self.ledger.hot_path_allocations,
            error: None,
        }
    }
}
