use crate::capabilities::snapshot::CapabilitySnapshot;
use crate::transport::json::json_opt_static_str;
use crate::transport::path::{
    TransferMode, TransportPathDecision, TransportPathKind, TransportPathRequest,
    plan_transport_path,
};
use nerva_core::types::error::Result;
use nerva_core::types::id::DeviceOrdinal;
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::TokenLedger;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportPathProbeStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportPathProbeSummary {
    pub status: TransportPathProbeStatus,
    pub requests: u64,
    pub decode_requests: u64,
    pub prefill_requests: u64,
    pub gpu_direct_paths: u64,
    pub pinned_host_paths: u64,
    pub cpu_produced_paths: u64,
    pub mapped_pinned_paths: u64,
    pub transport_events: u64,
    pub copy_events: u64,
    pub sync_events: u64,
    pub phase_handoff_syncs: u64,
    pub fallback_decisions: u64,
    pub nic_tx_bytes: usize,
    pub nic_rx_bytes: usize,
    pub explicit_copy_bytes: usize,
    pub pageable_copies: u64,
    pub per_token_registrations: u64,
    pub estimated_events: u64,
    pub estimated_latency_ns: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TransportProbeAccumulator {
    ledger: TokenLedger,
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
    fn new() -> Self {
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

    fn record(&mut self, decision: TransportPathDecision) {
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

    fn finish(self) -> TransportPathProbeSummary {
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

impl TransportPathProbeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            TransportPathProbeStatus::Ok => "ok",
            TransportPathProbeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"requests\":{},\"decode_requests\":{},\"prefill_requests\":{},\"gpu_direct_paths\":{},\"pinned_host_paths\":{},\"cpu_produced_paths\":{},\"mapped_pinned_paths\":{},\"transport_events\":{},\"copy_events\":{},\"sync_events\":{},\"phase_handoff_syncs\":{},\"fallback_decisions\":{},\"nic_tx_bytes\":{},\"nic_rx_bytes\":{},\"explicit_copy_bytes\":{},\"pageable_copies\":{},\"per_token_registrations\":{},\"estimated_events\":{},\"estimated_latency_ns\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.requests,
            self.decode_requests,
            self.prefill_requests,
            self.gpu_direct_paths,
            self.pinned_host_paths,
            self.cpu_produced_paths,
            self.mapped_pinned_paths,
            self.transport_events,
            self.copy_events,
            self.sync_events,
            self.phase_handoff_syncs,
            self.fallback_decisions,
            self.nic_tx_bytes,
            self.nic_rx_bytes,
            self.explicit_copy_bytes,
            self.pageable_copies,
            self.per_token_registrations,
            self.estimated_events,
            self.estimated_latency_ns,
            self.total_latency_ns,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}

pub fn run_transport_path_probe(
    device: DeviceOrdinal,
    capabilities: &CapabilitySnapshot,
) -> Result<TransportPathProbeSummary> {
    let sizes = [
        (32 * 1024, TransferMode::Decode),
        (256 * 1024, TransferMode::Decode),
        (1024 * 1024, TransferMode::Decode),
        (16 * 1024 * 1024, TransferMode::Prefill),
        (64 * 1024 * 1024, TransferMode::Prefill),
        (256 * 1024 * 1024, TransferMode::Prefill),
    ];
    let mut probe = TransportProbeAccumulator::new();

    for (bytes, mode) in sizes {
        let request = TransportPathRequest::from_capabilities(
            MemoryTier::Vram,
            MemoryTier::Vram,
            bytes,
            mode,
            ExecutionOwner::Gpu(device),
            capabilities,
        );
        let decision = plan_transport_path(request)?;
        probe.record(decision);
    }

    let cpu_request = TransportPathRequest::from_capabilities(
        MemoryTier::Dram,
        MemoryTier::PinnedDram,
        32 * 1024,
        TransferMode::Decode,
        ExecutionOwner::Cpu,
        capabilities,
    );
    let cpu_decision = plan_transport_path(cpu_request)?;
    probe.record(cpu_decision);
    probe.ledger.require_zero_hot_path_allocations()?;
    probe.ledger.require_classified_syncs()?;

    Ok(probe.finish())
}
