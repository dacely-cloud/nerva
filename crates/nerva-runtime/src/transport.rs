use crate::capabilities::{CapabilitySnapshot, CapabilityState};
use nerva_core::types::{DeviceOrdinal, ExecutionOwner, MemoryTier, NervaError, Result};
use nerva_ledger::types::{
    FallbackClass, FallbackDecision, LedgerEvent, LedgerEventKind, MetricSource, SyncClass,
    TokenLedger,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransferMode {
    Decode,
    Prefill,
}

impl TransferMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Prefill => "prefill",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportPathKind {
    TrueGpuDirectRdma,
    OptimizedPinnedHostBounce,
    CpuProducedBoundary,
    MappedPinnedHostWrite,
}

impl TransportPathKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TrueGpuDirectRdma => "true_gpu_direct_rdma",
            Self::OptimizedPinnedHostBounce => "optimized_pinned_host_bounce",
            Self::CpuProducedBoundary => "cpu_produced_boundary",
            Self::MappedPinnedHostWrite => "mapped_pinned_host_write",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportPathClass {
    GpuDirect,
    HostStaged,
    CpuProduced,
    MappedPinned,
}

impl TransportPathClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GpuDirect => "GPU_DIRECT",
            Self::HostStaged => "HOST_STAGED",
            Self::CpuProduced => "CPU_PRODUCED",
            Self::MappedPinned => "MAPPED_PINNED",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportPathRequest {
    pub source_tier: MemoryTier,
    pub destination_tier: MemoryTier,
    pub bytes: usize,
    pub mode: TransferMode,
    pub producer: ExecutionOwner,
    pub gpu_direct_rdma: CapabilityState,
    pub mapped_pinned_output: CapabilityState,
    pub pinned_host_staging: CapabilityState,
}

impl TransportPathRequest {
    pub const fn new(
        source_tier: MemoryTier,
        destination_tier: MemoryTier,
        bytes: usize,
        mode: TransferMode,
        producer: ExecutionOwner,
        gpu_direct_rdma: CapabilityState,
        mapped_pinned_output: CapabilityState,
        pinned_host_staging: CapabilityState,
    ) -> Self {
        Self {
            source_tier,
            destination_tier,
            bytes,
            mode,
            producer,
            gpu_direct_rdma,
            mapped_pinned_output,
            pinned_host_staging,
        }
    }

    pub fn from_capabilities(
        source_tier: MemoryTier,
        destination_tier: MemoryTier,
        bytes: usize,
        mode: TransferMode,
        producer: ExecutionOwner,
        capabilities: &CapabilitySnapshot,
    ) -> Self {
        Self::new(
            source_tier,
            destination_tier,
            bytes,
            mode,
            producer,
            capabilities.gpu_direct_rdma,
            CapabilityState::Unsupported,
            capabilities.pinned_host_staging,
        )
    }
}

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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportCapabilityMatrixStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportMatrixRequestedPath {
    GpuDirectRdma,
    PinnedHostBounce,
    CpuProducedBoundary,
    MappedPinnedWrite,
}

impl TransportMatrixRequestedPath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GpuDirectRdma => "A_GPU_DIRECT_RDMA",
            Self::PinnedHostBounce => "B_PINNED_HOST_BOUNCE",
            Self::CpuProducedBoundary => "C_CPU_PRODUCED_BOUNDARY",
            Self::MappedPinnedWrite => "D_MAPPED_PINNED_WRITE",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportCapabilityMatrixEntry {
    pub requested_path: TransportMatrixRequestedPath,
    pub size_bytes: usize,
    pub mode: TransferMode,
    pub source_tier: MemoryTier,
    pub destination_tier: MemoryTier,
    pub selected_path: TransportPathKind,
    pub class: TransportPathClass,
    pub capability_result: CapabilityState,
    pub estimated_visible_ns: u64,
    pub effective_payload_bandwidth_bps: u64,
    pub estimated_cpu_core_ns: u64,
    pub dram_read_bytes: usize,
    pub dram_write_bytes: usize,
    pub pcie_tx_bytes: usize,
    pub pcie_rx_bytes: usize,
    pub explicit_copy_bytes: usize,
    pub nic_tx_bytes: usize,
    pub nic_rx_bytes: usize,
    pub pageable_copy: bool,
    pub per_token_registration: bool,
    pub registration_cache_hit: bool,
    pub queue_depth: u32,
    pub credit_stall_ns: u64,
}

impl TransportCapabilityMatrixEntry {
    pub fn to_json(self) -> String {
        format!(
            "{{\"requested_path\":\"{}\",\"size_bytes\":{},\"mode\":\"{}\",\"source_tier\":\"{}\",\"destination_tier\":\"{}\",\"selected_path\":\"{}\",\"class\":\"{}\",\"capability_result\":\"{}\",\"estimated_visible_ns\":{},\"metric_source\":\"estimated_model\",\"effective_payload_bandwidth_bps\":{},\"estimated_cpu_core_ns\":{},\"dram_read_bytes\":{},\"dram_write_bytes\":{},\"pcie_tx_bytes\":{},\"pcie_rx_bytes\":{},\"explicit_copy_bytes\":{},\"nic_tx_bytes\":{},\"nic_rx_bytes\":{},\"pageable_copy\":{},\"per_token_registration\":{},\"registration_cache_hit\":{},\"queue_depth\":{},\"credit_stall_ns\":{}}}",
            self.requested_path.as_str(),
            self.size_bytes,
            self.mode.as_str(),
            memory_tier_to_str(self.source_tier),
            memory_tier_to_str(self.destination_tier),
            self.selected_path.as_str(),
            self.class.as_str(),
            self.capability_result.as_str(),
            self.estimated_visible_ns,
            self.effective_payload_bandwidth_bps,
            self.estimated_cpu_core_ns,
            self.dram_read_bytes,
            self.dram_write_bytes,
            self.pcie_tx_bytes,
            self.pcie_rx_bytes,
            self.explicit_copy_bytes,
            self.nic_tx_bytes,
            self.nic_rx_bytes,
            self.pageable_copy,
            self.per_token_registration,
            self.registration_cache_hit,
            self.queue_depth,
            self.credit_stall_ns,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportCapabilityMatrixSummary {
    pub status: TransportCapabilityMatrixStatus,
    pub sizes: u64,
    pub entries: Vec<TransportCapabilityMatrixEntry>,
    pub decode_entries: u64,
    pub prefill_entries: u64,
    pub gpu_direct_entries: u64,
    pub host_staged_entries: u64,
    pub cpu_produced_entries: u64,
    pub mapped_pinned_entries: u64,
    pub supported_verified_entries: u64,
    pub supported_unverified_entries: u64,
    pub degraded_to_pinned_host_entries: u64,
    pub unsupported_entries: u64,
    pub total_estimated_visible_ns: u64,
    pub p50_estimated_visible_ns: u64,
    pub p95_estimated_visible_ns: u64,
    pub p99_estimated_visible_ns: u64,
    pub explicit_copy_bytes: usize,
    pub nic_tx_bytes: usize,
    pub nic_rx_bytes: usize,
    pub estimated_cpu_core_ns: u64,
    pub dram_read_bytes: usize,
    pub dram_write_bytes: usize,
    pub pcie_tx_bytes: usize,
    pub pcie_rx_bytes: usize,
    pub pageable_copies: u64,
    pub per_token_registrations: u64,
    pub registration_cache_hits: u64,
    pub credit_stall_ns: u64,
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

impl TransportCapabilityMatrixSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            TransportCapabilityMatrixStatus::Ok => "ok",
            TransportCapabilityMatrixStatus::Failed => "failed",
        };
        let mut entries = String::from("[");
        for (index, entry) in self.entries.iter().enumerate() {
            if index != 0 {
                entries.push(',');
            }
            entries.push_str(&entry.to_json());
        }
        entries.push(']');
        format!(
            "{{\"status\":\"{}\",\"sizes\":{},\"entries_count\":{},\"decode_entries\":{},\"prefill_entries\":{},\"gpu_direct_entries\":{},\"host_staged_entries\":{},\"cpu_produced_entries\":{},\"mapped_pinned_entries\":{},\"supported_verified_entries\":{},\"supported_unverified_entries\":{},\"degraded_to_pinned_host_entries\":{},\"unsupported_entries\":{},\"total_estimated_visible_ns\":{},\"p50_estimated_visible_ns\":{},\"p95_estimated_visible_ns\":{},\"p99_estimated_visible_ns\":{},\"explicit_copy_bytes\":{},\"nic_tx_bytes\":{},\"nic_rx_bytes\":{},\"estimated_cpu_core_ns\":{},\"dram_read_bytes\":{},\"dram_write_bytes\":{},\"pcie_tx_bytes\":{},\"pcie_rx_bytes\":{},\"pageable_copies\":{},\"per_token_registrations\":{},\"registration_cache_hits\":{},\"credit_stall_ns\":{},\"hot_path_allocations\":{},\"error\":{},\"entries\":{}}}",
            status,
            self.sizes,
            self.entries.len(),
            self.decode_entries,
            self.prefill_entries,
            self.gpu_direct_entries,
            self.host_staged_entries,
            self.cpu_produced_entries,
            self.mapped_pinned_entries,
            self.supported_verified_entries,
            self.supported_unverified_entries,
            self.degraded_to_pinned_host_entries,
            self.unsupported_entries,
            self.total_estimated_visible_ns,
            self.p50_estimated_visible_ns,
            self.p95_estimated_visible_ns,
            self.p99_estimated_visible_ns,
            self.explicit_copy_bytes,
            self.nic_tx_bytes,
            self.nic_rx_bytes,
            self.estimated_cpu_core_ns,
            self.dram_read_bytes,
            self.dram_write_bytes,
            self.pcie_tx_bytes,
            self.pcie_rx_bytes,
            self.pageable_copies,
            self.per_token_registrations,
            self.registration_cache_hits,
            self.credit_stall_ns,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
            entries,
        )
    }
}

pub fn plan_transport_path(request: TransportPathRequest) -> Result<TransportPathDecision> {
    if request.bytes == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "transport path request bytes must be non-zero".to_string(),
        });
    }

    if matches!(request.producer, ExecutionOwner::Cpu)
        && matches!(
            request.source_tier,
            MemoryTier::Dram | MemoryTier::PinnedDram
        )
    {
        return Ok(make_transport_decision(
            request,
            TransportPathKind::CpuProducedBoundary,
            TransportPathClass::CpuProduced,
            "CPU owns boundary result and can produce it into a registered send buffer",
            0,
        ));
    }

    if request.source_tier == MemoryTier::Vram
        && request.destination_tier == MemoryTier::Vram
        && request.gpu_direct_rdma == CapabilityState::SupportedAndVerified
    {
        return Ok(make_transport_decision(
            request,
            TransportPathKind::TrueGpuDirectRdma,
            TransportPathClass::GpuDirect,
            "verified GPU-direct RDMA path avoids host staging",
            0,
        ));
    }

    if request.source_tier == MemoryTier::Vram
        && request.mode == TransferMode::Decode
        && request.bytes <= 256 * 1024
        && request.mapped_pinned_output == CapabilityState::SupportedAndVerified
    {
        return Ok(make_transport_decision(
            request,
            TransportPathKind::MappedPinnedHostWrite,
            TransportPathClass::MappedPinned,
            "small decode payload can be written directly to mapped pinned output",
            0,
        ));
    }

    if matches!(
        request.pinned_host_staging,
        CapabilityState::SupportedAndVerified | CapabilityState::SupportedUnverified
    ) {
        let copy_bytes = if request.source_tier == MemoryTier::Vram
            && request.destination_tier == MemoryTier::Vram
        {
            request.bytes.saturating_mul(2)
        } else {
            request.bytes
        };
        return Ok(make_transport_decision(
            request,
            TransportPathKind::OptimizedPinnedHostBounce,
            TransportPathClass::HostStaged,
            "GPU-direct path is not verified; using preallocated pinned-host staging",
            copy_bytes,
        ));
    }

    Err(NervaError::BackendUnavailable {
        backend: "transport",
        reason: "no verified direct path and pinned-host staging is unavailable".to_string(),
    })
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

pub fn run_transport_capability_matrix_probe(
    device: DeviceOrdinal,
    capabilities: &CapabilitySnapshot,
) -> Result<TransportCapabilityMatrixSummary> {
    let sizes = [
        (32 * 1024, TransferMode::Decode),
        (256 * 1024, TransferMode::Decode),
        (1024 * 1024, TransferMode::Decode),
        (16 * 1024 * 1024, TransferMode::Prefill),
        (64 * 1024 * 1024, TransferMode::Prefill),
        (256 * 1024 * 1024, TransferMode::Prefill),
    ];
    let requested_paths = [
        TransportMatrixRequestedPath::GpuDirectRdma,
        TransportMatrixRequestedPath::PinnedHostBounce,
        TransportMatrixRequestedPath::CpuProducedBoundary,
        TransportMatrixRequestedPath::MappedPinnedWrite,
    ];
    let mut entries = Vec::with_capacity(sizes.len() * requested_paths.len());
    let ledger = TokenLedger::new(0);

    for (bytes, mode) in sizes {
        for requested_path in requested_paths {
            let request =
                transport_matrix_request(requested_path, bytes, mode, device, capabilities);
            let decision = plan_transport_path(request)?;
            let capability_result =
                transport_matrix_capability_result(requested_path, decision, capabilities);
            let resource = transport_resource_estimate(decision);
            entries.push(TransportCapabilityMatrixEntry {
                requested_path,
                size_bytes: bytes,
                mode,
                source_tier: decision.request.source_tier,
                destination_tier: decision.request.destination_tier,
                selected_path: decision.path,
                class: decision.class,
                capability_result,
                estimated_visible_ns: decision.estimated_visible_ns,
                effective_payload_bandwidth_bps: effective_payload_bandwidth_bps(
                    decision.request.bytes,
                    decision.estimated_visible_ns,
                ),
                estimated_cpu_core_ns: resource.estimated_cpu_core_ns,
                dram_read_bytes: resource.dram_read_bytes,
                dram_write_bytes: resource.dram_write_bytes,
                pcie_tx_bytes: resource.pcie_tx_bytes,
                pcie_rx_bytes: resource.pcie_rx_bytes,
                explicit_copy_bytes: decision.explicit_copy_bytes,
                nic_tx_bytes: decision.nic_tx_bytes,
                nic_rx_bytes: decision.nic_rx_bytes,
                pageable_copy: decision.pageable_copy,
                per_token_registration: decision.per_token_registration,
                registration_cache_hit: resource.registration_cache_hit,
                queue_depth: resource.queue_depth,
                credit_stall_ns: resource.credit_stall_ns,
            });
        }
    }

    ledger.require_zero_hot_path_allocations()?;
    Ok(transport_capability_matrix_summary(
        sizes.len() as u64,
        entries,
        ledger.hot_path_allocations,
    ))
}

fn transport_matrix_request(
    requested_path: TransportMatrixRequestedPath,
    bytes: usize,
    mode: TransferMode,
    device: DeviceOrdinal,
    capabilities: &CapabilitySnapshot,
) -> TransportPathRequest {
    match requested_path {
        TransportMatrixRequestedPath::GpuDirectRdma => TransportPathRequest::from_capabilities(
            MemoryTier::Vram,
            MemoryTier::Vram,
            bytes,
            mode,
            ExecutionOwner::Gpu(device),
            capabilities,
        ),
        TransportMatrixRequestedPath::PinnedHostBounce => TransportPathRequest::new(
            MemoryTier::Vram,
            MemoryTier::Vram,
            bytes,
            mode,
            ExecutionOwner::Gpu(device),
            CapabilityState::Unsupported,
            CapabilityState::Unsupported,
            capabilities.pinned_host_staging,
        ),
        TransportMatrixRequestedPath::CpuProducedBoundary => TransportPathRequest::new(
            MemoryTier::Dram,
            MemoryTier::PinnedDram,
            bytes,
            mode,
            ExecutionOwner::Cpu,
            CapabilityState::Unsupported,
            CapabilityState::Unsupported,
            capabilities.pinned_host_staging,
        ),
        TransportMatrixRequestedPath::MappedPinnedWrite => TransportPathRequest::new(
            MemoryTier::Vram,
            MemoryTier::PinnedDram,
            bytes,
            mode,
            ExecutionOwner::Gpu(device),
            CapabilityState::Unsupported,
            CapabilityState::Unsupported,
            capabilities.pinned_host_staging,
        ),
    }
}

fn transport_matrix_capability_result(
    requested_path: TransportMatrixRequestedPath,
    decision: TransportPathDecision,
    _capabilities: &CapabilitySnapshot,
) -> CapabilityState {
    match requested_path {
        TransportMatrixRequestedPath::GpuDirectRdma => {
            if decision.class == TransportPathClass::GpuDirect {
                CapabilityState::SupportedAndVerified
            } else if decision.class == TransportPathClass::HostStaged {
                CapabilityState::DegradedToPinnedHost
            } else {
                CapabilityState::Unsupported
            }
        }
        TransportMatrixRequestedPath::PinnedHostBounce => {
            if decision.class == TransportPathClass::HostStaged {
                decision.request.pinned_host_staging
            } else {
                CapabilityState::Unsupported
            }
        }
        TransportMatrixRequestedPath::CpuProducedBoundary => {
            if decision.class == TransportPathClass::CpuProduced {
                CapabilityState::SupportedAndVerified
            } else {
                CapabilityState::Unsupported
            }
        }
        TransportMatrixRequestedPath::MappedPinnedWrite => {
            if decision.class == TransportPathClass::MappedPinned {
                CapabilityState::SupportedAndVerified
            } else if decision.class == TransportPathClass::HostStaged {
                CapabilityState::DegradedToPinnedHost
            } else {
                CapabilityState::Unsupported
            }
        }
    }
}

fn transport_capability_matrix_summary(
    sizes: u64,
    entries: Vec<TransportCapabilityMatrixEntry>,
    hot_path_allocations: u64,
) -> TransportCapabilityMatrixSummary {
    let decode_entries = entries
        .iter()
        .filter(|entry| entry.mode == TransferMode::Decode)
        .count() as u64;
    let prefill_entries = entries
        .iter()
        .filter(|entry| entry.mode == TransferMode::Prefill)
        .count() as u64;
    let gpu_direct_entries = entries
        .iter()
        .filter(|entry| entry.class == TransportPathClass::GpuDirect)
        .count() as u64;
    let host_staged_entries = entries
        .iter()
        .filter(|entry| entry.class == TransportPathClass::HostStaged)
        .count() as u64;
    let cpu_produced_entries = entries
        .iter()
        .filter(|entry| entry.class == TransportPathClass::CpuProduced)
        .count() as u64;
    let mapped_pinned_entries = entries
        .iter()
        .filter(|entry| entry.class == TransportPathClass::MappedPinned)
        .count() as u64;
    let supported_verified_entries = entries
        .iter()
        .filter(|entry| entry.capability_result == CapabilityState::SupportedAndVerified)
        .count() as u64;
    let supported_unverified_entries = entries
        .iter()
        .filter(|entry| entry.capability_result == CapabilityState::SupportedUnverified)
        .count() as u64;
    let degraded_to_pinned_host_entries = entries
        .iter()
        .filter(|entry| entry.capability_result == CapabilityState::DegradedToPinnedHost)
        .count() as u64;
    let unsupported_entries = entries
        .iter()
        .filter(|entry| entry.capability_result == CapabilityState::Unsupported)
        .count() as u64;
    let total_estimated_visible_ns = entries
        .iter()
        .map(|entry| entry.estimated_visible_ns)
        .sum::<u64>();
    let explicit_copy_bytes = entries
        .iter()
        .map(|entry| entry.explicit_copy_bytes)
        .sum::<usize>();
    let nic_tx_bytes = entries
        .iter()
        .map(|entry| entry.nic_tx_bytes)
        .sum::<usize>();
    let nic_rx_bytes = entries
        .iter()
        .map(|entry| entry.nic_rx_bytes)
        .sum::<usize>();
    let estimated_cpu_core_ns = entries
        .iter()
        .map(|entry| entry.estimated_cpu_core_ns)
        .sum::<u64>();
    let dram_read_bytes = entries
        .iter()
        .map(|entry| entry.dram_read_bytes)
        .sum::<usize>();
    let dram_write_bytes = entries
        .iter()
        .map(|entry| entry.dram_write_bytes)
        .sum::<usize>();
    let pcie_tx_bytes = entries
        .iter()
        .map(|entry| entry.pcie_tx_bytes)
        .sum::<usize>();
    let pcie_rx_bytes = entries
        .iter()
        .map(|entry| entry.pcie_rx_bytes)
        .sum::<usize>();
    let pageable_copies = entries.iter().filter(|entry| entry.pageable_copy).count() as u64;
    let per_token_registrations = entries
        .iter()
        .filter(|entry| entry.per_token_registration)
        .count() as u64;
    let registration_cache_hits = entries
        .iter()
        .filter(|entry| entry.registration_cache_hit)
        .count() as u64;
    let credit_stall_ns = entries
        .iter()
        .map(|entry| entry.credit_stall_ns)
        .sum::<u64>();
    let p50_estimated_visible_ns = percentile_estimated_visible_ns(&entries, 50);
    let p95_estimated_visible_ns = percentile_estimated_visible_ns(&entries, 95);
    let p99_estimated_visible_ns = percentile_estimated_visible_ns(&entries, 99);

    TransportCapabilityMatrixSummary {
        status: TransportCapabilityMatrixStatus::Ok,
        sizes,
        entries,
        decode_entries,
        prefill_entries,
        gpu_direct_entries,
        host_staged_entries,
        cpu_produced_entries,
        mapped_pinned_entries,
        supported_verified_entries,
        supported_unverified_entries,
        degraded_to_pinned_host_entries,
        unsupported_entries,
        total_estimated_visible_ns,
        p50_estimated_visible_ns,
        p95_estimated_visible_ns,
        p99_estimated_visible_ns,
        explicit_copy_bytes,
        nic_tx_bytes,
        nic_rx_bytes,
        estimated_cpu_core_ns,
        dram_read_bytes,
        dram_write_bytes,
        pcie_tx_bytes,
        pcie_rx_bytes,
        pageable_copies,
        per_token_registrations,
        registration_cache_hits,
        credit_stall_ns,
        hot_path_allocations,
        error: None,
    }
}

fn percentile_estimated_visible_ns(
    entries: &[TransportCapabilityMatrixEntry],
    percentile: u64,
) -> u64 {
    if entries.is_empty() {
        return 0;
    }
    let mut values = entries
        .iter()
        .map(|entry| entry.estimated_visible_ns)
        .collect::<Vec<_>>();
    values.sort_unstable();
    let rank = div_ceil_u64(percentile.saturating_mul(values.len() as u64), 100).saturating_sub(1)
        as usize;
    values[rank.min(values.len() - 1)]
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct TransportResourceEstimate {
    estimated_cpu_core_ns: u64,
    dram_read_bytes: usize,
    dram_write_bytes: usize,
    pcie_tx_bytes: usize,
    pcie_rx_bytes: usize,
    registration_cache_hit: bool,
    queue_depth: u32,
    credit_stall_ns: u64,
}

fn transport_resource_estimate(decision: TransportPathDecision) -> TransportResourceEstimate {
    let bytes = decision.request.bytes;
    let explicit_half = decision.explicit_copy_bytes / 2;
    let queue_depth = match decision.request.mode {
        TransferMode::Decode => 1,
        TransferMode::Prefill => 4,
    };
    let estimated_cpu_core_ns = match (decision.path, decision.request.mode) {
        (TransportPathKind::TrueGpuDirectRdma, TransferMode::Decode) => 300,
        (TransportPathKind::TrueGpuDirectRdma, TransferMode::Prefill) => 800,
        (TransportPathKind::OptimizedPinnedHostBounce, TransferMode::Decode) => 1_000,
        (TransportPathKind::OptimizedPinnedHostBounce, TransferMode::Prefill) => 2_500,
        (TransportPathKind::CpuProducedBoundary, TransferMode::Decode) => {
            decision.estimated_visible_ns / 2
        }
        (TransportPathKind::CpuProducedBoundary, TransferMode::Prefill) => {
            decision.estimated_visible_ns / 3
        }
        (TransportPathKind::MappedPinnedHostWrite, TransferMode::Decode) => 800,
        (TransportPathKind::MappedPinnedHostWrite, TransferMode::Prefill) => 1_800,
    };

    match decision.path {
        TransportPathKind::TrueGpuDirectRdma => TransportResourceEstimate {
            estimated_cpu_core_ns,
            dram_read_bytes: 0,
            dram_write_bytes: 0,
            pcie_tx_bytes: decision.nic_tx_bytes,
            pcie_rx_bytes: decision.nic_rx_bytes,
            registration_cache_hit: !decision.per_token_registration,
            queue_depth,
            credit_stall_ns: 0,
        },
        TransportPathKind::OptimizedPinnedHostBounce => TransportResourceEstimate {
            estimated_cpu_core_ns,
            dram_read_bytes: decision.nic_tx_bytes,
            dram_write_bytes: decision.nic_rx_bytes.saturating_add(explicit_half),
            pcie_tx_bytes: decision.nic_tx_bytes.saturating_add(explicit_half),
            pcie_rx_bytes: decision.nic_rx_bytes.saturating_add(explicit_half),
            registration_cache_hit: !decision.per_token_registration,
            queue_depth,
            credit_stall_ns: 0,
        },
        TransportPathKind::CpuProducedBoundary => TransportResourceEstimate {
            estimated_cpu_core_ns,
            dram_read_bytes: bytes,
            dram_write_bytes: bytes,
            pcie_tx_bytes: decision.nic_tx_bytes,
            pcie_rx_bytes: decision.nic_rx_bytes,
            registration_cache_hit: !decision.per_token_registration,
            queue_depth,
            credit_stall_ns: 0,
        },
        TransportPathKind::MappedPinnedHostWrite => TransportResourceEstimate {
            estimated_cpu_core_ns,
            dram_read_bytes: decision.nic_tx_bytes,
            dram_write_bytes: bytes,
            pcie_tx_bytes: bytes.saturating_add(decision.nic_tx_bytes),
            pcie_rx_bytes: decision.nic_rx_bytes,
            registration_cache_hit: !decision.per_token_registration,
            queue_depth,
            credit_stall_ns: 0,
        },
    }
}

fn effective_payload_bandwidth_bps(bytes: usize, latency_ns: u64) -> u64 {
    if latency_ns == 0 {
        return 0;
    }
    let bps = (bytes as u128).saturating_mul(1_000_000_000) / latency_ns as u128;
    bps.min(u64::MAX as u128) as u64
}

fn make_transport_decision(
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

fn estimate_transport_visible_ns(path: TransportPathKind, bytes: usize, mode: TransferMode) -> u64 {
    let setup_ns = match (path, mode) {
        (TransportPathKind::TrueGpuDirectRdma, TransferMode::Decode) => 900,
        (TransportPathKind::TrueGpuDirectRdma, TransferMode::Prefill) => 1_500,
        (TransportPathKind::OptimizedPinnedHostBounce, TransferMode::Decode) => 2_500,
        (TransportPathKind::OptimizedPinnedHostBounce, TransferMode::Prefill) => 4_000,
        (TransportPathKind::CpuProducedBoundary, TransferMode::Decode) => 1_200,
        (TransportPathKind::CpuProducedBoundary, TransferMode::Prefill) => 2_400,
        (TransportPathKind::MappedPinnedHostWrite, TransferMode::Decode) => 1_700,
        (TransportPathKind::MappedPinnedHostWrite, TransferMode::Prefill) => 4_500,
    };
    let bytes_per_ns = match path {
        TransportPathKind::TrueGpuDirectRdma => 64,
        TransportPathKind::OptimizedPinnedHostBounce => 24,
        TransportPathKind::CpuProducedBoundary => 48,
        TransportPathKind::MappedPinnedHostWrite => 32,
    };
    setup_ns + div_ceil_u64(bytes as u64, bytes_per_ns)
}

fn div_ceil_u64(value: u64, divisor: u64) -> u64 {
    value / divisor + u64::from(value % divisor != 0)
}

fn memory_tier_to_str(value: MemoryTier) -> &'static str {
    match value {
        MemoryTier::Vram => "VRAM",
        MemoryTier::SharedHbmOrLpddr => "SHARED_HBM_OR_LPDDR",
        MemoryTier::PinnedDram => "PINNED_DRAM",
        MemoryTier::Dram => "DRAM",
        MemoryTier::Cxl => "CXL",
        MemoryTier::Disk => "DISK",
    }
}

fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{value}\""))
}
