use crate::capabilities::snapshot::{CapabilitySnapshot, CapabilityState};
use crate::transport::estimate::estimate_transport_visible_ns;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::fallback::{FallbackClass, FallbackDecision};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::TokenLedger;

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
    #[allow(clippy::too_many_arguments)]
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
