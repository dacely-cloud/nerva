use crate::capabilities::snapshot::CapabilityState;
use crate::transport::path::decision::{TransportPathDecision, make_transport_decision};
use crate::transport::path::request::TransportPathRequest;
use crate::transport::path::types::{TransferMode, TransportPathClass, TransportPathKind};
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;

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
