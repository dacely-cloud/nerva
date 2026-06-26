use crate::capabilities::snapshot::{CapabilitySnapshot, CapabilityState};
use crate::transport::estimate::{effective_payload_bandwidth_bps, transport_resource_estimate};
use crate::transport::matrix::export::transport_matrix_export_evidence;
use crate::transport::matrix::types::{
    TransportCapabilityMatrixEntry, TransportMatrixRequestedPath,
};
use crate::transport::path::decision::TransportPathDecision;
use crate::transport::path::planner::plan_transport_path;
use crate::transport::path::request::TransportPathRequest;
use crate::transport::path::types::{TransferMode, TransportPathClass};
use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;

const MATRIX_SIZES: [(usize, TransferMode); 6] = [
    (32 * 1024, TransferMode::Decode),
    (256 * 1024, TransferMode::Decode),
    (1024 * 1024, TransferMode::Decode),
    (16 * 1024 * 1024, TransferMode::Prefill),
    (64 * 1024 * 1024, TransferMode::Prefill),
    (256 * 1024 * 1024, TransferMode::Prefill),
];

const REQUESTED_PATHS: [TransportMatrixRequestedPath; 4] = [
    TransportMatrixRequestedPath::GpuDirectRdma,
    TransportMatrixRequestedPath::PinnedHostBounce,
    TransportMatrixRequestedPath::CpuProducedBoundary,
    TransportMatrixRequestedPath::MappedPinnedWrite,
];

pub(crate) fn build_entries(
    device: DeviceOrdinal,
    capabilities: &CapabilitySnapshot,
) -> Result<(u64, Vec<TransportCapabilityMatrixEntry>)> {
    let mut entries = Vec::with_capacity(MATRIX_SIZES.len() * REQUESTED_PATHS.len());

    for (bytes, mode) in MATRIX_SIZES {
        for requested_path in REQUESTED_PATHS {
            let request =
                transport_matrix_request(requested_path, bytes, mode, device, capabilities);
            let decision = plan_transport_path(request)?;
            let capability_result =
                transport_matrix_capability_result(requested_path, decision, capabilities);
            let export_evidence = transport_matrix_export_evidence(requested_path, capabilities);
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
                gpu_memory_export_verified: export_evidence.gpu_memory_export_verified,
                cuda_vmm_posix_fd_export_verified: export_evidence
                    .cuda_vmm_posix_fd_export_verified,
                gpu_direct_rdma_verified: export_evidence.gpu_direct_rdma_verified,
                gpu_export_without_nic_direct: export_evidence.gpu_export_without_nic_direct,
                estimated_visible_ns: decision.estimated_visible_ns,
                visible_non_overlapped_ns: decision.estimated_visible_ns,
                effective_payload_bandwidth_bps: effective_payload_bandwidth_bps(
                    decision.request.bytes,
                    decision.estimated_visible_ns,
                ),
                host_event_wait_ns: decision.estimated_visible_ns,
                gpu_idle_ns: 0,
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

    Ok((MATRIX_SIZES.len() as u64, entries))
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
