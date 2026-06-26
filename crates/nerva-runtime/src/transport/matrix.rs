use crate::capabilities::snapshot::{CapabilitySnapshot, CapabilityState};
use crate::transport::estimate::{
    div_ceil_u64, effective_payload_bandwidth_bps, transport_resource_estimate,
};
use crate::transport::json::{json_opt_static_str, memory_tier_to_str};
use crate::transport::path::{
    TransferMode, TransportPathClass, TransportPathDecision, TransportPathKind,
    TransportPathRequest, plan_transport_path,
};
use nerva_core::types::error::Result;
use nerva_core::types::id::DeviceOrdinal;
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::token::TokenLedger;

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
