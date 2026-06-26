use crate::capabilities::snapshot::CapabilityState;
use crate::transport::path::{TransferMode, TransportPathClass, TransportPathKind};
use nerva_core::types::memory::MemoryTier;

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
