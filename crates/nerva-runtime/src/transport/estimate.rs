use crate::transport::path::decision::TransportPathDecision;
use crate::transport::path::types::{TransferMode, TransportPathKind};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct TransportResourceEstimate {
    pub estimated_cpu_core_ns: u64,
    pub dram_read_bytes: usize,
    pub dram_write_bytes: usize,
    pub pcie_tx_bytes: usize,
    pub pcie_rx_bytes: usize,
    pub registration_cache_hit: bool,
    pub queue_depth: u32,
    pub credit_stall_ns: u64,
}

pub(crate) fn transport_resource_estimate(
    decision: TransportPathDecision,
) -> TransportResourceEstimate {
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

pub(crate) fn effective_payload_bandwidth_bps(bytes: usize, latency_ns: u64) -> u64 {
    if latency_ns == 0 {
        return 0;
    }
    let bps = (bytes as u128).saturating_mul(1_000_000_000) / latency_ns as u128;
    bps.min(u64::MAX as u128) as u64
}

pub(crate) fn estimate_transport_visible_ns(
    path: TransportPathKind,
    bytes: usize,
    mode: TransferMode,
) -> u64 {
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

pub(crate) fn div_ceil_u64(value: u64, divisor: u64) -> u64 {
    value / divisor + u64::from(value % divisor != 0)
}
