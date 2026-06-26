use crate::capabilities::snapshot::CapabilityState;
use crate::transport::estimate::div_ceil_u64;
use crate::transport::matrix::types::{
    TransportCapabilityMatrixEntry, TransportCapabilityMatrixStatus,
    TransportCapabilityMatrixSummary,
};
use crate::transport::path::{TransferMode, TransportPathClass};

pub(crate) fn transport_capability_matrix_summary(
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
