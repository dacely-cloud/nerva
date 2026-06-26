use crate::capabilities::snapshot::CapabilityState;
use crate::transport::estimate::div_ceil_u64;
use crate::transport::matrix::types::TransportCapabilityMatrixEntry;
use crate::transport::path::types::{TransferMode, TransportPathClass};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct MatrixCounters {
    pub decode_entries: u64,
    pub prefill_entries: u64,
    pub gpu_direct_entries: u64,
    pub host_staged_entries: u64,
    pub cpu_produced_entries: u64,
    pub mapped_pinned_entries: u64,
    pub total_payload_bytes: usize,
    pub supported_verified_entries: u64,
    pub supported_unverified_entries: u64,
    pub degraded_to_pinned_host_entries: u64,
    pub unsupported_entries: u64,
    pub gpu_memory_export_verified_entries: u64,
    pub cuda_vmm_posix_fd_export_verified_entries: u64,
    pub gpu_direct_rdma_verified_entries: u64,
    pub gpu_export_without_nic_direct_entries: u64,
    pub false_gpu_direct_claims: u64,
    pub total_estimated_visible_ns: u64,
    pub visible_non_overlapped_ns: u64,
    pub host_event_wait_ns: u64,
    pub gpu_idle_ns: u64,
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
    pub registration_cache_hit_rate_per_mille: u64,
    pub max_queue_depth: u32,
    pub estimated_nic_utilization_per_mille: u64,
    pub credit_stall_ns: u64,
    pub p50_estimated_visible_ns: u64,
    pub p95_estimated_visible_ns: u64,
    pub p99_estimated_visible_ns: u64,
}

pub(crate) fn aggregate_entries(entries: &[TransportCapabilityMatrixEntry]) -> MatrixCounters {
    let registration_cache_hits = count_by(entries, |entry| entry.registration_cache_hit);
    let total_payload_bytes = entries.iter().map(|entry| entry.size_bytes).sum::<usize>();
    let nic_tx_bytes = sum_usize(entries, |entry| entry.nic_tx_bytes);

    MatrixCounters {
        decode_entries: count_by(entries, |entry| entry.mode == TransferMode::Decode),
        prefill_entries: count_by(entries, |entry| entry.mode == TransferMode::Prefill),
        gpu_direct_entries: count_by(entries, |entry| {
            entry.class == TransportPathClass::GpuDirect
        }),
        host_staged_entries: count_by(entries, |entry| {
            entry.class == TransportPathClass::HostStaged
        }),
        cpu_produced_entries: count_by(entries, |entry| {
            entry.class == TransportPathClass::CpuProduced
        }),
        mapped_pinned_entries: count_by(entries, |entry| {
            entry.class == TransportPathClass::MappedPinned
        }),
        total_payload_bytes,
        supported_verified_entries: count_capability(
            entries,
            CapabilityState::SupportedAndVerified,
        ),
        supported_unverified_entries: count_capability(
            entries,
            CapabilityState::SupportedUnverified,
        ),
        degraded_to_pinned_host_entries: count_capability(
            entries,
            CapabilityState::DegradedToPinnedHost,
        ),
        unsupported_entries: count_capability(entries, CapabilityState::Unsupported),
        gpu_memory_export_verified_entries: count_by(entries, |entry| {
            entry.gpu_memory_export_verified
        }),
        cuda_vmm_posix_fd_export_verified_entries: count_by(entries, |entry| {
            entry.cuda_vmm_posix_fd_export_verified
        }),
        gpu_direct_rdma_verified_entries: count_by(entries, |entry| entry.gpu_direct_rdma_verified),
        gpu_export_without_nic_direct_entries: count_by(entries, |entry| {
            entry.gpu_export_without_nic_direct
        }),
        false_gpu_direct_claims: count_by(entries, |entry| {
            entry.class == TransportPathClass::GpuDirect && !entry.gpu_direct_rdma_verified
        }),
        total_estimated_visible_ns: sum_u64(entries, |entry| entry.estimated_visible_ns),
        visible_non_overlapped_ns: sum_u64(entries, |entry| entry.visible_non_overlapped_ns),
        host_event_wait_ns: sum_u64(entries, |entry| entry.host_event_wait_ns),
        gpu_idle_ns: sum_u64(entries, |entry| entry.gpu_idle_ns),
        explicit_copy_bytes: sum_usize(entries, |entry| entry.explicit_copy_bytes),
        nic_tx_bytes,
        nic_rx_bytes: sum_usize(entries, |entry| entry.nic_rx_bytes),
        estimated_cpu_core_ns: sum_u64(entries, |entry| entry.estimated_cpu_core_ns),
        dram_read_bytes: sum_usize(entries, |entry| entry.dram_read_bytes),
        dram_write_bytes: sum_usize(entries, |entry| entry.dram_write_bytes),
        pcie_tx_bytes: sum_usize(entries, |entry| entry.pcie_tx_bytes),
        pcie_rx_bytes: sum_usize(entries, |entry| entry.pcie_rx_bytes),
        pageable_copies: count_by(entries, |entry| entry.pageable_copy),
        per_token_registrations: count_by(entries, |entry| entry.per_token_registration),
        registration_cache_hits,
        registration_cache_hit_rate_per_mille: ratio_per_mille(
            registration_cache_hits,
            entries.len() as u64,
        ),
        max_queue_depth: entries
            .iter()
            .map(|entry| entry.queue_depth)
            .max()
            .unwrap_or(0),
        estimated_nic_utilization_per_mille: ratio_per_mille(
            nic_tx_bytes as u64,
            total_payload_bytes as u64,
        ),
        credit_stall_ns: sum_u64(entries, |entry| entry.credit_stall_ns),
        p50_estimated_visible_ns: percentile_estimated_visible_ns(entries, 50),
        p95_estimated_visible_ns: percentile_estimated_visible_ns(entries, 95),
        p99_estimated_visible_ns: percentile_estimated_visible_ns(entries, 99),
    }
}

fn count_capability(
    entries: &[TransportCapabilityMatrixEntry],
    capability: CapabilityState,
) -> u64 {
    count_by(entries, |entry| entry.capability_result == capability)
}

fn count_by(
    entries: &[TransportCapabilityMatrixEntry],
    predicate: impl Fn(&TransportCapabilityMatrixEntry) -> bool,
) -> u64 {
    entries.iter().filter(|entry| predicate(entry)).count() as u64
}

fn sum_u64(
    entries: &[TransportCapabilityMatrixEntry],
    value: impl Fn(&TransportCapabilityMatrixEntry) -> u64,
) -> u64 {
    entries.iter().map(value).sum::<u64>()
}

fn sum_usize(
    entries: &[TransportCapabilityMatrixEntry],
    value: impl Fn(&TransportCapabilityMatrixEntry) -> usize,
) -> usize {
    entries.iter().map(value).sum::<usize>()
}

fn ratio_per_mille(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        0
    } else {
        numerator.saturating_mul(1_000) / denominator
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
