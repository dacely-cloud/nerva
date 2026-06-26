use crate::capabilities::snapshot::CapabilityState;
use crate::transport::dpdk_udp::protocol::DpdkUdpMemoryPath;
use nerva_core::types::error::{NervaError, Result};

pub(super) struct DpdkUdpPathSelection {
    pub selected_path: DpdkUdpMemoryPath,
    pub capability_result: CapabilityState,
    pub pinned_host_required: bool,
    pub direct_gpu_memory_claimed: bool,
}

pub(super) fn select_memory_path(
    dpdk_udp_gpu: CapabilityState,
    dpdk_udp_pinned_host: CapabilityState,
) -> Result<DpdkUdpPathSelection> {
    if dpdk_udp_gpu == CapabilityState::SupportedAndVerified {
        return Ok(DpdkUdpPathSelection {
            selected_path: DpdkUdpMemoryPath::GpuBuffer,
            capability_result: CapabilityState::SupportedAndVerified,
            pinned_host_required: false,
            direct_gpu_memory_claimed: true,
        });
    }

    if matches!(
        dpdk_udp_pinned_host,
        CapabilityState::SupportedAndVerified | CapabilityState::SupportedUnverified
    ) {
        return Ok(DpdkUdpPathSelection {
            selected_path: DpdkUdpMemoryPath::PinnedHostBuffer,
            capability_result: CapabilityState::DegradedToPinnedHost,
            pinned_host_required: true,
            direct_gpu_memory_claimed: false,
        });
    }

    Err(NervaError::BackendUnavailable {
        backend: "dpdk_udp",
        reason: "no verified GPU-buffer path and pinned-host DPDK UDP is unavailable".to_string(),
    })
}
