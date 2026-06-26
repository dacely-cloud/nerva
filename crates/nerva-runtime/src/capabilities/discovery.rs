use std::{env, fs};

use crate::capabilities::linux::read_trimmed_first_line;
use crate::capabilities::snapshot::{CapabilitySnapshot, CapabilityState};
use crate::capabilities::topology::discover_topology_snapshot;
use nerva_core::types::arch::host_arch;
use nerva_core::types::memory::fabric::MemoryFabricKind;
use nerva_cuda::smoke::probe::smoke;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_cuda::smoke::summary::CudaSmokeSummary;

pub fn cuda_smoke() -> CudaSmokeSummary {
    smoke()
}

pub fn discover_capabilities() -> CapabilitySnapshot {
    let cuda_smoke = cuda_smoke();
    let cuda = match cuda_smoke.status {
        SmokeStatus::Ok => CapabilityState::SupportedAndVerified,
        SmokeStatus::Unavailable | SmokeStatus::Failed => CapabilityState::Unsupported,
    };
    let cuda_status = match cuda_smoke.status {
        SmokeStatus::Ok => "ok",
        SmokeStatus::Unavailable => "unavailable",
        SmokeStatus::Failed => "failed",
    };
    let cuda_compute_capability = cuda_compute_capability(&cuda_smoke);
    let cuda_device_total_memory_bytes = cuda_smoke.device_total_memory_bytes;
    let cuda_pci_bus_id = cuda_smoke.pci_bus_id.clone();
    let topology = discover_topology_snapshot();
    let rdma_core_loaded = module_loaded("ib_core");
    let mlx5_core_loaded = module_loaded("mlx5_core");
    let nvidia_peer_memory_module = detect_nvidia_peer_memory_module();
    let gpu_direct_rdma = gpu_direct_rdma_capability(
        cuda,
        topology.rdma_device_count,
        nvidia_peer_memory_module.as_deref(),
    );

    CapabilitySnapshot {
        host_arch: host_arch(),
        target_os: env::consts::OS,
        target_arch: env::consts::ARCH,
        kernel_release: read_trimmed_first_line("/proc/sys/kernel/osrelease"),
        fabric: MemoryFabricKind::DiscreteExplicit,
        cuda,
        cuda_status,
        cuda_error: cuda_smoke.error,
        cuda_visible_devices: env::var("CUDA_VISIBLE_DEVICES").ok(),
        cuda_compute_capability,
        cuda_device_total_memory_bytes,
        cuda_pci_bus_id,
        hip: CapabilityState::Unsupported,
        hip_visible_devices: env::var("HIP_VISIBLE_DEVICES").ok(),
        nvidia_driver_version: read_trimmed_first_line("/proc/driver/nvidia/version"),
        rdma_core_loaded,
        mlx5_core_loaded,
        nvidia_peer_memory_module,
        pinned_host_staging: CapabilityState::SupportedUnverified,
        gpu_direct_rdma,
        amd_peerdirect: CapabilityState::Unsupported,
        dma_buf_export: CapabilityState::Unsupported,
        cxl: CapabilityState::Unsupported,
        topology,
    }
}

fn cuda_compute_capability(summary: &CudaSmokeSummary) -> Option<String> {
    match (
        summary.compute_capability_major,
        summary.compute_capability_minor,
    ) {
        (Some(major), Some(minor)) => Some(format!("{major}.{minor}")),
        _ => None,
    }
}

fn module_loaded(name: &str) -> bool {
    fs::metadata(format!("/sys/module/{name}"))
        .map(|metadata| metadata.is_dir())
        .unwrap_or(false)
}

fn detect_nvidia_peer_memory_module() -> Option<String> {
    ["nvidia_peermem", "nv_peer_mem"]
        .into_iter()
        .find(|name| module_loaded(name))
        .map(ToOwned::to_owned)
}

pub(crate) fn gpu_direct_rdma_capability(
    cuda: CapabilityState,
    rdma_device_count: usize,
    nvidia_peer_memory_module: Option<&str>,
) -> CapabilityState {
    if cuda == CapabilityState::SupportedAndVerified
        && rdma_device_count > 0
        && nvidia_peer_memory_module.is_some()
    {
        CapabilityState::SupportedUnverified
    } else {
        CapabilityState::DegradedToPinnedHost
    }
}
