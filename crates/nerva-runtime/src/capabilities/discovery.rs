use std::{env, fs};

use crate::capabilities::hip::{discover_hip_evidence, hip_runtime_usable};
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
    let cxl = cxl_capability(topology.cxl_device_count, topology.cxl_memory_device_count);
    let fabric = detected_memory_fabric(cxl);
    let hip_evidence = discover_hip_evidence();
    let hip = hip_capability(&hip_evidence);
    let amd_peerdirect = amd_peerdirect_capability(hip, topology.rdma_device_count);

    CapabilitySnapshot {
        host_arch: host_arch(),
        target_os: env::consts::OS,
        target_arch: env::consts::ARCH,
        kernel_release: read_trimmed_first_line("/proc/sys/kernel/osrelease"),
        fabric,
        cuda,
        cuda_status,
        cuda_error: cuda_smoke.error,
        cuda_visible_devices: env::var("CUDA_VISIBLE_DEVICES").ok(),
        cuda_compute_capability,
        cuda_device_total_memory_bytes,
        cuda_pci_bus_id,
        hip,
        hip_visible_devices: env::var("HIP_VISIBLE_DEVICES").ok(),
        hip_runtime_present: hip_evidence.runtime_present,
        hip_runtime_version: hip_evidence.runtime_version,
        hip_amd_gpu_count: hip_evidence.amd_gpu_count,
        hip_kfd_present: hip_evidence.kfd_present,
        hip_amdgpu_loaded: hip_evidence.amdgpu_loaded,
        nvidia_driver_version: read_trimmed_first_line("/proc/driver/nvidia/version"),
        rdma_core_loaded,
        mlx5_core_loaded,
        nvidia_peer_memory_module,
        pinned_host_staging: CapabilityState::SupportedUnverified,
        gpu_direct_rdma,
        amd_peerdirect,
        dma_buf_export: CapabilityState::Unsupported,
        cxl,
        topology,
    }
}

pub(crate) fn hip_capability(
    evidence: &crate::capabilities::hip::HipCapabilityEvidence,
) -> CapabilityState {
    if hip_runtime_usable(evidence) {
        CapabilityState::SupportedUnverified
    } else {
        CapabilityState::Unsupported
    }
}

pub(crate) fn amd_peerdirect_capability(
    hip: CapabilityState,
    rdma_device_count: usize,
) -> CapabilityState {
    if hip != CapabilityState::Unsupported && rdma_device_count > 0 {
        CapabilityState::SupportedUnverified
    } else {
        CapabilityState::Unsupported
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

pub(crate) fn cxl_capability(
    cxl_device_count: usize,
    cxl_memory_device_count: usize,
) -> CapabilityState {
    if cxl_memory_device_count > 0 || cxl_device_count > 0 {
        CapabilityState::SupportedUnverified
    } else {
        CapabilityState::Unsupported
    }
}

pub(crate) fn detected_memory_fabric(cxl: CapabilityState) -> MemoryFabricKind {
    if cxl != CapabilityState::Unsupported {
        MemoryFabricKind::CxlCoherentFabric
    } else {
        MemoryFabricKind::DiscreteExplicit
    }
}
