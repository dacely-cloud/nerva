use std::env;

use nerva_core::types::arch::host_arch;
use nerva_core::types::memory::fabric::MemoryFabricKind;

use crate::capabilities::dma_buf::DmaBufExportEvidence;
use crate::capabilities::hip::HipCapabilityEvidence;
use crate::capabilities::snapshot::CapabilityState;
use crate::engine::runtime::{Runtime, RuntimeConfig};

#[test]
fn capability_snapshot_reports_conservative_discrete_profile() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let snapshot = runtime.discover_capabilities();

    assert_eq!(snapshot.host_arch, host_arch());
    assert_eq!(snapshot.target_os, env::consts::OS);
    assert_eq!(snapshot.target_arch, env::consts::ARCH);
    assert!(
        snapshot
            .kernel_release
            .as_deref()
            .is_none_or(|value| !value.is_empty())
    );
    assert_eq!(
        snapshot.fabric,
        if snapshot.cxl == CapabilityState::Unsupported {
            MemoryFabricKind::DiscreteExplicit
        } else {
            MemoryFabricKind::CxlCoherentFabric
        }
    );
    assert!(matches!(
        snapshot.cuda,
        CapabilityState::SupportedAndVerified | CapabilityState::Unsupported
    ));
    assert_eq!(
        snapshot.hip,
        crate::capabilities::discovery::hip_capability(&HipCapabilityEvidence {
            runtime_present: snapshot.hip_runtime_present,
            runtime_version: snapshot.hip_runtime_version.clone(),
            amd_gpu_count: snapshot.hip_amd_gpu_count,
            kfd_present: snapshot.hip_kfd_present,
            amdgpu_loaded: snapshot.hip_amdgpu_loaded,
        })
    );
    assert_eq!(
        snapshot.pinned_host_staging,
        CapabilityState::SupportedUnverified
    );
    assert!(matches!(
        snapshot.gpu_direct_rdma,
        CapabilityState::DegradedToPinnedHost | CapabilityState::SupportedUnverified
    ));
    assert_eq!(
        snapshot.amd_peerdirect,
        crate::capabilities::discovery::amd_peerdirect_capability(
            snapshot.hip,
            snapshot.topology.rdma_device_count,
        )
    );
    assert_eq!(
        snapshot.dma_buf_export,
        crate::capabilities::dma_buf::dma_buf_export_capability(&DmaBufExportEvidence {
            kernel_dma_buf_present: snapshot.dma_buf_kernel_present,
            nvidia_driver_present: snapshot.dma_buf_nvidia_driver_present,
            nvidia_capability_entries: snapshot.dma_buf_nvidia_capability_entries,
            cuda_vmm_export_symbols_present: snapshot.dma_buf_cuda_vmm_export_symbols_present,
            cuda_posix_fd_handle_supported: snapshot.cuda_posix_fd_handle_supported,
            cuda_gpu_direct_rdma_supported: snapshot.cuda_gpu_direct_rdma_supported,
            cuda_gpu_direct_rdma_with_vmm_supported: snapshot
                .cuda_gpu_direct_rdma_with_vmm_supported,
        })
    );
    assert_eq!(
        snapshot.cxl,
        crate::capabilities::discovery::cxl_capability(
            snapshot.topology.cxl_device_count,
            snapshot.topology.cxl_memory_device_count,
        )
    );
    assert!(snapshot.topology.cpu_count > 0);

    let json = snapshot.to_json();
    assert!(json.contains("\"target_os\":\"linux\""));
    assert!(json.contains("\"kernel_release\""));
    assert!(json.contains("\"fabric\""));
    assert!(json.contains("\"cuda_compute_capability\""));
    assert!(json.contains("\"cuda_device_total_memory_bytes\""));
    assert!(json.contains("\"cuda_pci_bus_id\""));
    assert!(json.contains("\"hip_runtime_present\""));
    assert!(json.contains("\"hip_amd_gpu_count\""));
    assert!(json.contains("\"hip_kfd_present\""));
    assert!(json.contains("\"hip_amdgpu_loaded\""));
    assert!(json.contains("\"rdma_core_loaded\""));
    assert!(json.contains("\"mlx5_core_loaded\""));
    assert!(json.contains("\"nvidia_peer_memory_module\""));
    assert!(json.contains("\"gpu_direct_rdma\""));
    assert!(json.contains("\"dma_buf_export\""));
    assert!(json.contains("\"dma_buf_kernel_present\""));
    assert!(json.contains("\"dma_buf_nvidia_driver_present\""));
    assert!(json.contains("\"dma_buf_cuda_vmm_export_symbols_present\""));
    assert!(json.contains("\"cuda_posix_fd_handle_supported\""));
    assert!(json.contains("\"cuda_gpu_direct_rdma_supported\""));
    assert!(json.contains("\"cuda_gpu_direct_rdma_with_vmm_supported\""));
    assert!(json.contains("\"topology\""));
    assert!(json.contains("\"cpu_count\""));
    assert!(json.contains("\"cxl_device_count\""));
}

#[test]
fn cuda_probe_survives_capability_discovery_when_device_is_available() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let snapshot = runtime.discover_capabilities();
    if snapshot.cuda != CapabilityState::SupportedAndVerified {
        return;
    }

    let smoke = crate::capabilities::discovery::cuda_smoke();
    assert_eq!(
        smoke.status,
        nerva_cuda::smoke::status::SmokeStatus::Ok,
        "smoke after discovery: {smoke:?}"
    );
    assert_eq!(smoke.kernel_value, Some(0x4e45_5256));
    assert_eq!(smoke.hot_path_allocations, 0);
    assert!(smoke.posix_fd_handle_supported.is_some());
    assert!(smoke.gpu_direct_rdma_supported.is_some());
    assert!(smoke.gpu_direct_rdma_with_cuda_vmm_supported.is_some());
}
