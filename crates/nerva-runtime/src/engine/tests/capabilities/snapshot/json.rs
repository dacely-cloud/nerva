use nerva_core::types::arch::HostArch;
use nerva_core::types::memory::fabric::MemoryFabricKind;

use crate::capabilities::snapshot::{CapabilitySnapshot, CapabilityState, TopologySnapshot};

#[test]
fn capability_snapshot_json_escapes_cuda_error() {
    let snapshot = CapabilitySnapshot {
        host_arch: HostArch::X86_64,
        target_os: "linux",
        target_arch: "x86_64",
        kernel_release: Some("kernel\" release".to_string()),
        fabric: MemoryFabricKind::DiscreteExplicit,
        cuda: CapabilityState::Unsupported,
        cuda_status: "failed",
        cuda_error: Some("quote\" slash\\ newline\n".to_string()),
        cuda_visible_devices: Some("0,1".to_string()),
        cuda_compute_capability: Some("8.9".to_string()),
        cuda_device_total_memory_bytes: Some(24 * 1024 * 1024 * 1024),
        cuda_pci_bus_id: Some("0000:65:00.0".to_string()),
        hip: CapabilityState::SupportedUnverified,
        hip_visible_devices: Some("2".to_string()),
        hip_runtime_present: true,
        hip_runtime_version: Some("rocm\"6\\test".to_string()),
        hip_amd_gpu_count: 1,
        hip_kfd_present: true,
        hip_amdgpu_loaded: true,
        nvidia_driver_version: Some("driver\\version".to_string()),
        rdma_core_loaded: true,
        mlx5_core_loaded: true,
        nvidia_peer_memory_module: Some("nvidia_peermem".to_string()),
        pinned_host_staging: CapabilityState::SupportedUnverified,
        gpu_direct_rdma: CapabilityState::SupportedUnverified,
        amd_peerdirect: CapabilityState::SupportedUnverified,
        dma_buf_export: CapabilityState::SupportedUnverified,
        dma_buf_kernel_present: true,
        dma_buf_nvidia_driver_present: true,
        dma_buf_nvidia_capability_entries: 4,
        dma_buf_cuda_vmm_export_symbols_present: true,
        cuda_posix_fd_handle_supported: Some(true),
        cuda_gpu_direct_rdma_supported: Some(true),
        cuda_gpu_direct_rdma_with_vmm_supported: Some(false),
        cxl: CapabilityState::Unsupported,
        topology: topology_fixture(),
    };

    let json = snapshot.to_json();
    assert!(json.contains("quote\\\" slash\\\\ newline\\n"));
    assert!(json.contains("kernel\\\" release"));
    assert!(json.contains("driver\\\\version"));
    assert!(json.contains("\"cuda_compute_capability\":\"8.9\""));
    assert!(json.contains("\"cuda_device_total_memory_bytes\":25769803776"));
    assert!(json.contains("\"hip_runtime_version\":\"rocm\\\"6\\\\test\""));
    assert!(json.contains("\"nvidia_peer_memory_module\":\"nvidia_peermem\""));
    assert!(json.contains("\"gpu_direct_rdma\":\"SUPPORTED_UNVERIFIED\""));
    assert!(json.contains("\"dma_buf_export\":\"SUPPORTED_UNVERIFIED\""));
    assert!(json.contains("\"dma_buf_kernel_present\":true"));
    assert!(json.contains("\"dma_buf_nvidia_capability_entries\":4"));
    assert!(json.contains("\"cuda_posix_fd_handle_supported\":true"));
    assert!(json.contains("\"cuda_gpu_direct_rdma_supported\":true"));
    assert!(json.contains("\"cuda_gpu_direct_rdma_with_vmm_supported\":false"));
    assert!(json.contains("\"rdma_netdev_links\":[\"mlx5_0:enp1s0f0\"]"));
    assert!(json.contains("\"iommu_kernel_args\":\"intel_iommu=on iommu=pt\""));
}

fn topology_fixture() -> TopologySnapshot {
    TopologySnapshot {
        cpu_online: Some("0-1".to_string()),
        cpu_count: 2,
        numa_node_count: 1,
        pci_device_count: 3,
        pci_root_complex_count: 1,
        pci_bus_count: 2,
        pci_gpu_count: 1,
        pci_network_count: 1,
        pci_nvme_count: 1,
        block_device_count: 2,
        nvme_block_device_count: 1,
        cxl_device_count: 2,
        cxl_memory_device_count: 1,
        cxl_region_count: 1,
        rdma_device_count: 1,
        rdma_device_names: vec!["mlx5_0".to_string()],
        rdma_netdev_links: vec!["mlx5_0:enp1s0f0".to_string()],
        iommu_group_count: 3,
        iommu_mode: "passthrough_groups_present".to_string(),
        iommu_kernel_args: Some("intel_iommu=on iommu=pt".to_string()),
    }
}
