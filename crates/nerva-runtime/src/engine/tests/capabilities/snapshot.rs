use std::env;

use nerva_core::types::arch::{HostArch, host_arch};
use nerva_core::types::memory::fabric::MemoryFabricKind;

use crate::capabilities::snapshot::{CapabilitySnapshot, CapabilityState, TopologySnapshot};
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
    assert_eq!(snapshot.hip, CapabilityState::Unsupported);
    assert_eq!(
        snapshot.pinned_host_staging,
        CapabilityState::SupportedUnverified
    );
    assert!(matches!(
        snapshot.gpu_direct_rdma,
        CapabilityState::DegradedToPinnedHost | CapabilityState::SupportedUnverified
    ));
    assert_eq!(snapshot.amd_peerdirect, CapabilityState::Unsupported);
    assert_eq!(snapshot.dma_buf_export, CapabilityState::Unsupported);
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
    assert!(json.contains("\"rdma_core_loaded\""));
    assert!(json.contains("\"mlx5_core_loaded\""));
    assert!(json.contains("\"nvidia_peer_memory_module\""));
    assert!(json.contains("\"gpu_direct_rdma\""));
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
}

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
        hip: CapabilityState::Unsupported,
        hip_visible_devices: Some("2".to_string()),
        nvidia_driver_version: Some("driver\\version".to_string()),
        rdma_core_loaded: true,
        mlx5_core_loaded: true,
        nvidia_peer_memory_module: Some("nvidia_peermem".to_string()),
        pinned_host_staging: CapabilityState::SupportedUnverified,
        gpu_direct_rdma: CapabilityState::SupportedUnverified,
        amd_peerdirect: CapabilityState::Unsupported,
        dma_buf_export: CapabilityState::Unsupported,
        cxl: CapabilityState::Unsupported,
        topology: TopologySnapshot {
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
        },
    };

    let json = snapshot.to_json();
    assert!(json.contains("quote\\\" slash\\\\ newline\\n"));
    assert!(json.contains("kernel\\\" release"));
    assert!(json.contains("driver\\\\version"));
    assert!(json.contains("\"cuda_compute_capability\":\"8.9\""));
    assert!(json.contains("\"cuda_device_total_memory_bytes\":25769803776"));
    assert!(json.contains("\"cuda_pci_bus_id\":\"0000:65:00.0\""));
    assert!(json.contains("\"rdma_core_loaded\":true"));
    assert!(json.contains("\"mlx5_core_loaded\":true"));
    assert!(json.contains("\"nvidia_peer_memory_module\":\"nvidia_peermem\""));
    assert!(json.contains("\"gpu_direct_rdma\":\"SUPPORTED_UNVERIFIED\""));
    assert!(json.contains("\"cpu_online\":\"0-1\""));
    assert!(json.contains("\"pci_root_complex_count\":1"));
    assert!(json.contains("\"pci_bus_count\":2"));
    assert!(json.contains("\"cxl_device_count\":2"));
    assert!(json.contains("\"cxl_memory_device_count\":1"));
    assert!(json.contains("\"cxl_region_count\":1"));
    assert!(json.contains("\"rdma_device_names\":[\"mlx5_0\"]"));
    assert!(json.contains("\"rdma_netdev_links\":[\"mlx5_0:enp1s0f0\"]"));
    assert!(json.contains("\"iommu_mode\":\"passthrough_groups_present\""));
    assert!(json.contains("\"iommu_kernel_args\":\"intel_iommu=on iommu=pt\""));
}
