use std::env;

use nerva_core::types::arch::{HostArch, host_arch};
use nerva_core::types::memory::MemoryFabricKind;

use crate::capabilities::discovery::gpu_direct_rdma_capability;
use crate::capabilities::json::json_string_array;
use crate::capabilities::linux::{
    count_linux_id_list, discover_iommu_mode, extract_iommu_kernel_args, parse_pci_class,
};
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
    assert_eq!(snapshot.fabric, MemoryFabricKind::DiscreteExplicit);
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
    assert_eq!(snapshot.cxl, CapabilityState::Unsupported);
    assert!(snapshot.topology.cpu_count > 0);

    let json = snapshot.to_json();
    assert!(json.contains("\"target_os\":\"linux\""));
    assert!(json.contains("\"kernel_release\""));
    assert!(json.contains("\"fabric\":\"DiscreteExplicit\""));
    assert!(json.contains("\"cuda_compute_capability\""));
    assert!(json.contains("\"cuda_device_total_memory_bytes\""));
    assert!(json.contains("\"cuda_pci_bus_id\""));
    assert!(json.contains("\"rdma_core_loaded\""));
    assert!(json.contains("\"mlx5_core_loaded\""));
    assert!(json.contains("\"nvidia_peer_memory_module\""));
    assert!(json.contains("\"gpu_direct_rdma\""));
    assert!(json.contains("\"topology\""));
    assert!(json.contains("\"cpu_count\""));
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
    assert!(json.contains("\"rdma_device_names\":[\"mlx5_0\"]"));
    assert!(json.contains("\"rdma_netdev_links\":[\"mlx5_0:enp1s0f0\"]"));
    assert!(json.contains("\"iommu_mode\":\"passthrough_groups_present\""));
    assert!(json.contains("\"iommu_kernel_args\":\"intel_iommu=on iommu=pt\""));
}

#[test]
fn topology_snapshot_reports_basic_sysfs_counts() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let snapshot = runtime.discover_topology();

    assert!(snapshot.cpu_count > 0);
    assert!(snapshot.numa_node_count > 0);
    assert!(snapshot.pci_device_count >= snapshot.pci_gpu_count);
    assert!(snapshot.pci_device_count >= snapshot.pci_network_count);
    assert!(snapshot.pci_device_count >= snapshot.pci_nvme_count);
    if snapshot.pci_root_complex_count > 0 {
        assert!(snapshot.pci_bus_count >= snapshot.pci_root_complex_count);
    }
    assert!(snapshot.block_device_count >= snapshot.nvme_block_device_count);
    assert_eq!(snapshot.rdma_device_count, snapshot.rdma_device_names.len());
    assert!(snapshot.rdma_netdev_links.len() >= snapshot.rdma_device_names.len());
    assert!(!snapshot.iommu_mode.is_empty());
    let json = snapshot.to_json();
    assert!(json.contains("\"cpu_count\""));
    assert!(json.contains("\"pci_device_count\""));
    assert!(json.contains("\"pci_root_complex_count\""));
    assert!(json.contains("\"pci_bus_count\""));
    assert!(json.contains("\"rdma_device_names\""));
    assert!(json.contains("\"rdma_netdev_links\""));
    assert!(json.contains("\"iommu_mode\""));
}

#[test]
fn topology_helpers_parse_linux_id_and_pci_class_values() {
    assert_eq!(count_linux_id_list("0-3"), Some(4));
    assert_eq!(count_linux_id_list("0-1,4,8-9"), Some(5));
    assert_eq!(count_linux_id_list("2-1"), None);
    assert_eq!(parse_pci_class("0x030000"), Some(0x030000));
    assert_eq!(parse_pci_class("010802"), Some(0x010802));
    assert_eq!(parse_pci_class("not-hex"), None);
    assert_eq!(
        json_string_array(&["a\"b".to_string(), "c\\d".to_string()]),
        "[\"a\\\"b\",\"c\\\\d\"]"
    );
    assert_eq!(
        extract_iommu_kernel_args("root=/dev/sda intel_iommu=on quiet iommu=pt"),
        Some("intel_iommu=on iommu=pt".to_string())
    );
    assert_eq!(extract_iommu_kernel_args("root=/dev/sda quiet"), None);
    assert_eq!(
        discover_iommu_mode(9, Some("intel_iommu=on iommu=pt")),
        "passthrough_groups_present"
    );
    assert_eq!(
        discover_iommu_mode(9, Some("intel_iommu=off")),
        "disabled_by_kernel_arg"
    );
    assert_eq!(discover_iommu_mode(9, None), "enabled_groups_present");
    assert_eq!(
        discover_iommu_mode(0, Some("amd_iommu=on")),
        "enabled_requested"
    );
    assert_eq!(discover_iommu_mode(0, None), "not_detected");

    assert_eq!(
        gpu_direct_rdma_capability(
            CapabilityState::SupportedAndVerified,
            2,
            Some("nvidia_peermem")
        ),
        CapabilityState::SupportedUnverified
    );
    assert_eq!(
        gpu_direct_rdma_capability(CapabilityState::Unsupported, 2, Some("nvidia_peermem")),
        CapabilityState::DegradedToPinnedHost
    );
    assert_eq!(
        gpu_direct_rdma_capability(
            CapabilityState::SupportedAndVerified,
            0,
            Some("nv_peer_mem")
        ),
        CapabilityState::DegradedToPinnedHost
    );
    assert_eq!(
        gpu_direct_rdma_capability(CapabilityState::SupportedAndVerified, 2, None),
        CapabilityState::DegradedToPinnedHost
    );
}
