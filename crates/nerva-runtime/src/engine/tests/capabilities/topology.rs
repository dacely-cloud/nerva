use crate::capabilities::discovery::{
    cxl_capability, detected_memory_fabric, gpu_direct_rdma_capability,
};
use crate::capabilities::json::json_string_array;
use crate::capabilities::linux::{
    count_linux_id_list, discover_iommu_mode, extract_iommu_kernel_args, parse_pci_class,
};
use crate::capabilities::snapshot::CapabilityState;
use crate::engine::runtime::{Runtime, RuntimeConfig};
use nerva_core::types::memory::fabric::MemoryFabricKind;

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
    assert!(snapshot.cxl_device_count >= snapshot.cxl_memory_device_count);
    assert!(snapshot.cxl_device_count >= snapshot.cxl_region_count);
    assert_eq!(snapshot.rdma_device_count, snapshot.rdma_device_names.len());
    assert!(snapshot.rdma_netdev_links.len() >= snapshot.rdma_device_names.len());
    assert!(!snapshot.iommu_mode.is_empty());
    let json = snapshot.to_json();
    assert!(json.contains("\"cpu_count\""));
    assert!(json.contains("\"pci_device_count\""));
    assert!(json.contains("\"pci_root_complex_count\""));
    assert!(json.contains("\"pci_bus_count\""));
    assert!(json.contains("\"rdma_device_names\""));
    assert!(json.contains("\"cxl_device_count\""));
    assert!(json.contains("\"cxl_memory_device_count\""));
    assert!(json.contains("\"cxl_region_count\""));
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

    assert_eq!(cxl_capability(0, 0), CapabilityState::Unsupported);
    assert_eq!(cxl_capability(1, 0), CapabilityState::SupportedUnverified);
    assert_eq!(cxl_capability(1, 1), CapabilityState::SupportedUnverified);
    assert_eq!(
        detected_memory_fabric(CapabilityState::Unsupported),
        MemoryFabricKind::DiscreteExplicit
    );
    assert_eq!(
        detected_memory_fabric(CapabilityState::SupportedUnverified),
        MemoryFabricKind::CxlCoherentFabric
    );
}
