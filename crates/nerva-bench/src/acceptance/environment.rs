use nerva_core::types::memory::fabric::MemoryFabricKind;
use nerva_runtime::capabilities::snapshot::CapabilityState;
use nerva_runtime::engine::runtime::Runtime;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_capability_provenance(report: &mut AcceptanceReport, runtime: &Runtime) {
    let capabilities = runtime.discover_capabilities();
    let capability_passed = capabilities.target_os == "linux"
        && !capabilities.target_arch.is_empty()
        && capabilities.kernel_release.is_some()
        && matches!(
            capabilities.fabric,
            MemoryFabricKind::DiscreteExplicit | MemoryFabricKind::CxlCoherentFabric
        )
        && (capabilities.cxl != CapabilityState::Unsupported
            || matches!(capabilities.fabric, MemoryFabricKind::DiscreteExplicit))
        && (capabilities.hip == CapabilityState::Unsupported
            || (capabilities.hip_runtime_present
                && capabilities.hip_amd_gpu_count > 0
                && capabilities.hip_kfd_present
                && capabilities.hip_amdgpu_loaded))
        && !matches!(
            capabilities.pinned_host_staging,
            CapabilityState::Unsupported
        );
    report.push(
        "capability_provenance",
        capability_passed,
        format!(
            "target={}-{} kernel_present={} fabric={:?} cxl={:?} cxl_devices={} cxl_memory_devices={} cxl_regions={} hip={:?} hip_runtime_present={} hip_runtime_version={} hip_amd_gpu_count={} hip_kfd_present={} hip_amdgpu_loaded={} amd_peerdirect={:?} pinned_host_staging={:?} gpu_direct_rdma={:?} rdma_core_loaded={} mlx5_core_loaded={} peer_memory_module={} topology_cpu_count={}",
            capabilities.target_os,
            capabilities.target_arch,
            capabilities.kernel_release.is_some(),
            capabilities.fabric,
            capabilities.cxl,
            capabilities.topology.cxl_device_count,
            capabilities.topology.cxl_memory_device_count,
            capabilities.topology.cxl_region_count,
            capabilities.hip,
            capabilities.hip_runtime_present,
            capabilities
                .hip_runtime_version
                .as_deref()
                .unwrap_or("none"),
            capabilities.hip_amd_gpu_count,
            capabilities.hip_kfd_present,
            capabilities.hip_amdgpu_loaded,
            capabilities.amd_peerdirect,
            capabilities.pinned_host_staging,
            capabilities.gpu_direct_rdma,
            capabilities.rdma_core_loaded,
            capabilities.mlx5_core_loaded,
            capabilities
                .nvidia_peer_memory_module
                .as_deref()
                .unwrap_or("none"),
            capabilities.topology.cpu_count,
        ),
    );
}

pub(crate) fn push_topology_snapshot(report: &mut AcceptanceReport, runtime: &Runtime) {
    let topology = runtime.discover_topology();
    report.push(
        "topology_snapshot",
        topology.cpu_count > 0
            && topology.numa_node_count > 0
            && topology.pci_device_count >= topology.pci_gpu_count
            && topology.pci_device_count >= topology.pci_network_count
            && topology.pci_device_count >= topology.pci_nvme_count
            && (topology.pci_root_complex_count == 0
                || topology.pci_bus_count >= topology.pci_root_complex_count)
            && topology.block_device_count >= topology.nvme_block_device_count
            && topology.rdma_device_count == topology.rdma_device_names.len(),
        format!(
            "cpu_count={} numa_nodes={} pci_devices={} pci_roots={} pci_buses={} pci_gpu={} pci_network={} pci_nvme={} block_devices={} nvme_block_devices={} cxl_devices={} cxl_memory_devices={} cxl_regions={} rdma_devices={} rdma_links={} iommu_groups={} iommu_mode={}",
            topology.cpu_count,
            topology.numa_node_count,
            topology.pci_device_count,
            topology.pci_root_complex_count,
            topology.pci_bus_count,
            topology.pci_gpu_count,
            topology.pci_network_count,
            topology.pci_nvme_count,
            topology.block_device_count,
            topology.nvme_block_device_count,
            topology.cxl_device_count,
            topology.cxl_memory_device_count,
            topology.cxl_region_count,
            topology.rdma_device_count,
            topology.rdma_netdev_links.join("|"),
            topology.iommu_group_count,
            topology.iommu_mode,
        ),
    );
}
