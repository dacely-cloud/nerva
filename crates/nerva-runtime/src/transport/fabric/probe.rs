use crate::capabilities::snapshot::{CapabilitySnapshot, CapabilityState};
use crate::transport::fabric::summary::{
    FabricRdmaAffinity, FabricTopologyStatus, FabricTopologySummary,
};
use crate::transport::fabric::sysfs::{
    pci_device_location, rdma_device_pci_location, rdma_netdevs,
};

pub fn run_fabric_topology_probe(capabilities: &CapabilitySnapshot) -> FabricTopologySummary {
    let gpu = capabilities
        .cuda_pci_bus_id
        .as_deref()
        .map(pci_device_location);
    let gpu_pci_bus_id = gpu
        .as_ref()
        .and_then(|location| location.pci_bus_id.clone());
    let gpu_root_complex = gpu
        .as_ref()
        .and_then(|location| location.root_complex.clone());
    let gpu_numa_node = gpu.as_ref().and_then(|location| location.numa_node);

    let mut rdma_affinity = Vec::with_capacity(capabilities.topology.rdma_device_names.len());
    for rdma_device in &capabilities.topology.rdma_device_names {
        let location = rdma_device_pci_location(rdma_device);
        let same_root_as_gpu = gpu_root_complex.is_some()
            && location.root_complex.is_some()
            && location.root_complex == gpu_root_complex;
        let same_numa_as_gpu = gpu_numa_node.is_some()
            && location.numa_node.is_some()
            && location.numa_node == gpu_numa_node;
        rdma_affinity.push(FabricRdmaAffinity {
            rdma_device: rdma_device.clone(),
            pci_bus_id: location.pci_bus_id,
            root_complex: location.root_complex,
            numa_node: location.numa_node,
            netdevs: rdma_netdevs(rdma_device),
            same_root_as_gpu,
            same_numa_as_gpu,
        });
    }

    let rdma_devices = rdma_affinity.len() as u64;
    let rdma_with_pci_path = rdma_affinity
        .iter()
        .filter(|entry| entry.pci_bus_id.is_some() && entry.root_complex.is_some())
        .count() as u64;
    let rdma_same_root_as_gpu = rdma_affinity
        .iter()
        .filter(|entry| entry.same_root_as_gpu)
        .count() as u64;
    let rdma_same_numa_as_gpu = rdma_affinity
        .iter()
        .filter(|entry| entry.same_numa_as_gpu)
        .count() as u64;
    let gpu_direct_verified = capabilities.gpu_direct_rdma == CapabilityState::SupportedAndVerified;
    let degraded_to_pinned_host = capabilities.gpu_direct_rdma
        == CapabilityState::DegradedToPinnedHost
        || !gpu_direct_verified;
    let false_direct_claims = u64::from(gpu_direct_verified && rdma_same_root_as_gpu == 0)
        + u64::from(gpu_direct_verified && capabilities.nvidia_peer_memory_module.is_none());

    let topology_affinity_known = gpu_root_complex.is_some() && rdma_with_pci_path > 0;

    FabricTopologySummary {
        status: FabricTopologyStatus::Ok,
        evidence_source: "linux_sysfs",
        gpu_pci_bus_id,
        gpu_root_complex,
        gpu_numa_node,
        rdma_devices,
        rdma_with_pci_path,
        rdma_same_root_as_gpu,
        rdma_same_numa_as_gpu,
        rdma_affinity,
        iommu_group_count: capabilities.topology.iommu_group_count,
        iommu_mode: capabilities.topology.iommu_mode.clone(),
        rdma_core_loaded: capabilities.rdma_core_loaded,
        mlx5_core_loaded: capabilities.mlx5_core_loaded,
        peer_memory_module: capabilities.nvidia_peer_memory_module.clone(),
        gpu_direct_rdma: capabilities.gpu_direct_rdma,
        pinned_host_staging: capabilities.pinned_host_staging,
        gpu_direct_verified,
        degraded_to_pinned_host,
        topology_affinity_known,
        false_direct_claims,
        error: None,
    }
}
