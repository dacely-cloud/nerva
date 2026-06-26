use crate::capabilities::json::{
    host_arch_to_str, json_opt_string, json_opt_usize, json_string_array, memory_fabric_to_str,
};
use nerva_core::types::arch::HostArch;
use nerva_core::types::memory::MemoryFabricKind;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CapabilityState {
    SupportedAndVerified,
    SupportedUnverified,
    Unsupported,
    DegradedToPinnedHost,
}

impl CapabilityState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SupportedAndVerified => "SUPPORTED_AND_VERIFIED",
            Self::SupportedUnverified => "SUPPORTED_UNVERIFIED",
            Self::Unsupported => "UNSUPPORTED",
            Self::DegradedToPinnedHost => "DEGRADED_TO_PINNED_HOST",
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TopologySnapshot {
    pub cpu_online: Option<String>,
    pub cpu_count: usize,
    pub numa_node_count: usize,
    pub pci_device_count: usize,
    pub pci_root_complex_count: usize,
    pub pci_bus_count: usize,
    pub pci_gpu_count: usize,
    pub pci_network_count: usize,
    pub pci_nvme_count: usize,
    pub block_device_count: usize,
    pub nvme_block_device_count: usize,
    pub rdma_device_count: usize,
    pub rdma_device_names: Vec<String>,
    pub rdma_netdev_links: Vec<String>,
    pub iommu_group_count: usize,
    pub iommu_mode: String,
    pub iommu_kernel_args: Option<String>,
}

impl TopologySnapshot {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"cpu_online\":{},\"cpu_count\":{},\"numa_node_count\":{},\"pci_device_count\":{},\"pci_root_complex_count\":{},\"pci_bus_count\":{},\"pci_gpu_count\":{},\"pci_network_count\":{},\"pci_nvme_count\":{},\"block_device_count\":{},\"nvme_block_device_count\":{},\"rdma_device_count\":{},\"rdma_device_names\":{},\"rdma_netdev_links\":{},\"iommu_group_count\":{},\"iommu_mode\":\"{}\",\"iommu_kernel_args\":{}}}",
            json_opt_string(self.cpu_online.as_deref()),
            self.cpu_count,
            self.numa_node_count,
            self.pci_device_count,
            self.pci_root_complex_count,
            self.pci_bus_count,
            self.pci_gpu_count,
            self.pci_network_count,
            self.pci_nvme_count,
            self.block_device_count,
            self.nvme_block_device_count,
            self.rdma_device_count,
            json_string_array(&self.rdma_device_names),
            json_string_array(&self.rdma_netdev_links),
            self.iommu_group_count,
            crate::capabilities::json::json_escape(&self.iommu_mode),
            json_opt_string(self.iommu_kernel_args.as_deref()),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilitySnapshot {
    pub host_arch: HostArch,
    pub target_os: &'static str,
    pub target_arch: &'static str,
    pub kernel_release: Option<String>,
    pub fabric: MemoryFabricKind,
    pub cuda: CapabilityState,
    pub cuda_status: &'static str,
    pub cuda_error: Option<String>,
    pub cuda_visible_devices: Option<String>,
    pub cuda_compute_capability: Option<String>,
    pub cuda_device_total_memory_bytes: Option<usize>,
    pub cuda_pci_bus_id: Option<String>,
    pub hip: CapabilityState,
    pub hip_visible_devices: Option<String>,
    pub nvidia_driver_version: Option<String>,
    pub rdma_core_loaded: bool,
    pub mlx5_core_loaded: bool,
    pub nvidia_peer_memory_module: Option<String>,
    pub pinned_host_staging: CapabilityState,
    pub gpu_direct_rdma: CapabilityState,
    pub amd_peerdirect: CapabilityState,
    pub dma_buf_export: CapabilityState,
    pub cxl: CapabilityState,
    pub topology: TopologySnapshot,
}

impl CapabilitySnapshot {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"host_arch\":\"{}\",\"target_os\":\"{}\",\"target_arch\":\"{}\",\"kernel_release\":{},\"fabric\":\"{}\",\"cuda\":\"{}\",\"cuda_status\":\"{}\",\"cuda_error\":{},\"cuda_visible_devices\":{},\"cuda_compute_capability\":{},\"cuda_device_total_memory_bytes\":{},\"cuda_pci_bus_id\":{},\"hip\":\"{}\",\"hip_visible_devices\":{},\"nvidia_driver_version\":{},\"rdma_core_loaded\":{},\"mlx5_core_loaded\":{},\"nvidia_peer_memory_module\":{},\"pinned_host_staging\":\"{}\",\"gpu_direct_rdma\":\"{}\",\"amd_peerdirect\":\"{}\",\"dma_buf_export\":\"{}\",\"cxl\":\"{}\",\"topology\":{}}}",
            host_arch_to_str(self.host_arch),
            self.target_os,
            self.target_arch,
            json_opt_string(self.kernel_release.as_deref()),
            memory_fabric_to_str(self.fabric),
            self.cuda.as_str(),
            self.cuda_status,
            json_opt_string(self.cuda_error.as_deref()),
            json_opt_string(self.cuda_visible_devices.as_deref()),
            json_opt_string(self.cuda_compute_capability.as_deref()),
            json_opt_usize(self.cuda_device_total_memory_bytes),
            json_opt_string(self.cuda_pci_bus_id.as_deref()),
            self.hip.as_str(),
            json_opt_string(self.hip_visible_devices.as_deref()),
            json_opt_string(self.nvidia_driver_version.as_deref()),
            self.rdma_core_loaded,
            self.mlx5_core_loaded,
            json_opt_string(self.nvidia_peer_memory_module.as_deref()),
            self.pinned_host_staging.as_str(),
            self.gpu_direct_rdma.as_str(),
            self.amd_peerdirect.as_str(),
            self.dma_buf_export.as_str(),
            self.cxl.as_str(),
            self.topology.to_json(),
        )
    }
}
