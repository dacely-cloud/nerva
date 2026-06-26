use crate::capabilities::json::{json_escape, json_opt_bool, json_opt_string};
use crate::capabilities::snapshot::CapabilityState;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FabricTopologyStatus {
    Ok,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FabricRdmaAffinity {
    pub rdma_device: String,
    pub pci_bus_id: Option<String>,
    pub root_complex: Option<String>,
    pub numa_node: Option<i32>,
    pub netdevs: Vec<String>,
    pub same_root_as_gpu: bool,
    pub same_numa_as_gpu: bool,
}

impl FabricRdmaAffinity {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"rdma_device\":\"{}\",\"pci_bus_id\":{},\"root_complex\":{},\"numa_node\":{},\"netdevs\":{},\"same_root_as_gpu\":{},\"same_numa_as_gpu\":{}}}",
            json_escape(&self.rdma_device),
            json_opt_string(self.pci_bus_id.as_deref()),
            json_opt_string(self.root_complex.as_deref()),
            self.numa_node
                .map_or_else(|| "null".to_string(), |value| value.to_string()),
            json_string_array(&self.netdevs),
            self.same_root_as_gpu,
            self.same_numa_as_gpu,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FabricTopologySummary {
    pub status: FabricTopologyStatus,
    pub evidence_source: &'static str,
    pub gpu_pci_bus_id: Option<String>,
    pub gpu_root_complex: Option<String>,
    pub gpu_numa_node: Option<i32>,
    pub rdma_devices: u64,
    pub rdma_with_pci_path: u64,
    pub rdma_same_root_as_gpu: u64,
    pub rdma_same_numa_as_gpu: u64,
    pub rdma_affinity: Vec<FabricRdmaAffinity>,
    pub iommu_group_count: usize,
    pub iommu_mode: String,
    pub rdma_core_loaded: bool,
    pub mlx5_core_loaded: bool,
    pub peer_memory_module: Option<String>,
    pub dma_buf_export: CapabilityState,
    pub gpu_memory_export_verified: bool,
    pub cuda_vmm_posix_fd_export_verified: bool,
    pub cuda_gpu_direct_rdma_supported: Option<bool>,
    pub cuda_gpu_direct_rdma_with_vmm_supported: Option<bool>,
    pub gpu_direct_rdma: CapabilityState,
    pub pinned_host_staging: CapabilityState,
    pub gpu_direct_verified: bool,
    pub gpu_export_without_nic_direct: bool,
    pub degraded_to_pinned_host: bool,
    pub topology_affinity_known: bool,
    pub false_direct_claims: u64,
    pub error: Option<&'static str>,
}

impl FabricTopologySummary {
    pub fn passed(&self) -> bool {
        matches!(self.status, FabricTopologyStatus::Ok)
            && self.false_direct_claims == 0
            && self.rdma_devices == self.rdma_affinity.len() as u64
            && self.rdma_with_pci_path <= self.rdma_devices
            && (!self.gpu_direct_verified
                || (self.rdma_same_root_as_gpu > 0
                    && (self.peer_memory_module.is_some()
                        || self.cuda_gpu_direct_rdma_with_vmm_supported == Some(true))))
            && (self.gpu_export_without_nic_direct
                == (self.gpu_memory_export_verified && !self.gpu_direct_verified))
            && (self.gpu_direct_verified
                || (self.degraded_to_pinned_host
                    && self.pinned_host_staging != CapabilityState::Unsupported))
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            FabricTopologyStatus::Ok => "ok",
            FabricTopologyStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"evidence_source\":\"{}\",\"gpu_pci_bus_id\":{},\"gpu_root_complex\":{},\"gpu_numa_node\":{},\"rdma_devices\":{},\"rdma_with_pci_path\":{},\"rdma_same_root_as_gpu\":{},\"rdma_same_numa_as_gpu\":{},\"rdma_affinity\":{},\"iommu_group_count\":{},\"iommu_mode\":\"{}\",\"rdma_core_loaded\":{},\"mlx5_core_loaded\":{},\"peer_memory_module\":{},\"dma_buf_export\":\"{}\",\"gpu_memory_export_verified\":{},\"cuda_vmm_posix_fd_export_verified\":{},\"cuda_gpu_direct_rdma_supported\":{},\"cuda_gpu_direct_rdma_with_vmm_supported\":{},\"gpu_direct_rdma\":\"{}\",\"pinned_host_staging\":\"{}\",\"gpu_direct_verified\":{},\"gpu_export_without_nic_direct\":{},\"degraded_to_pinned_host\":{},\"topology_affinity_known\":{},\"false_direct_claims\":{},\"error\":{}}}",
            status,
            self.evidence_source,
            json_opt_string(self.gpu_pci_bus_id.as_deref()),
            json_opt_string(self.gpu_root_complex.as_deref()),
            self.gpu_numa_node
                .map_or_else(|| "null".to_string(), |value| value.to_string()),
            self.rdma_devices,
            self.rdma_with_pci_path,
            self.rdma_same_root_as_gpu,
            self.rdma_same_numa_as_gpu,
            rdma_affinity_to_json(&self.rdma_affinity),
            self.iommu_group_count,
            json_escape(&self.iommu_mode),
            self.rdma_core_loaded,
            self.mlx5_core_loaded,
            json_opt_string(self.peer_memory_module.as_deref()),
            self.dma_buf_export.as_str(),
            self.gpu_memory_export_verified,
            self.cuda_vmm_posix_fd_export_verified,
            json_opt_bool(self.cuda_gpu_direct_rdma_supported),
            json_opt_bool(self.cuda_gpu_direct_rdma_with_vmm_supported),
            self.gpu_direct_rdma.as_str(),
            self.pinned_host_staging.as_str(),
            self.gpu_direct_verified,
            self.gpu_export_without_nic_direct,
            self.degraded_to_pinned_host,
            self.topology_affinity_known,
            self.false_direct_claims,
            json_opt_string(self.error),
        )
    }
}

fn rdma_affinity_to_json(values: &[FabricRdmaAffinity]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&value.to_json());
    }
    out.push(']');
    out
}

fn json_string_array(values: &[String]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(value));
        out.push('"');
    }
    out.push(']');
    out
}
