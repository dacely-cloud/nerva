use crate::capabilities::json::{json_escape, json_opt_string};
use crate::capabilities::snapshot::CapabilityState;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FabricBackendStatus {
    Ok,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FabricBackendReadiness {
    pub backend: &'static str,
    pub capability: CapabilityState,
    pub evidence: &'static str,
    pub direct_gpu_memory: bool,
    pub pinned_host_required: bool,
}

impl FabricBackendReadiness {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"backend\":\"{}\",\"capability\":\"{}\",\"evidence\":\"{}\",\"direct_gpu_memory\":{},\"pinned_host_required\":{}}}",
            self.backend,
            self.capability.as_str(),
            json_escape(self.evidence),
            self.direct_gpu_memory,
            self.pinned_host_required,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FabricBackendSummary {
    pub status: FabricBackendStatus,
    pub evidence_source: &'static str,
    pub rdma_devices: u64,
    pub rdma_ports: u64,
    pub rdma_active_ports: u64,
    pub rdma_roce_ports: u64,
    pub rdma_infiniband_ports: u64,
    pub rdma_unknown_link_layer_ports: u64,
    pub rdma_uverbs_devices: u64,
    pub rdma_core_loaded: bool,
    pub mlx5_core_loaded: bool,
    pub peer_memory_module: Option<String>,
    pub dpdk_shim_sources_present: bool,
    pub dpdk_pkg_config: CapabilityState,
    pub dpdk_pkg_config_version: Option<String>,
    pub dpdk_mlx5_pmd_linked: bool,
    pub dpdk_gpudev_linked: bool,
    pub vfio_pci_loaded: bool,
    pub uio_pci_generic_loaded: bool,
    pub igb_uio_loaded: bool,
    pub hugepages_total: Option<u64>,
    pub rdma_gpu_direct: CapabilityState,
    pub rdma_pinned_host: CapabilityState,
    pub dpdk_udp_gpu: CapabilityState,
    pub dpdk_udp_pinned_host: CapabilityState,
    pub kernel_udp_test: CapabilityState,
    pub tcp_control_only: CapabilityState,
    pub verified_direct_backends: u64,
    pub host_staged_backends: u64,
    pub unsupported_backends: u64,
    pub explicit_degradations: u64,
    pub false_direct_claims: u64,
    pub backend_readiness: Vec<FabricBackendReadiness>,
    pub error: Option<&'static str>,
}

impl FabricBackendSummary {
    pub fn passed(&self) -> bool {
        matches!(self.status, FabricBackendStatus::Ok)
            && self.false_direct_claims == 0
            && self.backend_readiness.len() >= 6
            && self.kernel_udp_test != CapabilityState::Unsupported
            && self.tcp_control_only != CapabilityState::Unsupported
            && self.rdma_ports >= self.rdma_active_ports
            && self.rdma_ports
                == self.rdma_roce_ports
                    + self.rdma_infiniband_ports
                    + self.rdma_unknown_link_layer_ports
            && (self.rdma_pinned_host == CapabilityState::Unsupported
                || (self.rdma_active_ports > 0 && self.rdma_uverbs_devices > 0))
            && (self.rdma_active_ports == 0
                || self.rdma_uverbs_devices == 0
                || self.rdma_pinned_host != CapabilityState::Unsupported)
            && self.dpdk_udp_gpu != CapabilityState::SupportedAndVerified
            && self.verified_direct_backends
                == self
                    .backend_readiness
                    .iter()
                    .filter(|entry| entry.direct_gpu_memory)
                    .count() as u64
            && self.host_staged_backends
                == self
                    .backend_readiness
                    .iter()
                    .filter(|entry| entry.pinned_host_required)
                    .count() as u64
            && self.unsupported_backends
                == self
                    .backend_readiness
                    .iter()
                    .filter(|entry| entry.capability == CapabilityState::Unsupported)
                    .count() as u64
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            FabricBackendStatus::Ok => "ok",
            FabricBackendStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"evidence_source\":\"{}\",\"rdma_devices\":{},\"rdma_ports\":{},\"rdma_active_ports\":{},\"rdma_roce_ports\":{},\"rdma_infiniband_ports\":{},\"rdma_unknown_link_layer_ports\":{},\"rdma_uverbs_devices\":{},\"rdma_core_loaded\":{},\"mlx5_core_loaded\":{},\"peer_memory_module\":{},\"dpdk_shim_sources_present\":{},\"dpdk_pkg_config\":\"{}\",\"dpdk_pkg_config_version\":{},\"dpdk_mlx5_pmd_linked\":{},\"dpdk_gpudev_linked\":{},\"vfio_pci_loaded\":{},\"uio_pci_generic_loaded\":{},\"igb_uio_loaded\":{},\"hugepages_total\":{},\"rdma_gpu_direct\":\"{}\",\"rdma_pinned_host\":\"{}\",\"dpdk_udp_gpu\":\"{}\",\"dpdk_udp_pinned_host\":\"{}\",\"kernel_udp_test\":\"{}\",\"tcp_control_only\":\"{}\",\"verified_direct_backends\":{},\"host_staged_backends\":{},\"unsupported_backends\":{},\"explicit_degradations\":{},\"false_direct_claims\":{},\"backend_readiness\":{},\"error\":{}}}",
            status,
            self.evidence_source,
            self.rdma_devices,
            self.rdma_ports,
            self.rdma_active_ports,
            self.rdma_roce_ports,
            self.rdma_infiniband_ports,
            self.rdma_unknown_link_layer_ports,
            self.rdma_uverbs_devices,
            self.rdma_core_loaded,
            self.mlx5_core_loaded,
            json_opt_string(self.peer_memory_module.as_deref()),
            self.dpdk_shim_sources_present,
            self.dpdk_pkg_config.as_str(),
            json_opt_string(self.dpdk_pkg_config_version.as_deref()),
            self.dpdk_mlx5_pmd_linked,
            self.dpdk_gpudev_linked,
            self.vfio_pci_loaded,
            self.uio_pci_generic_loaded,
            self.igb_uio_loaded,
            self.hugepages_total
                .map_or_else(|| "null".to_string(), |value| value.to_string()),
            self.rdma_gpu_direct.as_str(),
            self.rdma_pinned_host.as_str(),
            self.dpdk_udp_gpu.as_str(),
            self.dpdk_udp_pinned_host.as_str(),
            self.kernel_udp_test.as_str(),
            self.tcp_control_only.as_str(),
            self.verified_direct_backends,
            self.host_staged_backends,
            self.unsupported_backends,
            self.explicit_degradations,
            self.false_direct_claims,
            backend_readiness_to_json(&self.backend_readiness),
            json_opt_string(self.error),
        )
    }
}

fn backend_readiness_to_json(values: &[FabricBackendReadiness]) -> String {
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
