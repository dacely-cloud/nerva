use nerva_runtime::capabilities::snapshot::CapabilityState;
use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::transport::fabric::backend::types::FabricBackendStatus;
use nerva_runtime::transport::fabric::summary::FabricTopologyStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_fabric_topology(report: &mut AcceptanceReport, runtime: &Runtime) {
    let summary = runtime.run_fabric_topology_probe();
    report.push(
        "fabric_topology_affinity",
        matches!(summary.status, FabricTopologyStatus::Ok)
            && summary.passed()
            && summary.evidence_source == "linux_sysfs"
            && summary.false_direct_claims == 0
            && summary.rdma_devices == summary.rdma_affinity.len() as u64
            && (!summary.gpu_direct_verified || summary.rdma_same_root_as_gpu > 0)
            && (summary.gpu_direct_verified || summary.degraded_to_pinned_host),
        format!(
            "evidence={} gpu_pci={} gpu_root={} gpu_numa={} rdma_devices={} rdma_with_pci_path={} rdma_same_root={} rdma_same_numa={} iommu_groups={} iommu_mode={} rdma_core_loaded={} mlx5_core_loaded={} peer_memory_module={} gpu_direct_rdma={:?} pinned_host_staging={:?} gpu_direct_verified={} degraded_to_pinned_host={} topology_affinity_known={} false_direct_claims={}",
            summary.evidence_source,
            summary.gpu_pci_bus_id.as_deref().unwrap_or("none"),
            summary.gpu_root_complex.as_deref().unwrap_or("none"),
            summary
                .gpu_numa_node
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            summary.rdma_devices,
            summary.rdma_with_pci_path,
            summary.rdma_same_root_as_gpu,
            summary.rdma_same_numa_as_gpu,
            summary.iommu_group_count,
            summary.iommu_mode,
            summary.rdma_core_loaded,
            summary.mlx5_core_loaded,
            summary.peer_memory_module.as_deref().unwrap_or("none"),
            summary.gpu_direct_rdma,
            summary.pinned_host_staging,
            summary.gpu_direct_verified,
            summary.degraded_to_pinned_host,
            summary.topology_affinity_known,
            summary.false_direct_claims,
        ),
    );
}

pub(crate) fn push_fabric_backends(report: &mut AcceptanceReport, runtime: &Runtime) {
    let summary = runtime.run_fabric_backend_probe();
    report.push(
        "fabric_backend_capabilities",
        matches!(summary.status, FabricBackendStatus::Ok)
            && summary.passed()
            && summary.evidence_source == "linux_sysfs_pkg_config"
            && summary.false_direct_claims == 0
            && summary.backend_readiness.len() >= 6
            && summary.kernel_udp_test != CapabilityState::Unsupported
            && summary.tcp_control_only != CapabilityState::Unsupported,
        format!(
            "evidence={} rdma_devices={} rdma_core_loaded={} mlx5_core_loaded={} peer_memory_module={} dpdk_shim_sources_present={} dpdk_pkg_config={:?} dpdk_version={} dpdk_mlx5_pmd_linked={} dpdk_gpudev_linked={} vfio_pci_loaded={} uio_pci_generic_loaded={} igb_uio_loaded={} hugepages_total={} rdma_gpu_direct={:?} rdma_pinned_host={:?} dpdk_udp_gpu={:?} dpdk_udp_pinned_host={:?} verified_direct_backends={} host_staged_backends={} unsupported_backends={} explicit_degradations={} false_direct_claims={}",
            summary.evidence_source,
            summary.rdma_devices,
            summary.rdma_core_loaded,
            summary.mlx5_core_loaded,
            summary.peer_memory_module.as_deref().unwrap_or("none"),
            summary.dpdk_shim_sources_present,
            summary.dpdk_pkg_config,
            summary.dpdk_pkg_config_version.as_deref().unwrap_or("none"),
            summary.dpdk_mlx5_pmd_linked,
            summary.dpdk_gpudev_linked,
            summary.vfio_pci_loaded,
            summary.uio_pci_generic_loaded,
            summary.igb_uio_loaded,
            summary
                .hugepages_total
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            summary.rdma_gpu_direct,
            summary.rdma_pinned_host,
            summary.dpdk_udp_gpu,
            summary.dpdk_udp_pinned_host,
            summary.verified_direct_backends,
            summary.host_staged_backends,
            summary.unsupported_backends,
            summary.explicit_degradations,
            summary.false_direct_claims,
        ),
    );
}
