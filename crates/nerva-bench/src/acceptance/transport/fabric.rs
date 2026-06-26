use nerva_runtime::engine::runtime::Runtime;
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
