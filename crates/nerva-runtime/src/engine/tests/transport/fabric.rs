use crate::capabilities::snapshot::CapabilityState;
use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::fabric::backend::types::FabricBackendStatus;
use crate::transport::fabric::summary::FabricTopologyStatus;

#[test]
fn fabric_topology_probe_reports_sysfs_affinity_without_false_direct_claims() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_fabric_topology_probe();

    assert_eq!(summary.status, FabricTopologyStatus::Ok);
    assert_eq!(summary.evidence_source, "linux_sysfs");
    assert_eq!(summary.rdma_devices, summary.rdma_affinity.len() as u64);
    assert!(summary.rdma_with_pci_path <= summary.rdma_devices);
    assert_eq!(summary.false_direct_claims, 0);
    assert!(summary.gpu_direct_verified || summary.degraded_to_pinned_host);
    assert!(summary.passed());
    let json = summary.to_json();
    assert!(json.contains("\"evidence_source\":\"linux_sysfs\""));
    assert!(json.contains("\"rdma_affinity\""));
    assert!(json.contains("\"false_direct_claims\":0"));
}

#[test]
fn fabric_backend_probe_reports_explicit_rdma_dpdk_readiness() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_fabric_backend_probe();

    assert_eq!(summary.status, FabricBackendStatus::Ok);
    assert_eq!(summary.evidence_source, "linux_sysfs_pkg_config");
    assert!(summary.dpdk_shim_sources_present);
    assert_eq!(summary.false_direct_claims, 0);
    assert!(summary.rdma_ports >= summary.rdma_active_ports);
    assert_eq!(
        summary.rdma_ports,
        summary.rdma_roce_ports
            + summary.rdma_infiniband_ports
            + summary.rdma_unknown_link_layer_ports
    );
    if summary.rdma_pinned_host != CapabilityState::Unsupported {
        assert!(summary.rdma_active_ports > 0);
        assert!(summary.rdma_uverbs_devices > 0);
    }
    assert!(summary.backend_readiness.len() >= 6);
    assert_ne!(summary.kernel_udp_test, CapabilityState::Unsupported);
    assert_ne!(summary.tcp_control_only, CapabilityState::Unsupported);
    assert_eq!(
        summary.verified_direct_backends,
        summary
            .backend_readiness
            .iter()
            .filter(|entry| entry.direct_gpu_memory)
            .count() as u64
    );
    assert_eq!(
        summary.host_staged_backends,
        summary
            .backend_readiness
            .iter()
            .filter(|entry| entry.pinned_host_required)
            .count() as u64
    );
    assert_eq!(
        summary.unsupported_backends,
        summary
            .backend_readiness
            .iter()
            .filter(|entry| entry.capability == CapabilityState::Unsupported)
            .count() as u64
    );
    assert!(summary.passed());
    let json = summary.to_json();
    assert!(json.contains("\"evidence_source\":\"linux_sysfs_pkg_config\""));
    assert!(json.contains("\"rdma_active_ports\""));
    assert!(json.contains("\"rdma_uverbs_devices\""));
    assert!(json.contains("\"backend\":\"rdma_pinned_host\""));
    assert!(json.contains("\"backend\":\"dpdk_udp_pinned_host\""));
    assert!(json.contains("\"false_direct_claims\":0"));
}
