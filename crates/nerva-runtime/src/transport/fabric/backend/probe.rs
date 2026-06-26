use crate::capabilities::snapshot::{CapabilitySnapshot, CapabilityState};
use crate::transport::fabric::backend::linux::{
    dpdk_shim_sources_present, hugepages_total, module_loaded,
};
use crate::transport::fabric::backend::pkg_config::read_dpdk_pkg_config;
use crate::transport::fabric::backend::rdma::collect_rdma_port_evidence;
use crate::transport::fabric::backend::types::{
    FabricBackendReadiness, FabricBackendStatus, FabricBackendSummary,
};
use crate::transport::fabric::summary::FabricTopologySummary;

pub fn run_fabric_backend_probe(
    capabilities: &CapabilitySnapshot,
    topology: &FabricTopologySummary,
) -> FabricBackendSummary {
    let pkg_config = read_dpdk_pkg_config();
    let rdma = collect_rdma_port_evidence(&capabilities.topology.rdma_device_names);
    let dpdk_pkg_config = if pkg_config.present {
        CapabilityState::SupportedUnverified
    } else {
        CapabilityState::Unsupported
    };
    let rdma_pinned_host = if capabilities.topology.rdma_device_count > 0
        && rdma.active_ports > 0
        && rdma.uverbs_devices > 0
        && capabilities.rdma_core_loaded
        && capabilities.pinned_host_staging != CapabilityState::Unsupported
    {
        CapabilityState::SupportedUnverified
    } else {
        CapabilityState::Unsupported
    };
    let dpdk_udp_pinned_host = if pkg_config.present
        && capabilities.pinned_host_staging != CapabilityState::Unsupported
        && (capabilities.mlx5_core_loaded || pkg_config.mlx5_pmd_linked)
    {
        CapabilityState::SupportedUnverified
    } else {
        CapabilityState::Unsupported
    };
    let dpdk_udp_gpu = if !pkg_config.present {
        CapabilityState::Unsupported
    } else if topology.gpu_direct_verified && pkg_config.gpudev_linked {
        CapabilityState::SupportedUnverified
    } else {
        CapabilityState::DegradedToPinnedHost
    };
    let rdma_gpu_direct = capabilities.gpu_direct_rdma;
    let kernel_udp_test = CapabilityState::SupportedUnverified;
    let tcp_control_only = CapabilityState::SupportedUnverified;
    let mut backend_readiness = backend_readiness(
        topology,
        rdma_gpu_direct,
        rdma_pinned_host,
        dpdk_udp_gpu,
        dpdk_udp_pinned_host,
        kernel_udp_test,
        tcp_control_only,
    );

    backend_readiness.sort_by_key(|entry| entry.backend);

    let verified_direct_backends = backend_readiness
        .iter()
        .filter(|entry| entry.direct_gpu_memory)
        .count() as u64;
    let host_staged_backends = backend_readiness
        .iter()
        .filter(|entry| entry.pinned_host_required)
        .count() as u64;
    let unsupported_backends = backend_readiness
        .iter()
        .filter(|entry| entry.capability == CapabilityState::Unsupported)
        .count() as u64;
    let explicit_degradations = backend_readiness
        .iter()
        .filter(|entry| entry.capability == CapabilityState::DegradedToPinnedHost)
        .count() as u64;
    let false_direct_claims = backend_readiness
        .iter()
        .filter(|entry| {
            entry.direct_gpu_memory && entry.capability != CapabilityState::SupportedAndVerified
        })
        .count() as u64;

    FabricBackendSummary {
        status: FabricBackendStatus::Ok,
        evidence_source: "linux_sysfs_pkg_config",
        rdma_devices: capabilities.topology.rdma_device_count as u64,
        rdma_ports: rdma.total_ports,
        rdma_active_ports: rdma.active_ports,
        rdma_roce_ports: rdma.roce_ports,
        rdma_infiniband_ports: rdma.infiniband_ports,
        rdma_unknown_link_layer_ports: rdma.unknown_link_layer_ports,
        rdma_uverbs_devices: rdma.uverbs_devices,
        rdma_core_loaded: capabilities.rdma_core_loaded,
        mlx5_core_loaded: capabilities.mlx5_core_loaded,
        peer_memory_module: capabilities.nvidia_peer_memory_module.clone(),
        dpdk_shim_sources_present: dpdk_shim_sources_present(),
        dpdk_pkg_config,
        dpdk_pkg_config_version: pkg_config.version,
        dpdk_mlx5_pmd_linked: pkg_config.mlx5_pmd_linked,
        dpdk_gpudev_linked: pkg_config.gpudev_linked,
        vfio_pci_loaded: module_loaded("vfio_pci"),
        uio_pci_generic_loaded: module_loaded("uio_pci_generic"),
        igb_uio_loaded: module_loaded("igb_uio"),
        hugepages_total: hugepages_total(),
        dma_buf_export: capabilities.dma_buf_export,
        gpu_memory_export_verified: topology.gpu_memory_export_verified,
        cuda_vmm_posix_fd_export_verified: topology.cuda_vmm_posix_fd_export_verified,
        cuda_gpu_direct_rdma_supported: capabilities.cuda_gpu_direct_rdma_supported,
        cuda_gpu_direct_rdma_with_vmm_supported: capabilities
            .cuda_gpu_direct_rdma_with_vmm_supported,
        gpu_export_without_nic_direct: topology.gpu_export_without_nic_direct,
        rdma_gpu_direct,
        rdma_pinned_host,
        dpdk_udp_gpu,
        dpdk_udp_pinned_host,
        kernel_udp_test,
        tcp_control_only,
        verified_direct_backends,
        host_staged_backends,
        unsupported_backends,
        explicit_degradations,
        false_direct_claims,
        backend_readiness,
        error: None,
    }
}

fn backend_readiness(
    topology: &FabricTopologySummary,
    rdma_gpu_direct: CapabilityState,
    rdma_pinned_host: CapabilityState,
    dpdk_udp_gpu: CapabilityState,
    dpdk_udp_pinned_host: CapabilityState,
    kernel_udp_test: CapabilityState,
    tcp_control_only: CapabilityState,
) -> Vec<FabricBackendReadiness> {
    vec![
        FabricBackendReadiness {
            backend: "rdma_gpu_direct",
            capability: rdma_gpu_direct,
            evidence: "capability_probe_peer_memory_topology",
            direct_gpu_memory: topology.gpu_direct_verified,
            pinned_host_required: !topology.gpu_direct_verified,
        },
        FabricBackendReadiness {
            backend: "rdma_pinned_host",
            capability: rdma_pinned_host,
            evidence: "linux_sysfs_active_rdma_ports_uverbs",
            direct_gpu_memory: false,
            pinned_host_required: rdma_pinned_host != CapabilityState::Unsupported,
        },
        FabricBackendReadiness {
            backend: "dpdk_udp_gpu",
            capability: dpdk_udp_gpu,
            evidence: "pkg_config_libdpdk_gpudev_topology",
            direct_gpu_memory: false,
            pinned_host_required: dpdk_udp_gpu == CapabilityState::DegradedToPinnedHost,
        },
        FabricBackendReadiness {
            backend: "dpdk_udp_pinned_host",
            capability: dpdk_udp_pinned_host,
            evidence: "pkg_config_libdpdk_mlx5_pinned_host",
            direct_gpu_memory: false,
            pinned_host_required: dpdk_udp_pinned_host != CapabilityState::Unsupported,
        },
        FabricBackendReadiness {
            backend: "kernel_udp_test",
            capability: kernel_udp_test,
            evidence: "linux_kernel_network_stack",
            direct_gpu_memory: false,
            pinned_host_required: false,
        },
        FabricBackendReadiness {
            backend: "tcp_control_only",
            capability: tcp_control_only,
            evidence: "linux_kernel_network_stack_control_plane",
            direct_gpu_memory: false,
            pinned_host_required: false,
        },
    ]
}
