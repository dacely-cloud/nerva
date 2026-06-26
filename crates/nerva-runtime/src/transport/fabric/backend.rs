use std::path::Path;
use std::process::Command;

use crate::capabilities::json::{json_escape, json_opt_string};
use crate::capabilities::snapshot::{CapabilitySnapshot, CapabilityState};
use crate::transport::fabric::summary::FabricTopologySummary;

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
            && (self.rdma_devices == 0 || self.rdma_pinned_host != CapabilityState::Unsupported)
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
            "{{\"status\":\"{}\",\"evidence_source\":\"{}\",\"rdma_devices\":{},\"rdma_core_loaded\":{},\"mlx5_core_loaded\":{},\"peer_memory_module\":{},\"dpdk_shim_sources_present\":{},\"dpdk_pkg_config\":\"{}\",\"dpdk_pkg_config_version\":{},\"dpdk_mlx5_pmd_linked\":{},\"dpdk_gpudev_linked\":{},\"vfio_pci_loaded\":{},\"uio_pci_generic_loaded\":{},\"igb_uio_loaded\":{},\"hugepages_total\":{},\"rdma_gpu_direct\":\"{}\",\"rdma_pinned_host\":\"{}\",\"dpdk_udp_gpu\":\"{}\",\"dpdk_udp_pinned_host\":\"{}\",\"kernel_udp_test\":\"{}\",\"tcp_control_only\":\"{}\",\"verified_direct_backends\":{},\"host_staged_backends\":{},\"unsupported_backends\":{},\"explicit_degradations\":{},\"false_direct_claims\":{},\"backend_readiness\":{},\"error\":{}}}",
            status,
            self.evidence_source,
            self.rdma_devices,
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

pub fn run_fabric_backend_probe(
    capabilities: &CapabilitySnapshot,
    topology: &FabricTopologySummary,
) -> FabricBackendSummary {
    let pkg_config = dpdk_pkg_config();
    let dpdk_pkg_config = if pkg_config.present {
        CapabilityState::SupportedUnverified
    } else {
        CapabilityState::Unsupported
    };
    let rdma_pinned_host = if capabilities.topology.rdma_device_count > 0
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

    let mut backend_readiness = vec![
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
            evidence: "linux_sysfs_rdma_pinned_host",
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
    ];
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct DpdkPkgConfig {
    present: bool,
    version: Option<String>,
    mlx5_pmd_linked: bool,
    gpudev_linked: bool,
}

fn dpdk_pkg_config() -> DpdkPkgConfig {
    if !command_success("pkg-config", &["--exists", "libdpdk"]) {
        return DpdkPkgConfig::default();
    }
    let version = command_stdout("pkg-config", &["--modversion", "libdpdk"]);
    let libs = command_stdout("pkg-config", &["--libs", "libdpdk"]).unwrap_or_default();
    let static_libs =
        command_stdout("pkg-config", &["--libs", "--static", "libdpdk"]).unwrap_or_default();
    let link_flags = format!("{libs} {static_libs}");
    DpdkPkgConfig {
        present: true,
        version,
        mlx5_pmd_linked: link_flags.contains("mlx5") || link_flags.contains("rte_net_mlx5"),
        gpudev_linked: link_flags.contains("gpudev")
            || link_flags.contains("rte_gpudev")
            || link_flags.contains("rte_gpu"),
    }
}

fn command_success(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn module_loaded(name: &str) -> bool {
    Path::new("/sys/module").join(name).is_dir()
}

fn dpdk_shim_sources_present() -> bool {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shim_root = manifest.join("../../native/dpdk/dpdk-shim");
    shim_root.join("Cargo.toml").is_file()
        && shim_root.join("shim.c").is_file()
        && shim_root.join("wrapper.h").is_file()
        && shim_root.join("src/lib.rs").is_file()
}

fn hugepages_total() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    contents.lines().find_map(|line| {
        let value = line.strip_prefix("HugePages_Total:")?.trim();
        value.parse::<u64>().ok()
    })
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
