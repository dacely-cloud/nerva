use std::{env, fs};

use nerva_core::{HostArch, MemoryFabricKind, host_arch};

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
            json_escape(&self.iommu_mode),
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

pub fn cuda_smoke() -> nerva_cuda::CudaSmokeSummary {
    nerva_cuda::smoke()
}

pub fn discover_capabilities() -> CapabilitySnapshot {
    let cuda_smoke = cuda_smoke();
    let cuda = match cuda_smoke.status {
        nerva_cuda::SmokeStatus::Ok => CapabilityState::SupportedAndVerified,
        nerva_cuda::SmokeStatus::Unavailable | nerva_cuda::SmokeStatus::Failed => {
            CapabilityState::Unsupported
        }
    };
    let cuda_status = match cuda_smoke.status {
        nerva_cuda::SmokeStatus::Ok => "ok",
        nerva_cuda::SmokeStatus::Unavailable => "unavailable",
        nerva_cuda::SmokeStatus::Failed => "failed",
    };
    let cuda_compute_capability = cuda_compute_capability(&cuda_smoke);
    let cuda_device_total_memory_bytes = cuda_smoke.device_total_memory_bytes;
    let cuda_pci_bus_id = cuda_smoke.pci_bus_id.clone();
    let topology = discover_topology_snapshot();
    let rdma_core_loaded = module_loaded("ib_core");
    let mlx5_core_loaded = module_loaded("mlx5_core");
    let nvidia_peer_memory_module = detect_nvidia_peer_memory_module();
    let gpu_direct_rdma = gpu_direct_rdma_capability(
        cuda,
        topology.rdma_device_count,
        nvidia_peer_memory_module.as_deref(),
    );

    CapabilitySnapshot {
        host_arch: host_arch(),
        target_os: env::consts::OS,
        target_arch: env::consts::ARCH,
        kernel_release: read_trimmed_first_line("/proc/sys/kernel/osrelease"),
        fabric: MemoryFabricKind::DiscreteExplicit,
        cuda,
        cuda_status,
        cuda_error: cuda_smoke.error,
        cuda_visible_devices: env::var("CUDA_VISIBLE_DEVICES").ok(),
        cuda_compute_capability,
        cuda_device_total_memory_bytes,
        cuda_pci_bus_id,
        hip: CapabilityState::Unsupported,
        hip_visible_devices: env::var("HIP_VISIBLE_DEVICES").ok(),
        nvidia_driver_version: read_trimmed_first_line("/proc/driver/nvidia/version"),
        rdma_core_loaded,
        mlx5_core_loaded,
        nvidia_peer_memory_module,
        pinned_host_staging: CapabilityState::SupportedUnverified,
        gpu_direct_rdma,
        amd_peerdirect: CapabilityState::Unsupported,
        dma_buf_export: CapabilityState::Unsupported,
        cxl: CapabilityState::Unsupported,
        topology,
    }
}

pub fn discover_topology_snapshot() -> TopologySnapshot {
    let cpu_online = read_trimmed_first_line("/sys/devices/system/cpu/online");
    let cpu_count = cpu_online
        .as_deref()
        .and_then(count_linux_id_list)
        .filter(|count| *count > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        });
    let numa_node_count = read_trimmed_first_line("/sys/devices/system/node/online")
        .as_deref()
        .and_then(count_linux_id_list)
        .filter(|count| *count > 0)
        .unwrap_or_else(|| count_prefixed_entries("/sys/devices/system/node", "node").max(1));
    let pci = pci_class_counts("/sys/bus/pci/devices");
    let iommu_group_count = count_dirs("/sys/kernel/iommu_groups");
    let kernel_cmdline = read_trimmed_first_line("/proc/cmdline");
    let iommu_kernel_args = kernel_cmdline
        .as_deref()
        .and_then(extract_iommu_kernel_args);
    let iommu_mode = discover_iommu_mode(iommu_group_count, iommu_kernel_args.as_deref());
    let rdma_device_names = list_entry_names("/sys/class/infiniband");
    let rdma_netdev_links = rdma_netdev_links("/sys/class/infiniband", &rdma_device_names);

    TopologySnapshot {
        cpu_online,
        cpu_count,
        numa_node_count,
        pci_device_count: pci.total,
        pci_root_complex_count: count_prefixed_entries("/sys/devices", "pci"),
        pci_bus_count: count_entries("/sys/class/pci_bus"),
        pci_gpu_count: pci.gpu,
        pci_network_count: pci.network,
        pci_nvme_count: pci.nvme,
        block_device_count: count_entries("/sys/block"),
        nvme_block_device_count: count_prefixed_entries("/sys/block", "nvme"),
        rdma_device_count: rdma_device_names.len(),
        rdma_device_names,
        rdma_netdev_links,
        iommu_group_count,
        iommu_mode,
        iommu_kernel_args,
    }
}

fn cuda_compute_capability(summary: &nerva_cuda::CudaSmokeSummary) -> Option<String> {
    match (
        summary.compute_capability_major,
        summary.compute_capability_minor,
    ) {
        (Some(major), Some(minor)) => Some(format!("{major}.{minor}")),
        _ => None,
    }
}

fn module_loaded(name: &str) -> bool {
    fs::metadata(format!("/sys/module/{name}"))
        .map(|metadata| metadata.is_dir())
        .unwrap_or(false)
}

fn detect_nvidia_peer_memory_module() -> Option<String> {
    ["nvidia_peermem", "nv_peer_mem"]
        .into_iter()
        .find(|name| module_loaded(name))
        .map(ToOwned::to_owned)
}

pub(crate) fn gpu_direct_rdma_capability(
    cuda: CapabilityState,
    rdma_device_count: usize,
    nvidia_peer_memory_module: Option<&str>,
) -> CapabilityState {
    if cuda == CapabilityState::SupportedAndVerified
        && rdma_device_count > 0
        && nvidia_peer_memory_module.is_some()
    {
        CapabilityState::SupportedUnverified
    } else {
        CapabilityState::DegradedToPinnedHost
    }
}

pub(crate) fn extract_iommu_kernel_args(cmdline: &str) -> Option<String> {
    let args = cmdline
        .split_whitespace()
        .filter(|arg| arg.contains("iommu"))
        .collect::<Vec<_>>();
    (!args.is_empty()).then(|| args.join(" "))
}

pub(crate) fn discover_iommu_mode(
    iommu_group_count: usize,
    iommu_kernel_args: Option<&str>,
) -> String {
    let args = iommu_kernel_args.unwrap_or_default();
    if has_kernel_arg(args, &["iommu=off", "intel_iommu=off", "amd_iommu=off"]) {
        return "disabled_by_kernel_arg".to_string();
    }
    if iommu_group_count > 0 && has_kernel_arg(args, &["iommu=pt"]) {
        return "passthrough_groups_present".to_string();
    }
    if iommu_group_count > 0 {
        return "enabled_groups_present".to_string();
    }
    if has_kernel_arg(args, &["iommu=pt"]) {
        return "passthrough_requested".to_string();
    }
    if has_kernel_arg(args, &["iommu=on", "intel_iommu=on", "amd_iommu=on"]) {
        return "enabled_requested".to_string();
    }
    "not_detected".to_string()
}

fn has_kernel_arg(args: &str, candidates: &[&str]) -> bool {
    args.split_whitespace()
        .any(|arg| candidates.iter().any(|candidate| arg == *candidate))
}

fn rdma_netdev_links(root: &str, rdma_device_names: &[String]) -> Vec<String> {
    let mut links = Vec::new();
    for rdma in rdma_device_names {
        let netdev_path = format!("{root}/{rdma}/device/net");
        let netdevs = list_entry_names(&netdev_path);
        if netdevs.is_empty() {
            links.push(format!("{rdma}:"));
        } else {
            links.extend(netdevs.into_iter().map(|netdev| format!("{rdma}:{netdev}")));
        }
    }
    links.sort();
    links
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct PciClassCounts {
    total: usize,
    gpu: usize,
    network: usize,
    nvme: usize,
}

fn pci_class_counts(path: &str) -> PciClassCounts {
    let Ok(entries) = fs::read_dir(path) else {
        return PciClassCounts::default();
    };
    let mut counts = PciClassCounts::default();
    for entry in entries.flatten() {
        counts.total = counts.total.saturating_add(1);
        let class_path = entry.path().join("class");
        let Some(class) = read_trimmed_first_line(&class_path.to_string_lossy()) else {
            continue;
        };
        let Some(class_value) = parse_pci_class(&class) else {
            continue;
        };
        let base_class = ((class_value >> 16) & 0xff) as u8;
        let subclass = ((class_value >> 8) & 0xff) as u8;
        let programming_interface = (class_value & 0xff) as u8;
        if base_class == 0x03 {
            counts.gpu = counts.gpu.saturating_add(1);
        }
        if base_class == 0x02 {
            counts.network = counts.network.saturating_add(1);
        }
        if base_class == 0x01 && subclass == 0x08 && programming_interface == 0x02 {
            counts.nvme = counts.nvme.saturating_add(1);
        }
    }
    counts
}

pub(crate) fn parse_pci_class(value: &str) -> Option<u32> {
    u32::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok()
}

pub(crate) fn count_linux_id_list(value: &str) -> Option<usize> {
    let mut total = 0usize;
    for part in value.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let start = start.trim().parse::<usize>().ok()?;
            let end = end.trim().parse::<usize>().ok()?;
            if end < start {
                return None;
            }
            total = total.checked_add(end - start + 1)?;
        } else {
            part.parse::<usize>().ok()?;
            total = total.checked_add(1)?;
        }
    }
    Some(total)
}

fn count_entries(path: &str) -> usize {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries.flatten().count()
}

fn count_dirs(path: &str) -> usize {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
        .count()
}

fn count_prefixed_entries(path: &str, prefix: &str) -> usize {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(prefix))
        .count()
}

fn list_entry_names(path: &str) -> Vec<String> {
    let Ok(entries) = fs::read_dir(path) else {
        return Vec::new();
    };
    let mut names = entries
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn read_trimmed_first_line(path: &str) -> Option<String> {
    let contents = fs::read_to_string(path).ok()?;
    contents
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToOwned::to_owned)
}

fn json_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn host_arch_to_str(value: HostArch) -> &'static str {
    match value {
        HostArch::X86_64 => "x86_64",
        HostArch::Aarch64 => "aarch64",
        HostArch::Other => "other",
    }
}

fn memory_fabric_to_str(value: MemoryFabricKind) -> &'static str {
    match value {
        MemoryFabricKind::DiscreteExplicit => "DiscreteExplicit",
        MemoryFabricKind::UnifiedVirtualManaged => "UnifiedVirtualManaged",
        MemoryFabricKind::CoherentSharedPhysical => "CoherentSharedPhysical",
        MemoryFabricKind::CxlCoherentFabric => "CxlCoherentFabric",
    }
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", ch as u32));
            }
            ch => escaped.push(ch),
        }
    }
    escaped
}

fn json_opt_string(value: Option<&str>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", json_escape(value)),
    )
}

pub(crate) fn json_string_array(values: &[String]) -> String {
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
