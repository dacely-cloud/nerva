use std::fs;

use crate::capabilities::linux::{
    count_dirs, count_entries, count_linux_id_list, count_prefixed_entries, discover_iommu_mode,
    extract_iommu_kernel_args, list_entry_names, parse_pci_class, read_trimmed_first_line,
};
use crate::capabilities::snapshot::TopologySnapshot;

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
