use std::fs;
use std::path::Path;

use crate::capabilities::linux::{list_entry_names, read_trimmed_first_line};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PciDeviceLocation {
    pub pci_bus_id: Option<String>,
    pub root_complex: Option<String>,
    pub numa_node: Option<i32>,
}

pub(crate) fn pci_device_location(pci_bus_id: &str) -> PciDeviceLocation {
    let normalized = normalize_pci_bus_id(pci_bus_id);
    let canonical = normalized
        .as_deref()
        .and_then(|bus_id| fs::canonicalize(format!("/sys/bus/pci/devices/{bus_id}")).ok());
    let root_complex = canonical
        .as_deref()
        .and_then(root_complex_from_canonical_path);
    let numa_node = normalized.as_deref().and_then(read_pci_numa_node);
    PciDeviceLocation {
        pci_bus_id: normalized,
        root_complex,
        numa_node,
    }
}

pub(crate) fn rdma_device_pci_location(rdma_device: &str) -> PciDeviceLocation {
    let canonical = fs::canonicalize(format!("/sys/class/infiniband/{rdma_device}/device")).ok();
    let pci_bus_id = canonical
        .as_deref()
        .and_then(pci_bus_id_from_canonical_path);
    let root_complex = canonical
        .as_deref()
        .and_then(root_complex_from_canonical_path);
    let numa_node = pci_bus_id.as_deref().and_then(read_pci_numa_node);
    PciDeviceLocation {
        pci_bus_id,
        root_complex,
        numa_node,
    }
}

pub(crate) fn rdma_netdevs(rdma_device: &str) -> Vec<String> {
    let mut netdevs = list_entry_names(&format!("/sys/class/infiniband/{rdma_device}/device/net"));
    netdevs.sort();
    netdevs
}

pub(crate) fn normalize_pci_bus_id(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase();
    let parts = value.split(':').collect::<Vec<_>>();
    if parts.len() != 3 {
        return None;
    }
    let domain = parts[0];
    let bus = parts[1];
    let slot_function = parts[2];
    if bus.len() != 2 || !bus.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return None;
    }
    if !valid_slot_function(slot_function) {
        return None;
    }
    let domain = if domain.len() >= 4 {
        &domain[domain.len() - 4..]
    } else {
        domain
    };
    if domain.len() != 4 || !domain.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return None;
    }
    Some(format!("{domain}:{bus}:{slot_function}"))
}

pub(crate) fn root_complex_from_canonical_path(path: &Path) -> Option<String> {
    path.components().find_map(|component| {
        let text = component.as_os_str().to_string_lossy();
        if text.starts_with("pci") {
            Some(text.into_owned())
        } else {
            None
        }
    })
}

fn pci_bus_id_from_canonical_path(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(normalize_pci_bus_id)
}

fn read_pci_numa_node(pci_bus_id: &str) -> Option<i32> {
    read_trimmed_first_line(&format!("/sys/bus/pci/devices/{pci_bus_id}/numa_node"))
        .and_then(|value| value.parse::<i32>().ok())
        .filter(|value| *value >= 0)
}

fn valid_slot_function(value: &str) -> bool {
    let mut parts = value.split('.');
    let Some(slot) = parts.next() else {
        return false;
    };
    let Some(function) = parts.next() else {
        return false;
    };
    parts.next().is_none()
        && slot.len() == 2
        && function.len() == 1
        && slot.chars().all(|ch| ch.is_ascii_hexdigit())
        && function.chars().all(|ch| ch.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::{normalize_pci_bus_id, root_complex_from_canonical_path};
    use std::path::PathBuf;

    #[test]
    fn normalizes_cuda_and_sysfs_pci_bus_ids() {
        assert_eq!(
            normalize_pci_bus_id("00000000:65:00.0").as_deref(),
            Some("0000:65:00.0")
        );
        assert_eq!(
            normalize_pci_bus_id("0000:B4:00.0").as_deref(),
            Some("0000:b4:00.0")
        );
        assert!(normalize_pci_bus_id("65:00.0").is_none());
        assert!(normalize_pci_bus_id("0000:65:00").is_none());
    }

    #[test]
    fn extracts_first_pci_root_complex_from_canonical_path() {
        let path = PathBuf::from("/sys/devices/pci0000:64/0000:64:00.0/0000:65:00.0");
        assert_eq!(
            root_complex_from_canonical_path(&path).as_deref(),
            Some("pci0000:64")
        );
    }
}
