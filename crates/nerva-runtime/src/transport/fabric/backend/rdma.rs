use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RdmaPortEvidence {
    pub total_ports: u64,
    pub active_ports: u64,
    pub roce_ports: u64,
    pub infiniband_ports: u64,
    pub unknown_link_layer_ports: u64,
    pub uverbs_devices: u64,
}

pub fn collect_rdma_port_evidence(device_names: &[String]) -> RdmaPortEvidence {
    let mut evidence = RdmaPortEvidence {
        uverbs_devices: count_prefixed_entries("/dev/infiniband", "uverbs"),
        ..RdmaPortEvidence::default()
    };
    for device in device_names {
        collect_device_ports(device, &mut evidence);
    }
    evidence
}

fn collect_device_ports(device: &str, evidence: &mut RdmaPortEvidence) {
    let ports = Path::new("/sys/class/infiniband")
        .join(device)
        .join("ports");
    let Ok(entries) = fs::read_dir(ports) else {
        return;
    };
    for entry in entries.flatten() {
        let port_path = entry.path();
        if !port_path.is_dir() {
            continue;
        }
        evidence.total_ports += 1;
        if read_trimmed(&port_path.join("state")).is_some_and(|state| rdma_port_active(&state)) {
            evidence.active_ports += 1;
        }
        match read_trimmed(&port_path.join("link_layer"))
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "ethernet" => evidence.roce_ports += 1,
            "infiniband" => evidence.infiniband_ports += 1,
            _ => evidence.unknown_link_layer_ports += 1,
        }
    }
}

fn rdma_port_active(state: &str) -> bool {
    let normalized = state.to_ascii_uppercase();
    normalized.starts_with("4:") || normalized == "ACTIVE" || normalized.ends_with(": ACTIVE")
}

fn count_prefixed_entries(path: &str, prefix: &str) -> u64 {
    fs::read_dir(path)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(prefix))
        })
        .count() as u64
}

fn read_trimmed(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
}

#[cfg(test)]
mod tests {
    #[test]
    fn rdma_port_active_accepts_kernel_state_formats() {
        assert!(crate::transport::fabric::backend::rdma::rdma_port_active(
            "4: ACTIVE"
        ));
        assert!(crate::transport::fabric::backend::rdma::rdma_port_active(
            "ACTIVE"
        ));
        assert!(!crate::transport::fabric::backend::rdma::rdma_port_active(
            "1: DOWN"
        ));
        assert!(!crate::transport::fabric::backend::rdma::rdma_port_active(
            "2: INIT"
        ));
    }
}
