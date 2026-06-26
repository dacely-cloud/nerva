use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HipCapabilityEvidence {
    pub runtime_present: bool,
    pub runtime_version: Option<String>,
    pub amd_gpu_count: usize,
    pub kfd_present: bool,
    pub amdgpu_loaded: bool,
}

pub fn discover_hip_evidence() -> HipCapabilityEvidence {
    let runtime_version = rocm_version().or_else(hipcc_version);
    HipCapabilityEvidence {
        runtime_present: runtime_version.is_some() || hipcc_path().is_some(),
        runtime_version,
        amd_gpu_count: count_amd_drm_devices(),
        kfd_present: Path::new("/dev/kfd").exists(),
        amdgpu_loaded: Path::new("/sys/module/amdgpu").is_dir(),
    }
}

fn hipcc_path() -> Option<&'static str> {
    ["/opt/rocm/bin/hipcc", "/usr/bin/hipcc"]
        .into_iter()
        .find(|path| Path::new(path).is_file())
}

fn rocm_version() -> Option<String> {
    ["/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"]
        .into_iter()
        .find_map(read_nonempty_first_line)
}

fn hipcc_version() -> Option<String> {
    let path = hipcc_path()?;
    let output = Command::new(path).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find(|line| !line.trim().is_empty())
        .map(|line| line.trim().to_string())
}

fn count_amd_drm_devices() -> usize {
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                return false;
            };
            name.starts_with("card") && !name.contains('-') && drm_vendor_is_amd(entry.path())
        })
        .count()
}

fn drm_vendor_is_amd(card_path: std::path::PathBuf) -> bool {
    read_nonempty_first_line(card_path.join("device/vendor"))
        .is_some_and(|vendor| vendor.eq_ignore_ascii_case("0x1002"))
}

fn read_nonempty_first_line(path: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(path).ok()?.lines().find_map(|line| {
        let trimmed = line.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

pub(crate) fn hip_runtime_usable(evidence: &HipCapabilityEvidence) -> bool {
    evidence.runtime_present
        && evidence.amd_gpu_count > 0
        && evidence.kfd_present
        && evidence.amdgpu_loaded
}

#[cfg(test)]
mod tests {
    #[test]
    fn hip_runtime_usable_requires_runtime_gpu_kfd_and_driver() {
        let mut evidence = crate::capabilities::hip::HipCapabilityEvidence {
            runtime_present: true,
            runtime_version: Some("6.0".to_string()),
            amd_gpu_count: 1,
            kfd_present: true,
            amdgpu_loaded: true,
        };
        assert!(crate::capabilities::hip::hip_runtime_usable(&evidence));
        evidence.kfd_present = false;
        assert!(!crate::capabilities::hip::hip_runtime_usable(&evidence));
    }
}
