use std::fs;
use std::path::Path;

use crate::capabilities::snapshot::CapabilityState;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DmaBufExportEvidence {
    pub kernel_dma_buf_present: bool,
    pub nvidia_driver_present: bool,
    pub nvidia_capability_entries: usize,
    pub cuda_vmm_export_symbols_present: bool,
}

pub fn discover_dma_buf_export_evidence() -> DmaBufExportEvidence {
    DmaBufExportEvidence {
        kernel_dma_buf_present: kernel_dma_buf_present(),
        nvidia_driver_present: Path::new("/proc/driver/nvidia/version").is_file(),
        nvidia_capability_entries: count_entries(Path::new("/proc/driver/nvidia/capabilities")),
        cuda_vmm_export_symbols_present: cuda_vmm_export_symbols_present(),
    }
}

pub(crate) fn dma_buf_export_capability(evidence: &DmaBufExportEvidence) -> CapabilityState {
    if dma_buf_export_supported_unverified(evidence) {
        CapabilityState::SupportedUnverified
    } else {
        CapabilityState::Unsupported
    }
}

pub(crate) fn dma_buf_export_supported_unverified(evidence: &DmaBufExportEvidence) -> bool {
    evidence.kernel_dma_buf_present
        && evidence.nvidia_driver_present
        && evidence.cuda_vmm_export_symbols_present
}

fn kernel_dma_buf_present() -> bool {
    [
        "/sys/kernel/debug/dma_buf/bufinfo",
        "/sys/kernel/dma_buf/bufinfo",
        "/sys/module/dma_buf",
    ]
    .into_iter()
    .any(|path| Path::new(path).exists())
}

fn cuda_vmm_export_symbols_present() -> bool {
    let headers = [
        "/usr/local/cuda/include/cuda.h",
        "/usr/local/cuda/include/cudaTypedefs.h",
        "/usr/include/cuda.h",
        "/usr/include/cudaTypedefs.h",
    ];
    let mut has_export = false;
    let mut has_posix_fd = false;

    for header in headers {
        let Ok(content) = fs::read_to_string(header) else {
            continue;
        };
        has_export |= content.contains("cuMemExportToShareableHandle");
        has_posix_fd |= content.contains("CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR");
        if has_export && has_posix_fd {
            return true;
        }
    }
    false
}

fn count_entries(path: &Path) -> usize {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            1 + path.is_dir().then(|| count_entries(&path)).unwrap_or(0)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    #[test]
    fn dma_buf_export_requires_kernel_driver_and_cuda_symbols() {
        let mut evidence = crate::capabilities::dma_buf::DmaBufExportEvidence {
            kernel_dma_buf_present: true,
            nvidia_driver_present: true,
            nvidia_capability_entries: 3,
            cuda_vmm_export_symbols_present: true,
        };
        assert!(crate::capabilities::dma_buf::dma_buf_export_supported_unverified(&evidence));
        evidence.cuda_vmm_export_symbols_present = false;
        assert!(!crate::capabilities::dma_buf::dma_buf_export_supported_unverified(&evidence));
    }
}
