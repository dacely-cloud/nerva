use crate::types::error::{NervaError, Result};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HostArch {
    X86_64,
    Aarch64,
    Other,
}

pub fn host_arch() -> HostArch {
    #[cfg(target_arch = "x86_64")]
    {
        HostArch::X86_64
    }
    #[cfg(target_arch = "aarch64")]
    {
        HostArch::Aarch64
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        HostArch::Other
    }
}

pub fn ensure_supported_linux_host() -> Result<()> {
    match host_arch() {
        HostArch::X86_64 | HostArch::Aarch64 => Ok(()),
        arch => Err(NervaError::UnsupportedHost { arch }),
    }
}
