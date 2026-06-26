#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!(
    "NERVA currently supports Linux only. Ubuntu x86_64 and aarch64 are the M0 host targets."
);

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

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct DeviceOrdinal(pub i32);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ResidentBlockId(pub u64);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MemoryTier {
    Vram,
    PinnedDram,
    Dram,
    Disk,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResidentBlockKind {
    Weight,
    KvCache,
    Activation,
    TokenState,
    SamplerState,
    Ledger,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentBlock {
    pub id: ResidentBlockId,
    pub kind: ResidentBlockKind,
    pub tier: MemoryTier,
    pub bytes: usize,
}

impl ResidentBlock {
    pub const fn new(
        id: ResidentBlockId,
        kind: ResidentBlockKind,
        tier: MemoryTier,
        bytes: usize,
    ) -> Self {
        Self {
            id,
            kind,
            tier,
            bytes,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NervaError {
    UnsupportedHost {
        arch: HostArch,
    },
    BackendUnavailable {
        backend: &'static str,
        reason: String,
    },
    AllocationFailed {
        bytes: usize,
        reason: String,
    },
    InvalidArgument {
        reason: String,
    },
}

pub type Result<T> = std::result::Result<T, NervaError>;

pub fn ensure_supported_linux_host() -> Result<()> {
    match host_arch() {
        HostArch::X86_64 | HostArch::Aarch64 => Ok(()),
        arch => Err(NervaError::UnsupportedHost { arch }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_host_gate_accepts_build_host_or_reports_other() {
        let result = ensure_supported_linux_host();
        if matches!(host_arch(), HostArch::Other) {
            assert!(matches!(result, Err(NervaError::UnsupportedHost { .. })));
        } else {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn resident_block_carries_identity_and_tier() {
        let block = ResidentBlock::new(
            ResidentBlockId(7),
            ResidentBlockKind::KvCache,
            MemoryTier::Vram,
            4096,
        );
        assert_eq!(block.id, ResidentBlockId(7));
        assert_eq!(block.tier, MemoryTier::Vram);
    }
}
