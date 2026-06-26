#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransferMode {
    Decode,
    Prefill,
}

impl TransferMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Prefill => "prefill",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportPathKind {
    TrueGpuDirectRdma,
    OptimizedPinnedHostBounce,
    CpuProducedBoundary,
    MappedPinnedHostWrite,
}

impl TransportPathKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TrueGpuDirectRdma => "true_gpu_direct_rdma",
            Self::OptimizedPinnedHostBounce => "optimized_pinned_host_bounce",
            Self::CpuProducedBoundary => "cpu_produced_boundary",
            Self::MappedPinnedHostWrite => "mapped_pinned_host_write",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportPathClass {
    GpuDirect,
    HostStaged,
    CpuProduced,
    MappedPinned,
}

impl TransportPathClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GpuDirect => "GPU_DIRECT",
            Self::HostStaged => "HOST_STAGED",
            Self::CpuProduced => "CPU_PRODUCED",
            Self::MappedPinned => "MAPPED_PINNED",
        }
    }
}
