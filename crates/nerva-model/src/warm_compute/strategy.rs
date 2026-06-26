use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::ownership::owner::ExecutionOwner;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WarmComputeStrategy {
    CpuDram,
    GpuResident,
    GpuStaged,
    HybridSplit,
}

impl WarmComputeStrategy {
    pub const fn label(self) -> &'static str {
        match self {
            Self::CpuDram => "cpu-dram",
            Self::GpuResident => "gpu-resident",
            Self::GpuStaged => "gpu-staged",
            Self::HybridSplit => "hybrid-split",
        }
    }

    pub const fn executor(self) -> ExecutionOwner {
        match self {
            Self::CpuDram => ExecutionOwner::Cpu,
            Self::GpuResident | Self::GpuStaged | Self::HybridSplit => {
                ExecutionOwner::Gpu(DeviceOrdinal(0))
            }
        }
    }
}
