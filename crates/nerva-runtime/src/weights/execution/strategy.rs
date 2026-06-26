#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResidentWeightExecutionStrategy {
    CpuDram,
    GpuResident,
    GpuStaged,
    CpuExactFallback,
}

impl ResidentWeightExecutionStrategy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CpuDram => "cpu-dram",
            Self::GpuResident => "gpu-resident",
            Self::GpuStaged => "gpu-staged",
            Self::CpuExactFallback => "cpu-exact-fallback",
        }
    }
}
