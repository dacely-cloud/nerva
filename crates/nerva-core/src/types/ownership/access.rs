#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AccessPolicy {
    CpuOnly,
    GpuOnly,
    NicOnly,
    CpuGpuReadOnly,
    CpuThenGpu,
    GpuThenCpu,
    GpuThenNic,
    NicThenGpu,
    PhaseOwned,
}
