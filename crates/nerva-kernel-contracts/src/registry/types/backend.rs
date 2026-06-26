#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelBackend {
    CpuReference,
    Cuda,
    Hip,
}
