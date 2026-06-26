use crate::types::id::{DeviceOrdinal, TransportDeviceId};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ExecutionOwner {
    Cpu,
    Gpu(DeviceOrdinal),
    Nic(TransportDeviceId),
    SharedReadOnly,
    PhaseTransition,
    None,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CoherencePolicy {
    ExplicitVersioned,
    CoherentReadMostly,
    CoherentPhaseOwned,
    AtomicControlOnly,
}

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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MutationSemantics {
    Immutable,
    AppendOnly,
    SingleWriter,
    Ephemeral,
    AtomicControl,
}
