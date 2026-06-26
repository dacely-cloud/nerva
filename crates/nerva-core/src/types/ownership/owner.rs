use crate::types::id::device::DeviceOrdinal;
use crate::types::id::transport::TransportDeviceId;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ExecutionOwner {
    Cpu,
    Gpu(DeviceOrdinal),
    Nic(TransportDeviceId),
    SharedReadOnly,
    PhaseTransition,
    None,
}
