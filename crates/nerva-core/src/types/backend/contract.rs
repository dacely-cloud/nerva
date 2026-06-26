use crate::types::backend::capabilities::DeviceBackendCapabilities;
use crate::types::backend::operation::{BackendSubmissionId, BackendTransactionDescriptor};
use crate::types::error::Result;
use crate::types::id::device::DeviceOrdinal;

pub trait DeviceBackend {
    type Device;
    type Queue;
    type Event;
    type GraphExec;
    type DeviceAllocation;
    type PinnedAllocation;

    fn capabilities(&self) -> &DeviceBackendCapabilities;
    fn create_device(&self, id: DeviceOrdinal) -> Result<Self::Device>;
    fn create_queue(&self, device: &Self::Device) -> Result<Self::Queue>;
    fn create_event(&self, device: &Self::Device) -> Result<Self::Event>;
    fn allocate_device(
        &self,
        device: &Self::Device,
        bytes: usize,
        alignment: usize,
    ) -> Result<Self::DeviceAllocation>;
    fn allocate_pinned(&self, bytes: usize, alignment: usize) -> Result<Self::PinnedAllocation>;
    fn capture(&self, transaction: &BackendTransactionDescriptor) -> Result<Self::GraphExec>;
    fn submit(&mut self, executable: &Self::GraphExec) -> Result<BackendSubmissionId>;
}
