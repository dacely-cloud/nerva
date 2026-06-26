use crate::types::id::device::DeviceOrdinal;
use crate::types::id::transaction::TransactionId;

use crate::types::memory::tier::MemoryTier;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BackendDeviceHandle {
    pub device: DeviceOrdinal,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BackendQueueContract {
    pub device: DeviceOrdinal,
    pub bounded: bool,
    pub stream_ordered: bool,
    pub preallocated: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BackendEventContract {
    pub device: DeviceOrdinal,
    pub timing_enabled: bool,
    pub preallocated: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BackendAllocationContract {
    pub tier: MemoryTier,
    pub bytes: usize,
    pub alignment: usize,
    pub preallocated: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BackendTransactionDescriptor {
    pub id: TransactionId,
    pub operation_count: usize,
    pub block_use_count: usize,
    pub graph_capturable: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BackendGraphExecContract {
    pub transaction: BackendTransactionDescriptor,
    pub replayable: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BackendSubmissionId(pub u64);
