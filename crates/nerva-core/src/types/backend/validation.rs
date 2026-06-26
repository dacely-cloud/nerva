use crate::types::backend::capabilities::DeviceBackendCapabilities;
use crate::types::backend::operation::{
    BackendAllocationContract, BackendGraphExecContract, BackendQueueContract,
};
use crate::types::memory::tier::MemoryTier;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BackendContractValidation {
    pub bootstrap_decode_ready: bool,
    pub device_allocation_ready: bool,
    pub pinned_allocation_ready: bool,
    pub queue_ready: bool,
    pub graph_ready: bool,
}

impl BackendContractValidation {
    pub const fn passed(self) -> bool {
        self.bootstrap_decode_ready
            && self.device_allocation_ready
            && self.pinned_allocation_ready
            && self.queue_ready
            && self.graph_ready
    }
}

pub fn validate_backend_contract(
    capabilities: &DeviceBackendCapabilities,
    device_allocation: BackendAllocationContract,
    pinned_allocation: BackendAllocationContract,
    queue: BackendQueueContract,
    graph: BackendGraphExecContract,
) -> BackendContractValidation {
    let device_allocation_ready = device_allocation.tier == MemoryTier::Vram
        && device_allocation.bytes > 0
        && device_allocation.alignment > 0
        && device_allocation.preallocated;
    let pinned_allocation_ready = pinned_allocation.tier == MemoryTier::PinnedDram
        && pinned_allocation.bytes > 0
        && pinned_allocation.alignment > 0
        && pinned_allocation.preallocated;
    let queue_ready = queue.bounded && queue.stream_ordered && queue.preallocated;
    let graph_ready = graph.replayable && graph.transaction.graph_capturable;

    BackendContractValidation {
        bootstrap_decode_ready: capabilities.supports_bootstrap_decode_contract(),
        device_allocation_ready,
        pinned_allocation_ready,
        queue_ready,
        graph_ready,
    }
}
