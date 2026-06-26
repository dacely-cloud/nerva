use crate::capabilities::snapshot::{CapabilitySnapshot, CapabilityState};
use crate::transport::path::types::TransferMode;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportPathRequest {
    pub source_tier: MemoryTier,
    pub destination_tier: MemoryTier,
    pub bytes: usize,
    pub mode: TransferMode,
    pub producer: ExecutionOwner,
    pub gpu_direct_rdma: CapabilityState,
    pub mapped_pinned_output: CapabilityState,
    pub pinned_host_staging: CapabilityState,
}

impl TransportPathRequest {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        source_tier: MemoryTier,
        destination_tier: MemoryTier,
        bytes: usize,
        mode: TransferMode,
        producer: ExecutionOwner,
        gpu_direct_rdma: CapabilityState,
        mapped_pinned_output: CapabilityState,
        pinned_host_staging: CapabilityState,
    ) -> Self {
        Self {
            source_tier,
            destination_tier,
            bytes,
            mode,
            producer,
            gpu_direct_rdma,
            mapped_pinned_output,
            pinned_host_staging,
        }
    }

    pub fn from_capabilities(
        source_tier: MemoryTier,
        destination_tier: MemoryTier,
        bytes: usize,
        mode: TransferMode,
        producer: ExecutionOwner,
        capabilities: &CapabilitySnapshot,
    ) -> Self {
        Self::new(
            source_tier,
            destination_tier,
            bytes,
            mode,
            producer,
            capabilities.gpu_direct_rdma,
            CapabilityState::Unsupported,
            capabilities.pinned_host_staging,
        )
    }
}
